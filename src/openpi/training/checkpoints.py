from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            # "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        # "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                # "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])



from flax import nnx

def save_state_trainable_only(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    trainable_filter: nnx.filterlib.Filter,
    variables_filter: nnx.filterlib.Filter,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    ema_params = state.ema_params
    params = state.params
    train_state = dataclasses.replace(state, ema_params=None, params=nnx.statelib.State({}))
    
    if ema_params is not None:
        params_to_save = ema_params
    else:
        params_to_save = params.filter(trainable_filter)
                
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {
            "params":  params_to_save,
            "variables": params.filter(variables_filter),
        },
    }
    checkpoint_manager.save(step, items)
    

def restore_state_trainable_only(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None,
    trainable_filter: nnx.filterlib.Filter,
    variables_filter: nnx.filterlib.Filter,
    partial_params: at.Params,
    no_sharding: bool = False
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        ema_params = state.ema_params
        params = state.params
        train_state = dataclasses.replace(state, ema_params=None, params=nnx.statelib.State({}))
        
        if no_sharding:
            import numpy as np
            from flax import traverse_util
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
            path = checkpoint_manager.directory
            step = checkpoint_manager.latest_step()
            ckpt_path = path / str(step)

            with ocp.PyTreeCheckpointer() as ckptr:
                params_path = ckpt_path / "params"
                item = ckptr.metadata(params_path)
                restored_params = ckptr.restore(
                    params_path,
                    ocp.args.PyTreeRestore(
                        item=item,
                        restore_args=jax.tree.map(
                            lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray), item
                        ),
                    ),
                )
                
                flat_params = traverse_util.flatten_dict(restored_params)
                if all(kp[-1] == "value" for kp in flat_params):
                    flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
                restored_params = traverse_util.unflatten_dict(flat_params)

                partial_params = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), partial_params)        
                params.replace_by_pure_dict(partial_params) # restore the frozen params (bfloat16)
                params.replace_by_pure_dict(restored_params["params"]) # restore the trainable params
                if "variables" in restored_params:
                    params.replace_by_pure_dict(restored_params["variables"]) # restore the variables params
                
            
        else:
            trainable_params = params.filter(trainable_filter)
            variables_params = params.filter(variables_filter)
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": trainable_params, "variables": variables_params},
                },
            )
            partial_params = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), partial_params)        
            params.replace_by_pure_dict(partial_params) # restore the frozen params (bfloat16)
            params.replace_by_pure_dict(restored["params"]["params"].to_pure_dict()) # restore the trainable params
            params.replace_by_pure_dict(restored["params"]["variables"].to_pure_dict()) # restore the variables params
            
            
    if ema_params is not None:
        return dataclasses.replace(restored["train_state"], ema_params=restored["params"]["params"] , params=params)
    else:
        return dataclasses.replace(restored["train_state"], params=params)
