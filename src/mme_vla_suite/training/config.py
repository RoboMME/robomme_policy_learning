import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro


import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import numpy as np

from openpi.transforms import DataTransformFn, DataDict
import sentencepiece
import os

from mme_vla_suite.models.integration import history_pi0
from mme_vla_suite.policies.robomme_policy import RoboMMEInputs, RoboMMEOutputs
from mme_vla_suite.models.config.utils import get_history_config


ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""



class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self, 
        prompt: str, 
        state: np.ndarray | None = None,
        subgoal: str | None = None, 
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            if subgoal is None:
                # This is original Pi05 format, where the state is part of the discrete language input.
                state = np.clip(state, -1, 1)
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}; State: {state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            else:
                # this is used only for real robot symbolic variant. We do not use proprioceptive states in our experiment 
                subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
                full_prompt = f"Task: {cleaned_text}\nCurrent Subgoal: {subgoal}.\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
                
        elif subgoal is not None:
            # This is the subgoal format, where the subgoal is part of the language input. for both simple and grounded.
            subgoal = subgoal.strip().replace("_", " ").replace("\n", " ")
            full_prompt = f"Task: {cleaned_text};\nCurrent Subgoal: {subgoal};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        
        tokens_len = len(tokens)
        
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                # assert False, f"Token length ({len(tokens)}) exceeds max length ({self._max_len})!!!!!!!!!!!"
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)



@dataclasses.dataclass(frozen=True)
class TokenizePromptWithSymbolicMemory(DataTransformFn):
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = True
    symbolic_memory_type: str | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        
        if self.symbolic_memory_type is None:
            data.pop("simple_subgoal")
            data.pop("grounded_subgoal")
            return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
        
        if self.symbolic_memory_type == "simple_subgoal":
            simple_subgoal = data['simple_subgoal']
            symbolic_tokenized_prompt, symbolic_tokenized_prompt_mask = self.tokenizer.tokenize(prompt=prompt, subgoal=simple_subgoal, state=state)
        elif self.symbolic_memory_type == "grounded_subgoal":
            grounded_subgoal = data['grounded_subgoal']
            symbolic_tokenized_prompt, symbolic_tokenized_prompt_mask = self.tokenizer.tokenize(prompt=prompt, subgoal=grounded_subgoal, state=state)
        else:
            raise ValueError(f"Invalid symbolic memory type: {self.symbolic_memory_type}")
        
        data.pop("simple_subgoal")
        data.pop("grounded_subgoal")
                    
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks, 
                "symbolic_tokenized_prompt": symbolic_tokenized_prompt,
                "symbolic_tokenized_prompt_mask": symbolic_tokenized_prompt_mask}

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # This is MME-VLA set up
                symbolic_memory_type = None
                max_token_len = model_config.max_token_len
                
                if model_config.use_history and model_config.history_config is not None:
                    loaded_config = get_history_config(model_config.history_config)

                    if loaded_config.representation_type == "symbolic":
                        symbolic_memory_type = loaded_config.symbolic_memory.type
                        max_token_len *= 2 # it's enough for subgoals, no need to set into 512.
                
                print(f"max_token_len: {max_token_len}")
                
                return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            TokenizePromptWithSymbolicMemory(
                                PaligemmaTokenizer(max_token_len),
                                discrete_state_input=model_config.discrete_state_input,
                                symbolic_memory_type=symbolic_memory_type,
                            ),
                            _transforms.PadStatesAndActions(model_config.action_dim),
                        ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)



@dataclasses.dataclass(frozen=True)
class RoboMMEDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb", # (b, l, d1)
                        "static_pos_emb": "static_pos_emb", # (b, l, d2)
                        "static_state_emb": "static_state_emb", # (b, l, d3)
                        "static_mask": "static_mask", # (b, l)
                        # recurrent memory
                        "recur_image_emb": "recur_image_emb", # (b, t, v, p, d1)
                        "recur_pos_emb": "recur_pos_emb", # (b, t, v, p, d2)
                        "recur_state_emb": "recur_state_emb", # (b, t, d3)
                        "recur_mask": "recur_mask",
                        # symbolic memory
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[RoboMMEInputs(model_type=model_config.model_type)],
            outputs=[RoboMMEOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )




@dataclasses.dataclass(frozen=True)
class LeRobotMMEVLARealRobotDataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/left_shoulder_image": "left_shoulder_image",
                        "observation/right_shoulder_image": "right_shoulder_image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        # ---- New added keys ----
                        # perceptual memory
                        "static_image_emb": "static_image_emb", # (b, l, d1)
                        "static_pos_emb": "static_pos_emb", # (b, l, d2)
                        "static_state_emb": "static_state_emb", # (b, l, d3)
                        "static_mask": "static_mask", # (b, l)
                        # recurrent memory
                        "recur_image_emb": "recur_image_emb", # (b, t, v, p, d1)
                        "recur_pos_emb": "recur_pos_emb", # (b, t, v, p, d2)
                        "recur_state_emb": "recur_state_emb", # (b, t, d3)
                        "recur_mask": "recur_mask",
                        # symbolic memory
                        "simple_subgoal": "simple_subgoal",
                        "grounded_subgoal": "grounded_subgoal",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[RoboMMERealRobotInputs(model_type=model_config.model_type)],
            outputs=[RoboMMERealRobotOutputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

import openpi.shared.array_typing as at
from openpi.training.weight_loaders import WeightLoader, download, _merge_params
@dataclasses.dataclass(frozen=True)
class MMEVLAWeightLoader(WeightLoader):
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights. And our new weights
        return _merge_params(loaded_params, params, missing_regex=".*")


####################### RoboMME #######################


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "runs/assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "runs/ckpts"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
        
    resum_ckpt_id: int | None = None
    
    dataset_path: str = "data/robomme"

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


OPENPI_DATA_HOME = os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")

_CONFIGS = [
    TrainConfig(
        name="pi05_baseline",
        model=history_pi0.HistoryPi0Config(
            pi05=True, 
            action_horizon=20,
            use_history=False, 
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000, 
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="mme_vla_suite",
        model=history_pi0.HistoryPi0Config(
            pi05=True, 
            action_horizon=20,
            use_history=True, 
            history_config=None,
            discrete_state_input=False,
        ),
        data=RoboMMEDataConfig(
            repo_id=f"robomme",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=history_pi0.HistoryPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            os.path.join(OPENPI_DATA_HOME, "openpi-assets/checkpoints/pi05_base/params"),
        ),
        num_train_steps=80_000, 
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
