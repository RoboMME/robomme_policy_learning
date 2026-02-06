from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


@at.typecheck
def state_tree_to_info(
    state: nnx.State, 
    trainable_filter: nnx.filterlib.Filter,
    frozen_filter: nnx.filterlib.Filter,
) -> str:
    trainable_params = state.filter(trainable_filter)
    frozen_params = state.filter(frozen_filter)
    
    trainable_params_size = sum(x.size for x in jax.tree_util.tree_leaves(trainable_params)) / 1024 / 1024
    frozen_params_size = sum(x.size for x in jax.tree_util.tree_leaves(frozen_params)) / 1024 / 1024
    
    # Create separate info strings
    trainable_info = tree_to_info(trainable_params, lambda x: f"{x.shape}@{x.dtype} ({x.size / 1024 / 1024:.2f} MB)")
    frozen_info = tree_to_info(frozen_params, lambda x: f"{x.shape}@{x.dtype} ({x.size / 1024 / 1024:.2f} MB)")
    
    # Combine them
    result = []
    if trainable_info:
        result.append(f"Trainable parameters: {trainable_params_size:.2f} MB\n{trainable_info}")
    if frozen_info:
        result.append(f"Frozen parameters: {frozen_params_size:.2f} MB\n{frozen_info}")
    
    return "\n\n".join(result)
