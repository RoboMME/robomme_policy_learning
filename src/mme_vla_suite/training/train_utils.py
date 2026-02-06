
import optax
from flax import nnx, struct


import openpi.shared.array_typing as at
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training.optimizer import OptimizerConfig, LRScheduleConfig


@at.typecheck
@struct.dataclass
class DualOptimizerTrainState:
    # We train the origianl pi05 parames with samller lr
    # but train with newly added parameters (memory-related parameters) with larger lr
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    pretrained_opt_state: optax.OptState
    memory_opt_state: optax.OptState
    pretrained_tx: optax.GradientTransformation = struct.field(pytree_node=False)
    memory_tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


def create_dual_optimizer(
    optimizer: OptimizerConfig, 
    pretrained_lr_schedule: LRScheduleConfig, 
    memory_lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None
) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """Create two optimizers with different learning rates for pretrained and memory weights."""
    pretrained_lr = pretrained_lr_schedule.create()
    memory_lr = memory_lr_schedule.create()
    
    pretrained_tx = optimizer.create(pretrained_lr, weight_decay_mask=weight_decay_mask)
    memory_tx = optimizer.create(memory_lr, weight_decay_mask=weight_decay_mask)
    
    return pretrained_tx, memory_tx