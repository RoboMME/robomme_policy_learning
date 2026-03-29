"""Typed history / memory configuration (replaces OmegaConf DictConfig for LSP-friendly access)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal, cast

# Where memory feeds the VLM: prefix tokens, FiLM-style, dedicated expert stream, or context extension.
IntegrationType = Literal["input", "modulation", "expert", "context"]
RepresentationType = Literal["perceptual", "recurrent", "symbolic"]
PerceptualMemoryKind = Literal["frame_sampling", "token_dropping"]
RecurrentMemoryKind = Literal["ttt", "rmt"]
SymbolicMemoryKind = Literal["simple_subgoal", "grounded_subgoal"]
PoolType = Literal["mean", "max"]


@dataclass(frozen=True)
class MemoryImgFeatureConfig:
    # Small MLP name for image side of mem encoder (typically identity / passthrough).
    net: str = "identity"
    # SigLIP patch embedding dim (e.g. 2048) fed into the memory encoder.
    input_dim: int = 2048


@dataclass(frozen=True)
class MemoryPosFeatureConfig:
    # Dim of 3D timestep / spatial position features concatenated to image tokens.
    input_dim: int = 768
    # Dim after encoding position (mem encoder output width for pos branch).
    hidden_dim: int = 768


@dataclass(frozen=True)
class MemoryStateFeatureConfig:
    # Proprio dim per timestep (e.g. 8 for Panda joints + gripper).
    input_dim: int = 8
    # Dim after encoding state for the memory pathway.
    hidden_dim: int = 512


@dataclass(frozen=True)
class MemoryFeatureConfig:
    # Sub-config for image / position / state branches into the perceptual or recurrent memory encoder.
    img: MemoryImgFeatureConfig = field(default_factory=MemoryImgFeatureConfig)
    pos: MemoryPosFeatureConfig = field(default_factory=MemoryPosFeatureConfig)
    state: MemoryStateFeatureConfig = field(default_factory=MemoryStateFeatureConfig)


@dataclass(frozen=True)
class PerceptualMemoryConfig:
    # frame_sampling: subsample timesteps; token_dropping: score and keep patches across history.
    type: PerceptualMemoryKind = "frame_sampling"


@dataclass(frozen=True)
class TTTConfig:
    # TTT variant label (e.g. linear head); used by recurrent TTT path.
    ttt_type: str = "linear"
    num_ttt_heads: int = 2
    base_lr: float = 0.01
    max_lr: float = 0.1
    qk_norm: bool = True
    v_norm: bool = True
    max_grad_norm: float = 5.0
    # Checkpointed scan: rematerialize every N mini-batches to save memory.
    remat_n_mini_batches: int = 4


@dataclass(frozen=True)
class RMTConfig:
    num_attn_heads: int = 8
    # GQA: fewer KV heads than Q heads.
    num_kv_heads: int = 1


@dataclass(frozen=True)
class RecurrentMemoryConfig:
    # ttt: test-time training stack; rmt: recurrent memory transformer.
    type: RecurrentMemoryKind = "ttt"
    hidden_dim: int = 512
    remat_n_mini_batches: int = 4
    # If True, recurrent module may emit debug stats during training.
    output_stats: bool = True
    # Max number of recurrent “memory” steps (time axis in recurrent tensors).
    max_recur_steps: int = 64
    # Max length of pre-execution video / demo phase before control starts (dataset alignment).
    max_pretraj_steps: int = 40
    ttt_config: TTTConfig = field(default_factory=TTTConfig)
    rmt_config: RMTConfig = field(default_factory=RMTConfig)
    # Upper bound on token length processed inside the recurrent block (chunking cap).
    max_input_tokens: int = 4096
    # Micro-batch size for internal recurrent scan / TTT steps.
    mini_batch_size: int = 64


@dataclass(frozen=True)
class SymbolicMemoryConfig:
    # Which text channel is appended to the language input (simple vs grounded subgoals).
    type: SymbolicMemoryKind = "simple_subgoal"
    # Optional override for symbolic token budget; None means use top-level HistoryConfig.budget.
    budget: int | None = None


@dataclass(frozen=True)
class HistoryConfig:
    """RoboMME history / memory settings (see field comments; presets live in ``presets.py``)."""

    # Total memory token slots for static perceptual buffer (after flattening frames × tokens per frame).
    budget: int = 512
    # Number of camera views stored per timestep in memory (e.g. 1 = single fixed camera).
    num_views: int = 1
    # Pooled SigLIP tokens per frame per view (e.g. 16 = 4×4 grid, 64 = 8×8); keys precomputed feature files.
    token_per_image: int = 64
    # Replenish cadence for memory / obs streaming (16): paired with model action_horizon=20 (execute 16, predict 20).
    streaming_obs_horizon: int = 16
    # When downsampling patch grids in MemoryBuffer, use mean or max pool.
    pool_type: PoolType = "mean"
    # Concatenate encoded 3D position embeddings to image memory tokens.
    use_pos_emb: bool = True
    # Concatenate encoded proprio to memory tokens.
    use_state_emb: bool = False
    # Dims and small nets for memory encoder inputs (image / pos / state).
    memory_feature: MemoryFeatureConfig = field(default_factory=MemoryFeatureConfig)
    # How memory injects into PaliGemma: extra prefix tokens, adapters, or side expert stream.
    integration_type: IntegrationType = "input"
    # Width of memory tokens passed into the LM (2048 context, 1024 modul/expert presets).
    memory_token_dim: int = 2048
    # perceptual: static/recurrent visual memory; recurrent: RMT/TTT stack; symbolic: text subgoals only.
    representation_type: RepresentationType = "perceptual"
    # Set when representation_type == perceptual (frame sampling vs token dropping).
    perceptual_memory: PerceptualMemoryConfig | None = None
    # Set when representation_type == recurrent (TTT vs RMT and their hyperparameters).
    recurrent_memory: RecurrentMemoryConfig | None = None
    # Set when representation_type == symbolic (subgoal strings in the language channel).
    symbolic_memory: SymbolicMemoryConfig | None = None


@dataclass(frozen=True)
class PastFramesConfig:
    """Load past execution steps for dense history / auxiliary losses.

    For current ``step_idx``, loads timesteps ``step_idx - stride``, ``step_idx - 2*stride``, ...,
    ``step_idx - num_past * stride`` (index ``0`` is the most recent past step).

    Requires ``meta/episode_step_to_sample_idx.json`` from ``scripts/build_episode_step_index.py``.
    """

    num_past: int
    stride: int

    def __post_init__(self) -> None:
        if self.num_past < 1:
            raise ValueError("num_past must be >= 1")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")


def _memory_feature_from_dict(mf: dict[str, Any]) -> MemoryFeatureConfig:
    img = mf.get("img", {})
    pos = mf.get("pos", {})
    st = mf.get("state", {})
    return MemoryFeatureConfig(
        img=MemoryImgFeatureConfig(
            net=str(img.get("net", "identity")),
            input_dim=int(img.get("input_dim", 2048)),
        ),
        pos=MemoryPosFeatureConfig(
            input_dim=int(pos.get("input_dim", 768)),
            hidden_dim=int(pos.get("hidden_dim", 768)),
        ),
        state=MemoryStateFeatureConfig(
            input_dim=int(st.get("input_dim", 8)),
            hidden_dim=int(st.get("hidden_dim", 512)),
        ),
    )


def _recurrent_memory_from_dict(rm: dict[str, Any]) -> RecurrentMemoryConfig:
    base = RecurrentMemoryConfig()
    ttc = dataclasses.replace(TTTConfig(), **dict(rm.get("ttt_config", {})))
    rmtc = dataclasses.replace(RMTConfig(), **dict(rm.get("rmt_config", {})))
    flat: dict[str, Any] = {f.name: getattr(base, f.name) for f in dataclasses.fields(RecurrentMemoryConfig)}
    for k, v in rm.items():
        if k in ("ttt_config", "rmt_config"):
            continue
        flat[k] = v
    flat["ttt_config"] = ttc
    flat["rmt_config"] = rmtc
    return RecurrentMemoryConfig(**flat)


def history_config_from_dict(d: dict[str, Any]) -> HistoryConfig:
    """Build HistoryConfig from a plain dict (e.g. OmegaConf resolved to container)."""
    memory_feature = _memory_feature_from_dict(dict(d.get("memory_feature", {})))
    repr_type = str(d.get("representation_type", "perceptual"))
    if repr_type not in ("perceptual", "recurrent", "symbolic"):
        raise ValueError(f"Invalid representation_type: {repr_type}")

    perceptual: PerceptualMemoryConfig | None = None
    recurrent: RecurrentMemoryConfig | None = None
    symbolic: SymbolicMemoryConfig | None = None

    if repr_type == "perceptual":
        pm = dict(d.get("perceptual_memory", {}))
        pt = str(pm.get("type", "frame_sampling"))
        if pt not in ("frame_sampling", "token_dropping"):
            raise ValueError(f"Invalid perceptual_memory.type: {pt}")
        perceptual = PerceptualMemoryConfig(type=cast(PerceptualMemoryKind, pt))
    elif repr_type == "recurrent":
        recurrent = _recurrent_memory_from_dict(dict(d.get("recurrent_memory", {})))
    else:
        sm = dict(d.get("symbolic_memory", {}))
        st = str(sm.get("type", "simple_subgoal"))
        if st not in ("simple_subgoal", "grounded_subgoal"):
            raise ValueError(f"Invalid symbolic_memory.type: {st}")
        budget = sm.get("budget")
        symbolic = SymbolicMemoryConfig(
            type=cast(SymbolicMemoryKind, st),
            budget=int(budget) if budget is not None else None,
        )

    pool = str(d.get("pool_type", "mean"))
    if pool not in ("mean", "max"):
        raise ValueError(f"Invalid pool_type: {pool}")

    integration = str(d.get("integration_type", "input"))
    if integration not in ("input", "modulation", "expert", "context"):
        raise ValueError(f"Invalid integration_type: {integration}")

    return HistoryConfig(
        budget=int(d.get("budget", 512)),
        num_views=int(d.get("num_views", 1)),
        token_per_image=int(d.get("token_per_image", 64)),
        streaming_obs_horizon=int(d.get("streaming_obs_horizon", 16)),
        pool_type=cast(PoolType, pool),
        use_pos_emb=bool(d.get("use_pos_emb", True)),
        use_state_emb=bool(d.get("use_state_emb", False)),
        memory_feature=memory_feature,
        integration_type=cast(IntegrationType, integration),
        memory_token_dim=int(d.get("memory_token_dim", 2048)),
        representation_type=cast(RepresentationType, repr_type),
        perceptual_memory=perceptual,
        recurrent_memory=recurrent,
        symbolic_memory=symbolic,
    )


def load_history_config_yaml(filename: str) -> HistoryConfig:
    """Resolve a named RoboMME preset (legacy name: same as former ``robomme/*.yaml`` keys)."""
    from mme_vla_suite.models.config.presets import resolve_history_preset

    return resolve_history_preset(filename)
