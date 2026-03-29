"""Named RoboMME history presets (replaces robomme/*.yaml)."""

from __future__ import annotations

import dataclasses

from mme_vla_suite.models.config.schema import (
    HistoryConfig,
    IntegrationType,
    MemoryFeatureConfig,
    MemoryImgFeatureConfig,
    MemoryPosFeatureConfig,
    MemoryStateFeatureConfig,
    PerceptualMemoryConfig,
    RecurrentMemoryConfig,
    RMTConfig,
    SymbolicMemoryConfig,
    SymbolicMemoryKind,
    TTTConfig,
)

_ROBOMME_MEMORY = MemoryFeatureConfig(
    img=MemoryImgFeatureConfig(net="identity", input_dim=2048),
    pos=MemoryPosFeatureConfig(input_dim=768, hidden_dim=768),
    state=MemoryStateFeatureConfig(input_dim=8, hidden_dim=512),
)

_RECURRENT_SHARED = RecurrentMemoryConfig(
    type="rmt",
    hidden_dim=512,
    remat_n_mini_batches=4,
    output_stats=True,
    max_recur_steps=64,
    max_pretraj_steps=40,
    ttt_config=TTTConfig(),
    rmt_config=RMTConfig(),
    max_input_tokens=4096,
    mini_batch_size=64,
)

_RECURRENT_RMT = _RECURRENT_SHARED
_RECURRENT_TTT = dataclasses.replace(_RECURRENT_SHARED, type="ttt")


def _perceptual_frame_sampling(
    *,
    integration_type: IntegrationType,
    memory_token_dim: int,
) -> HistoryConfig:
    return HistoryConfig(
        budget=512,
        num_views=1,
        token_per_image=16,
        streaming_obs_horizon=16,
        pool_type="mean",
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=_ROBOMME_MEMORY,
        integration_type=integration_type,
        memory_token_dim=memory_token_dim,
        representation_type="perceptual",
        perceptual_memory=PerceptualMemoryConfig(type="frame_sampling"),
    )


def _perceptual_token_dropping(
    *,
    integration_type: IntegrationType,
    memory_token_dim: int,
) -> HistoryConfig:
    return HistoryConfig(
        budget=512,
        num_views=1,
        token_per_image=64,
        streaming_obs_horizon=16,
        pool_type="mean",
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=_ROBOMME_MEMORY,
        integration_type=integration_type,
        memory_token_dim=memory_token_dim,
        representation_type="perceptual",
        perceptual_memory=PerceptualMemoryConfig(type="token_dropping"),
    )


def _recurrent(
    *,
    integration_type: IntegrationType,
    memory_token_dim: int,
    recurrent: RecurrentMemoryConfig,
) -> HistoryConfig:
    return HistoryConfig(
        budget=512,
        num_views=1,
        token_per_image=64,
        streaming_obs_horizon=16,
        pool_type="mean",
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=_ROBOMME_MEMORY,
        integration_type=integration_type,
        memory_token_dim=memory_token_dim,
        representation_type="recurrent",
        recurrent_memory=recurrent,
    )


def _symbolic(*, memory_type: SymbolicMemoryKind) -> HistoryConfig:
    return HistoryConfig(
        budget=512,
        num_views=1,
        token_per_image=64,
        streaming_obs_horizon=16,
        pool_type="mean",
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=_ROBOMME_MEMORY,
        integration_type="context",
        memory_token_dim=2048,
        representation_type="symbolic",
        symbolic_memory=SymbolicMemoryConfig(type=memory_type),
    )


PERCEPTUAL_FRAMESAMP_CONTEXT = _perceptual_frame_sampling(
    integration_type="context", memory_token_dim=2048
)
PERCEPTUAL_FRAMESAMP_EXPERT = _perceptual_frame_sampling(
    integration_type="expert", memory_token_dim=1024
)
PERCEPTUAL_FRAMESAMP_MODUL = _perceptual_frame_sampling(
    integration_type="modulation", memory_token_dim=1024
)

PERCEPTUAL_TOKENDROP_CONTEXT = _perceptual_token_dropping(
    integration_type="context", memory_token_dim=2048
)
PERCEPTUAL_TOKENDROP_EXPERT = _perceptual_token_dropping(
    integration_type="expert", memory_token_dim=1024
)
PERCEPTUAL_TOKENDROP_MODUL = _perceptual_token_dropping(
    integration_type="modulation", memory_token_dim=1024
)

RECURRENT_RMT_CONTEXT = _recurrent(
    integration_type="context",
    memory_token_dim=2048,
    recurrent=_RECURRENT_RMT,
)
RECURRENT_RMT_EXPERT = _recurrent(
    integration_type="expert",
    memory_token_dim=1024,
    recurrent=_RECURRENT_RMT,
)
RECURRENT_RMT_MODUL = _recurrent(
    integration_type="modulation",
    memory_token_dim=1024,
    recurrent=_RECURRENT_RMT,
)

RECURRENT_TTT_CONTEXT = _recurrent(
    integration_type="context",
    memory_token_dim=2048,
    recurrent=_RECURRENT_TTT,
)
RECURRENT_TTT_EXPERT = _recurrent(
    integration_type="expert",
    memory_token_dim=1024,
    recurrent=_RECURRENT_TTT,
)
RECURRENT_TTT_MODUL = _recurrent(
    integration_type="modulation",
    memory_token_dim=1024,
    recurrent=_RECURRENT_TTT,
)

SYMBOLIC_SIMPLE_SUBGOAL = _symbolic(memory_type="simple_subgoal")
SYMBOLIC_GROUNDED_SUBGOAL = _symbolic(memory_type="grounded_subgoal")

PERCEPTUAL_DENSE_CONTEXT = HistoryConfig(
    budget=2048, # at 256 tokens per image, this is 8 images
    num_views=2,
    token_per_image=256, # default by SIGLIP, make sure this applies a no-op
    streaming_obs_horizon=None,
    pool_type=None,
    use_pos_emb=True,
    use_state_emb=False,
    memory_feature=_ROBOMME_MEMORY,
    integration_type="context",
    memory_token_dim=2048, # for main network context
    representation_type="perceptual",
    perceptual_memory=PerceptualMemoryConfig(type="dense"),
)




HISTORY_CONFIG_PRESETS: dict[str, HistoryConfig] = {
    "perceptual-dense-context.yaml": PERCEPTUAL_DENSE_CONTEXT,
    "perceptual-framesamp-context.yaml": PERCEPTUAL_FRAMESAMP_CONTEXT,
    "perceptual-framesamp-expert.yaml": PERCEPTUAL_FRAMESAMP_EXPERT,
    "perceptual-framesamp-modul.yaml": PERCEPTUAL_FRAMESAMP_MODUL,
    "perceptual-tokendrop-context.yaml": PERCEPTUAL_TOKENDROP_CONTEXT,
    "perceptual-tokendrop-expert.yaml": PERCEPTUAL_TOKENDROP_EXPERT,
    "perceptual-tokendrop-modul.yaml": PERCEPTUAL_TOKENDROP_MODUL,
    "recurrent-rmt-context.yaml": RECURRENT_RMT_CONTEXT,
    "recurrent-rmt-expert.yaml": RECURRENT_RMT_EXPERT,
    "recurrent-rmt-modul.yaml": RECURRENT_RMT_MODUL,
    "recurrent-ttt-context.yaml": RECURRENT_TTT_CONTEXT,
    "recurrent-ttt-expert.yaml": RECURRENT_TTT_EXPERT,
    "recurrent-ttt-modul.yaml": RECURRENT_TTT_MODUL,
    "symbolic-simple-subgoal.yaml": SYMBOLIC_SIMPLE_SUBGOAL,
    "symbolic-grounded-subgoal.yaml": SYMBOLIC_GROUNDED_SUBGOAL,
}


def resolve_history_preset(name: str) -> HistoryConfig:
    """Return a frozen preset by filename (with or without ``.yaml`` suffix)."""
    key = name.strip()
    if key in HISTORY_CONFIG_PRESETS:
        return HISTORY_CONFIG_PRESETS[key]
    if not key.endswith(".yaml"):
        key_yaml = f"{key}.yaml"
        if key_yaml in HISTORY_CONFIG_PRESETS:
            return HISTORY_CONFIG_PRESETS[key_yaml]
    raise KeyError(
        f"Unknown history preset {name!r}. Known: {sorted(HISTORY_CONFIG_PRESETS)!r}"
    )
