import os

import omegaconf

from mme_vla_suite.models.config.presets import resolve_history_preset
from mme_vla_suite.models.config.schema import HistoryConfig, history_config_from_dict


def get_history_config(
    history_config: str | HistoryConfig | omegaconf.DictConfig | None,
) -> HistoryConfig | None:
    """Resolve training/checkpoint history config to a typed HistoryConfig."""
    if history_config in ("None", "none", None):
        return None
    if isinstance(history_config, HistoryConfig):
        return history_config
    if isinstance(history_config, omegaconf.DictConfig):
        resolved = omegaconf.OmegaConf.to_container(history_config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError(f"Expected dict from DictConfig, got {type(resolved)}")
        return history_config_from_dict(resolved)
    if isinstance(history_config, str):
        # Preset name (``*.yaml`` suffix optional) or absolute path to a custom YAML file
        if os.path.isabs(history_config) and os.path.isfile(history_config):
            raw = omegaconf.OmegaConf.load(history_config)
            resolved = omegaconf.OmegaConf.to_container(raw, resolve=True)
            if not isinstance(resolved, dict):
                raise TypeError(f"Expected dict from YAML, got {type(resolved)}")
            return history_config_from_dict(resolved)
        try:
            return resolve_history_preset(history_config)
        except KeyError as e:
            raise ValueError(str(e)) from e
    raise ValueError(f"Invalid history config: {history_config!r}")
