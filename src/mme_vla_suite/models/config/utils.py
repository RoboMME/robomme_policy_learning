import os
from omegaconf import DictConfig


def get_history_config(history_config: str | DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf
        history_config = omegaconf.OmegaConf.load(
            os.path.join("src/mme_vla_suite/models/config/robomme", history_config))
        return history_config
    elif isinstance(history_config, DictConfig):
        return history_config
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")