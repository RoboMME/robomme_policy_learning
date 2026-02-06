import os
from omegaconf import DictConfig, OmegaConf, open_dict


def get_history_config(history_config: str | DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf
        
        if "realrobot" in history_config:
            history_config = omegaconf.OmegaConf.load(os.path.join("src/mme_vla_suite/models/config/mme_vla_suite_realrobot", history_config))
        else:
            history_config = omegaconf.OmegaConf.load(
                os.path.join("src/mme_vla_suite/models/config/robomme_sim", history_config)
            )
        return history_config
    elif isinstance(history_config, DictConfig):
        return history_config
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")


def parse_config(config: DictConfig):
    OmegaConf.set_struct(config, False)
    with open_dict(config):
        if config.integration_type == "input":
            config.memory_token_dim = 2048
        else:
            config.memory_token_dim = 1024

        perceptual_mem_config = config.perceptual_memory
        if perceptual_mem_config.token_sampling.type == "fast":
            static_token_per_image = perceptual_mem_config.token_sampling.fast_token_per_image
        elif perceptual_mem_config.token_sampling.type == "slow":
            static_token_per_image = perceptual_mem_config.token_sampling.slow_token_per_image
        elif perceptual_mem_config.token_sampling.type == "mid":
            static_token_per_image = perceptual_mem_config.token_sampling.mid_token_per_image
        else:
            raise ValueError(f"Not supported token sampling type: {perceptual_mem_config.token_sampling.type}")
        perceptual_mem_config.token_per_image = static_token_per_image


        recur_mem_config = config.recurrent_memory
        recur_mem_config.max_input_tokens = (
            recur_mem_config.max_recur_steps * config.num_views * config.token_per_image
        )
        assert (
            recur_mem_config.budget % (config.num_views * config.token_per_image) == 0
        )

        recur_mem_config.mini_batch_size = config.num_views * config.token_per_image

    return config
