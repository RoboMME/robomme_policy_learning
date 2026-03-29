"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

import jax
import logging
from pathlib import Path
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader, TorchDataLoader,transform_dataset
import openpi.training.config as _config

from mme_vla_suite.training.dataset import RoboMMEDataset
from mme_vla_suite.training.episode_cache_dataset import EpisodeCacheDataset
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.config.schema import HistoryConfig, PastFramesConfig
from mme_vla_suite.models.config.utils import get_history_config



class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        self._total_samples = len(data_loader._data_loader.dataset)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield HistAugObservation.from_dict(batch), batch["actions"]
            

def create_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    history_config: str | HistoryConfig | None,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    past_frames_config: PastFramesConfig | None = None,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:
    
    history_config = get_history_config(history_config)

    dataset = RoboMMEDataset(
        dataset_path=dataset_path,
        data_config=data_config, 
        history_config=history_config, 
        action_horizon=action_horizon,
        past_frames_config=past_frames_config,
    )
    
    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")
    
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)


def create_dense_history_data_loader(
    dataset_path: str,
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    past_frames_config: PastFramesConfig | None = None,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[HistAugObservation, _model.Actions]]:
    """Data loader backed entirely by the episode cache (no .pkl reads)."""
    cache_root = Path(dataset_path) / "episode_cache"
    dataset = EpisodeCacheDataset(
        cache_root=cache_root,
        past_frames_config=past_frames_config,
        action_horizon=action_horizon,
    )

    dataset = transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats
    )

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)