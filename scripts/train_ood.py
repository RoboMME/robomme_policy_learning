"""Fine-tune OOD data from a complete checkpoint, invoking training only once.

The stock scripts/train.py entry point runs a tentative pass before its full
pass. This entry point uses main directly and puts the new experiment under
mme_vla_suite_ood, with a fresh optimizer and schedule starting at step zero.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

import jax

from mme_vla_suite.training.config import cli
from openpi.training.weight_loaders import CheckpointWeightLoader
from scripts.train import main


class StrictCheckpointWeightLoader(CheckpointWeightLoader):
    """Require pretrained weights for the backbone and the memory modules."""

    def load(self, params):
        loaded = super().load(params)
        if any(isinstance(value, jax.ShapeDtypeStruct) for value in jax.tree.leaves(loaded)):
            raise ValueError("Source checkpoint is missing model weights; refusing partial initialization")
        logging.info("Loaded all model weights from the source checkpoint")
        return loaded


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    config = cli()
    config = dataclasses.replace(
        config,
        name="mme_vla_suite_ood",
        weight_loader=StrictCheckpointWeightLoader(config.weight_loader.params_path),
    )
    stats = json.loads((Path(config.dataset_path) / "meta/stats.json").read_text())
    if stats["execution_samples"] < config.batch_size:
        raise ValueError("Preprocessed dataset has fewer execution samples than the global batch size")
    if config.checkpoint_dir.exists() and not (config.resume or config.overwrite):
        raise FileExistsError(f"Choose a new MME_VLA_EXP_NAME; output already exists: {config.checkpoint_dir}")
    print(
        json.dumps(
            {
                "source_params": config.weight_loader.params_path,
                "normalization_assets": config.data.assets.assets_dir,
                "dataset": config.dataset_path,
                "execution_samples": stats["execution_samples"],
                "output": str(config.checkpoint_dir),
                "batch_size": config.batch_size,
                "fsdp_devices": config.fsdp_devices,
                "num_train_steps": config.num_train_steps,
                "lr_schedule": dataclasses.asdict(config.lr_schedule),
                "save_interval": config.save_interval,
                "keep_period": config.keep_period,
                "history_config": config.model.history_config,
            },
            indent=2,
        ),
        flush=True,
    )
    if os.environ.get("MME_VLA_VALIDATE_ONLY") == "1":
        return
    devices = jax.devices()
    if len(devices) != config.fsdp_devices or any(device.platform != "gpu" for device in devices):
        raise RuntimeError(f"Expected {config.fsdp_devices} GPUs allocated by Slurm; got {devices}")
    logging.info("Training devices: %s", devices)
    main(config)


if __name__ == "__main__":
    run()
