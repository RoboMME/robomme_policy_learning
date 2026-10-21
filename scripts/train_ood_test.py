import dataclasses
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from mme_vla_suite.training.config import get_config
from openpi.training.weight_loaders import CheckpointWeightLoader
from scripts import train_ood


def test_missing_memory_weights_are_rejected(monkeypatch):
    reference = {
        "backbone": np.ones((2,), dtype=np.float32),
        "memory": jax.ShapeDtypeStruct((2,), np.float32),
    }
    monkeypatch.setattr(CheckpointWeightLoader, "load", lambda _self, _params: reference)
    with pytest.raises(ValueError, match="missing model weights"):
        train_ood.StrictCheckpointWeightLoader("unused").load(reference)


@pytest.mark.parametrize("validate_only", ["0", "1"])
def test_entrypoint_invokes_training_once_or_validates(monkeypatch, tmp_path, validate_only):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/stats.json").write_text(json.dumps({"execution_samples": 128}))
    config = dataclasses.replace(
        get_config("mme_vla_suite"),
        exp_name="test-ood",
        checkpoint_base_dir=str(tmp_path / "ckpts"),
        dataset_path=str(dataset),
        weight_loader=CheckpointWeightLoader("source/79999/params"),
        batch_size=64,
        fsdp_devices=4,
    )
    calls = []
    monkeypatch.setattr(train_ood, "cli", lambda: config)
    monkeypatch.setattr(train_ood, "main", calls.append)
    monkeypatch.setattr(jax, "devices", lambda: [SimpleNamespace(platform="gpu")] * 4)
    monkeypatch.setenv("MME_VLA_VALIDATE_ONLY", validate_only)

    train_ood.run()

    if validate_only == "1":
        assert not calls
    else:
        assert len(calls) == 1
        launched = calls[0]
        assert launched.checkpoint_dir == tmp_path / "ckpts/mme_vla_suite_ood/test-ood"
        assert isinstance(launched.weight_loader, train_ood.StrictCheckpointWeightLoader)
        assert launched.weight_loader.params_path == "source/79999/params"
        assert not launched.resume
        assert not launched.overwrite
