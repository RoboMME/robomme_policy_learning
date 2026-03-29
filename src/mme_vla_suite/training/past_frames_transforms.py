"""Transforms for optional past-frame + action-chunk batches (see ``PastFramesConfig``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from openpi import transforms
from openpi.transforms import DataDict, pad_to_dim

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class PastFramesDeltaActions(transforms.DataTransformFn):
    """Same delta convention as ``DeltaActions``, using each past row's proprio state."""

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None or "past_actions" not in data or "past_state" not in data:
            return data
        past_state = np.asarray(data["past_state"])
        past_actions = np.asarray(data["past_actions"]).copy()
        valid = data.get("past_frame_mask")
        if valid is None:
            valid = np.ones(past_state.shape[0], dtype=bool)
        else:
            valid = np.asarray(valid, dtype=bool)
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        for i in range(past_state.shape[0]):
            if not valid[i]:
                continue
            past_actions[i, ..., :dims] -= np.expand_dims(
                np.where(mask, past_state[i, :dims], 0), axis=-2
            )
        data["past_actions"] = past_actions
        return data


@dataclasses.dataclass(frozen=True)
class PastFramesPadStatesAndActions(transforms.DataTransformFn):
    """Pad ``past_state`` / ``past_actions`` to ``model_action_dim`` like ``PadStatesAndActions``."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "past_state" in data:
            data["past_state"] = pad_to_dim(
                np.asarray(data["past_state"]), self.model_action_dim, axis=-1
            )
        if "past_actions" in data:
            data["past_actions"] = pad_to_dim(
                np.asarray(data["past_actions"]), self.model_action_dim, axis=-1
            )
        return data
