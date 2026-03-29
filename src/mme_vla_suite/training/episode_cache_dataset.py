"""Dataset backed entirely by the episode cache — no .pkl reads.

Requires ``scripts/build_episode_cache.py`` to have been run first.
All arrays are loaded via ``np.load(..., mmap_mode='r')`` so only the
pages actually touched by a ``__getitem__`` call are read from disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from openpi.training.data_loader import Dataset
from mme_vla_suite.models.config.schema import PastFramesConfig

logger = logging.getLogger(__name__)


class _EpisodeArrays:
    """Lazy-loaded, mmap-backed arrays for a single episode."""

    __slots__ = ("_dir", "_meta", "_images", "_wrist_images", "_states", "_actions")

    def __init__(self, episode_dir: Path):
        self._dir = episode_dir
        self._meta: dict | None = None
        self._images: np.ndarray | None = None
        self._wrist_images: np.ndarray | None = None
        self._states: np.ndarray | None = None
        self._actions: np.ndarray | None = None

    @property
    def meta(self) -> dict:
        if self._meta is None:
            with open(self._dir / "meta.json") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self._dir / "images.npy", mmap_mode="r")
        return self._images

    @property
    def wrist_images(self) -> np.ndarray:
        if self._wrist_images is None:
            self._wrist_images = np.load(self._dir / "wrist_images.npy", mmap_mode="r")
        return self._wrist_images

    @property
    def states(self) -> np.ndarray:
        if self._states is None:
            self._states = np.load(self._dir / "states.npy", mmap_mode="r")
        return self._states

    @property
    def actions(self) -> np.ndarray:
        if self._actions is None:
            self._actions = np.load(self._dir / "actions.npy", mmap_mode="r")
        return self._actions

    def step_to_local(self, step_idx: int) -> int | None:
        return self.meta["step_to_local"].get(str(step_idx))

    @property
    def prompt(self) -> str:
        return self.meta.get("prompt", "")

    def subgoal(self, local_idx: int, kind: str = "simple_subgoal") -> str:
        arr = self.meta.get(f"{kind}s", [])
        if local_idx < len(arr):
            return arr[local_idx]
        return ""


class EpisodeCacheDataset(Dataset):
    """Random-access dataset that reads exclusively from the episode cache.

    Parameters
    ----------
    cache_root:
        Path to ``episode_cache/`` directory produced by ``build_episode_cache.py``.
    past_frames_config:
        If set, each sample includes ``num_past`` historical frames at the
        given ``stride`` (same episode).  Otherwise only the current timestep
        is returned.
    action_horizon:
        Number of action steps to include per chunk.  Arrays in the cache
        may be longer; they are truncated to this length.
    """

    def __init__(
        self,
        cache_root: str | Path,
        past_frames_config: PastFramesConfig | None = None,
        action_horizon: int = 20,
    ):
        self.cache_root = Path(cache_root)
        if not self.cache_root.is_dir():
            raise FileNotFoundError(f"Episode cache not found: {self.cache_root}")

        self.past_frames_config = past_frames_config
        self.action_horizon = action_horizon

        ep_dirs = sorted(self.cache_root.glob("episode_*"), key=lambda p: int(p.name.split("_")[1]))
        self._episodes: dict[int, _EpisodeArrays] = {}
        self._flat_index: list[tuple[int, int]] = []

        for ep_dir in ep_dirs:
            epis_idx = int(ep_dir.name.split("_")[1])
            ep = _EpisodeArrays(ep_dir)
            self._episodes[epis_idx] = ep
            for local_idx in range(len(ep.meta["local_to_step"])):
                self._flat_index.append((epis_idx, local_idx))

        logger.info(
            f"EpisodeCacheDataset: {len(self._episodes)} episodes, "
            f"{len(self._flat_index)} samples"
        )

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> dict:
        epis_idx, local_idx = self._flat_index[idx]
        ep = self._episodes[epis_idx]
        ah = self.action_horizon

        step_idx = ep.meta["local_to_step"][local_idx]

        data: dict = {
            "image": np.array(ep.images[local_idx]),
            "wrist_image": np.array(ep.wrist_images[local_idx]),
            "state": np.array(ep.states[local_idx], dtype=np.float32),
            "actions": np.array(ep.actions[local_idx, :ah], dtype=np.float32),
            "prompt": ep.prompt,
            "simple_subgoal": ep.subgoal(local_idx, "simple_subgoal"),
            "grounded_subgoal": ep.subgoal(local_idx, "grounded_subgoal"),
            "epis_idx": np.array([epis_idx], dtype=np.int32),
            "step_idx": np.array([step_idx], dtype=np.int32),
        }

        if self.past_frames_config is not None:
            data.update(self._gather_past_frames(ep, step_idx))

        return data

    # ------------------------------------------------------------------

    def _gather_past_frames(self, ep: _EpisodeArrays, step_idx: int) -> dict[str, np.ndarray]:
        cfg = self.past_frames_config
        assert cfg is not None
        N = cfg.num_past
        stride = cfg.stride
        ah = self.action_horizon

        img_shape = ep.images.shape[1:]
        wrist_shape = ep.wrist_images.shape[1:]
        state_dim = ep.states.shape[-1]
        act_shape = (ah, ep.actions.shape[-1])

        past_image = np.zeros((N, *img_shape), dtype=np.uint8)
        past_wrist = np.zeros((N, *wrist_shape), dtype=np.uint8)
        past_state = np.zeros((N, state_dim), dtype=np.float32)
        past_actions = np.zeros((N, *act_shape), dtype=np.float32)
        mask = np.zeros((N,), dtype=np.bool_)

        for i in range(N):
            t = step_idx - (i + 1) * stride
            loc = ep.step_to_local(t)
            if loc is None:
                continue
            past_image[i] = ep.images[loc]
            past_wrist[i] = ep.wrist_images[loc]
            past_state[i] = ep.states[loc]
            past_actions[i] = ep.actions[loc, :ah]
            mask[i] = True

        return {
            "past_image": past_image,
            "past_wrist_image": past_wrist,
            "past_state": past_state,
            "past_actions": past_actions,
            "past_frame_mask": mask,
        }
