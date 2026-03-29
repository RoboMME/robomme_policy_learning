#!/usr/bin/env python3
"""Repack per-sample PKLs into per-episode npy arrays for mmap-friendly loading.

Creates ``episode_cache/episode_{E}/`` under the dataset root with:
  - ``images.npy``       (T, H, W, 3) uint8
  - ``wrist_images.npy`` (T, H, W, 3) uint8
  - ``states.npy``       (T, state_dim) float32
  - ``actions.npy``      (T, action_horizon, action_dim) float32
  - ``meta.json``        step_to_local, local_to_step, prompt, subgoals per step, etc.

Local index 0 corresponds to the **smallest** step_idx in that episode.
All arrays share the same T (first axis) ordering.

Usage:
  uv run scripts/build_episode_cache.py data/robomme_preprocessed_data
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--workers", type=int, default=8, help="Parallel pkl loaders")
    args = parser.parse_args()

    ds = args.dataset_path
    idx_path = ds / "meta" / "episode_step_to_sample_idx.json"
    if not idx_path.is_file():
        raise SystemExit(
            f"Missing {idx_path}; run build_episode_step_index.py first."
        )

    with open(idx_path, encoding="utf-8") as f:
        raw_idx: dict[str, int] = json.load(f)

    episodes: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key, sample_idx in raw_idx.items():
        epis, step = key.split("_", 1)
        episodes[int(epis)].append((int(step), sample_idx))

    for ep in episodes:
        episodes[ep].sort()

    cache_root = ds / "episode_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    data_dir = ds / "data"
    total_samples = sum(len(v) for v in episodes.values())
    pbar = tqdm(total=total_samples, desc="episode cache")

    for epis_idx in sorted(episodes):
        pairs = episodes[epis_idx]
        T = len(pairs)

        first_pkl_path = data_dir / f"{pairs[0][1]}.pkl"
        with open(first_pkl_path, "rb") as f:
            sample0 = pickle.load(f)
        img_shape = np.asarray(sample0["image"]).shape
        wrist_shape = np.asarray(sample0["wrist_image"]).shape
        state_dim = np.asarray(sample0["state"]).shape[-1]
        act = np.asarray(sample0["actions"])
        action_horizon, action_dim = act.shape

        images = np.zeros((T, *img_shape), dtype=np.uint8)
        wrist_images = np.zeros((T, *wrist_shape), dtype=np.uint8)
        states = np.zeros((T, state_dim), dtype=np.float32)
        actions = np.zeros((T, action_horizon, action_dim), dtype=np.float32)
        step_to_local: dict[str, int] = {}
        local_to_step: list[int] = []

        prompt: str | None = None
        simple_subgoals: list[str] = []
        grounded_subgoals: list[str] = []
        simple_subgoals_online: list[str] = []
        grounded_subgoals_online: list[str] = []

        for local_idx, (step_idx, sample_idx) in enumerate(pairs):
            pkl_path = data_dir / f"{sample_idx}.pkl"
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            images[local_idx] = d["image"]
            wrist_images[local_idx] = d["wrist_image"]
            states[local_idx] = np.asarray(d["state"], dtype=np.float32)
            a = np.asarray(d["actions"], dtype=np.float32)
            actions[local_idx] = a[:action_horizon]
            step_to_local[str(step_idx)] = local_idx
            local_to_step.append(step_idx)

            if prompt is None:
                prompt = d.get("prompt", "")
            simple_subgoals.append(d.get("simple_subgoal", ""))
            grounded_subgoals.append(d.get("grounded_subgoal", ""))
            simple_subgoals_online.append(d.get("simple_subgoal_online", ""))
            grounded_subgoals_online.append(d.get("grounded_subgoal_online", ""))

            pbar.update(1)

        ep_dir = cache_root / f"episode_{epis_idx}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.save(ep_dir / "images.npy", images)
        np.save(ep_dir / "wrist_images.npy", wrist_images)
        np.save(ep_dir / "states.npy", states)
        np.save(ep_dir / "actions.npy", actions)
        with open(ep_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "step_to_local": step_to_local,
                    "local_to_step": local_to_step,
                    "prompt": prompt,
                    "simple_subgoals": simple_subgoals,
                    "grounded_subgoals": grounded_subgoals,
                    "simple_subgoals_online": simple_subgoals_online,
                    "grounded_subgoals_online": grounded_subgoals_online,
                },
                f,
            )

    pbar.close()
    print(f"Done — {len(episodes)} episodes cached under {cache_root}")


if __name__ == "__main__":
    main()
