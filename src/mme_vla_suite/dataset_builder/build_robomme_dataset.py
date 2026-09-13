"""Preprocess raw HDF5 RoboMME data into training-ready format.

Converts episodes to features, token-drop indices, and per-step pickle samples.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time

import cv2
import h5py
import imageio
import numpy as np

from mme_vla_suite.shared.mem_buffer import MemoryBuffer, create_dict

from mme_vla_suite.dataset_builder.robomme_h5_utils import (
    first_execution_step,
    remove_redundant_keyframes,
    resolve_subgoal,
)


# Action and state
ACTION_CHUNK_HORIZON = 20
JOINT_STATE_DIM = 8

# Token-dropping visualization (8x8 spatial grid, 32x32 patches)
NUM_SPATIAL_TOKENS = 64
SPATIAL_GRID_SIZE = 8
PATCH_HALF = 16
DROPPED_TOKEN_ALPHA = 0.3

# Frame-sampling visualization
FRAME_SAMPLE_COUNT = 32
VIS_FPS_ORIGINAL = 30
VIS_FPS_SAMPLED = 2
VIS_FPS_TOKENDROP = 10


def get_action_chunk(
    data: h5py.Group, idx: int, horizon: int = ACTION_CHUNK_HORIZON
) -> np.ndarray:
    """Return (horizon, action_dim) chunk; pads with last valid action at end of episode."""
    chunk: list[np.ndarray] = []
    last_action: np.ndarray | None = None
    for i in range(horizon):
        try:
            action = data[f"timestep_{idx + i}"]["action"]["joint_action"][()]
            chunk.append(action)
            last_action = action
        except (KeyError, IndexError):
            chunk.append(last_action)
    return np.stack(chunk, axis=0)


def _apply_dropped_token_overlay(
    img: np.ndarray, kept_per_frame: dict[int, tuple], frame_idx: int
) -> np.ndarray:
    """Mask out dropped spatial tokens (white overlay) for one frame."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for spatial_idx in range(NUM_SPATIAL_TOKENS):
        if spatial_idx in kept_per_frame[frame_idx][0]:
            continue
        h_center = spatial_idx // SPATIAL_GRID_SIZE * (PATCH_HALF * 2) + PATCH_HALF
        w_center = spatial_idx % SPATIAL_GRID_SIZE * (PATCH_HALF * 2) + PATCH_HALF
        cv2.rectangle(
            mask,
            (w_center - PATCH_HALF, h_center - PATCH_HALF),
            (w_center + PATCH_HALF, h_center + PATCH_HALF),
            255,
            -1,
        )
    out = img.copy()
    out[mask == 255] = out[mask == 255] * DROPPED_TOKEN_ALPHA + np.array([255, 255, 255]) * (1 - DROPPED_TOKEN_ALPHA)
    return out


def visualize_token_dropping(
    indices: list,
    videos: list[np.ndarray],
    output_dir: str,
    exec_start_idx: int,
    task_goal: str,
) -> None:
    """Write two MP4s: annotated (frame idx + demo border) and raw with token-drop overlay."""
    kept_per_frame = create_dict(sorted(indices))
    images_anno: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for frame_idx in kept_per_frame:
        img = videos[frame_idx].copy()
        img_anno = cv2.putText(
            img.copy(),
            str(frame_idx),
            (img.shape[1] // 2, img.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
        )
        img_anno = _apply_dropped_token_overlay(img_anno, kept_per_frame, frame_idx)
        img_masked = _apply_dropped_token_overlay(img, kept_per_frame, frame_idx)
        if frame_idx < exec_start_idx:
            cv2.rectangle(
                img_anno, (0, 0), (img_anno.shape[1], img_anno.shape[0]), (0, 0, 255), 4
            )
        images_anno.append(img_anno)
        images.append(img_masked)

    def _save(name: str, frames: list[np.ndarray]) -> None:
        path = os.path.join(output_dir, f"{name}_{task_goal}.mp4")
        imageio.mimsave(path, frames, fps=VIS_FPS_TOKENDROP)

    _save("token_dropping_anno", images_anno)
    _save("token_dropping", images)


def visualize_frame_sampling(
    videos: list[np.ndarray],
    output_dir: str,
    exec_start_idx: int,
    task_goal: str,
    keyframe_idxs: list[int],
) -> None:
    """Write original (demo border) and evenly sampled 8-frame MP4s."""
    with_border: list[np.ndarray] = []
    for i, img in enumerate(videos):
        frame = img.copy()
        if i < exec_start_idx:
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10
            )
        if i in keyframe_idxs:
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 10
            )
        with_border.append(frame)
        if i in keyframe_idxs:
            for _ in range(3): # halt for a while to show the keyframe
                with_border.append(frame)
            

    imageio.mimsave(
        os.path.join(output_dir, f"original_video_{task_goal}.mp4"),
        with_border,
        fps=VIS_FPS_ORIGINAL,
    )
    indices = np.linspace(0, len(with_border) - 1, FRAME_SAMPLE_COUNT, dtype=np.int32)
    sampled = [with_border[i] for i in indices]
    imageio.mimsave(
        os.path.join(output_dir, f"frame_sampling_{task_goal}.mp4"),
        sampled,
        fps=VIS_FPS_SAMPLED,
    )


class DatasetProcessor:
    """Converts raw HDF5 episodes to preprocessed features, token-drop indices, and execution samples."""

    def __init__(
        self,
        raw_data_path: str = "data/raw",
        preprocessed_data_path: str = "data/preprocessed",
        execution_horizon: int = 16,
        visualize: bool = False,
        max_episodes: int | None = None,
    ) -> None:
        self.raw_data_path = raw_data_path
        self.dataset_path = preprocessed_data_path
        self.execution_horizon = execution_horizon
        self.visualize = visualize
        self.max_episodes = max_episodes
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        os.makedirs(self.dataset_path, exist_ok=True)

        self.feature_path = os.path.join(self.dataset_path, "features")
        self.data_path = os.path.join(self.dataset_path, "data")
        self.meta_path = os.path.join(self.dataset_path, "meta")
        for p in (self.feature_path, self.data_path, self.meta_path):
            os.makedirs(p, exist_ok=True)

    def run(self) -> None:
        """Process all .h5 files; optionally cap episodes per file (default: process all)."""
        global_episode_idx = 0
        mem_buffer = MemoryBuffer(
            num_views=3,
            compute_token_drop_score=True,
            token_drop_stride=self.execution_horizon // 2,
            prepare_buffer=True,
        )
        exec_sample_id = 0
        total_sample_id = 0

        for fname in os.listdir(self.raw_data_path):
            if not fname.endswith(".h5"):
                continue
            print(f"Processing file: {fname}")
            path = os.path.join(self.raw_data_path, fname)
            with h5py.File(path, "r") as data:
                episode_indices = sorted(
                    int(k.split("_")[1])
                    for k in data.keys()
                    if k.startswith("episode_")
                )
                if self.max_episodes is not None:
                    episode_indices = episode_indices[:self.max_episodes]
                for episode_idx in episode_indices:  
                    global_episode_idx, mem_buffer, exec_sample_id, total_sample_id = (
                        self._process_episode(
                            data, episode_idx, global_episode_idx,
                            mem_buffer, exec_sample_id, total_sample_id,
                        )
                    )

        stats = {"execution_samples": exec_sample_id, "total_samples": total_sample_id}
        with open(os.path.join(self.meta_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    def _first_execution_step(self, episode_data: h5py.Group) -> int:
        return first_execution_step(episode_data)

    def _remove_redundant_keyframes(
        self, keyframe_idxs: list[int], exec_start_idx: int, threshold: int = 10
    ) -> list[int]:
        return remove_redundant_keyframes(keyframe_idxs, exec_start_idx, threshold)

    def _process_episode(
        self,
        data: h5py.File,
        episode_idx: int,
        global_episode_idx: int,
        mem_buffer: MemoryBuffer,
        exec_sample_id: int,
        total_sample_id: int,
    ) -> tuple[int, MemoryBuffer, int, int]:
        episode_data = data[f"episode_{episode_idx}"]
        task_goal = episode_data["setup"]["task_goal"][()][0].decode()
        num_timesteps = sum(1 for k in episode_data.keys() if k.startswith("timestep_"))
        exec_start_idx = self._first_execution_step(episode_data)

        visualization_videos: list[np.ndarray] = []
        record_videos: list[np.ndarray] = []
        keyframe_idxs: list[int] = []

        episode_feature_dir = os.path.join(self.feature_path, f"episode_{global_episode_idx}")
        os.makedirs(episode_feature_dir, exist_ok=True)

        for step_idx in range(num_timesteps):
            ts = episode_data[f"timestep_{step_idx}"]
            action_chunk = get_action_chunk(episode_data, step_idx, horizon=ACTION_CHUNK_HORIZON)
            joint_state = ts["obs"]["joint_state"][()]
            gripper_state = ts["obs"]["gripper_state"][()]
            state = np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)
            image = ts["obs"]["front_rgb"][()]
            wrist_image = ts["obs"]["wrist_rgb"][()]
            # Load top camera if available
            if "top_rgb" in ts["obs"]:
                top_image = ts["obs"]["top_rgb"][()]
            else:
                top_image = None
            is_video_demo = step_idx < exec_start_idx
            assert ts["info"]["is_video_demo"][()] == is_video_demo, "is_video_demo mismatch"

            if not ts['info']['is_completed'][()]:
                simple_subgoal = ts["info"]["simple_subgoal"][()].decode()
                grounded_subgoal = ts["info"]["grounded_subgoal"][()].decode()
                simple_subgoal_online = ts["info"]["simple_subgoal_online"][()].decode()
                grounded_subgoal_online = ts["info"]["grounded_subgoal_online"][()].decode()
            
            if ts["info"]["is_subgoal_boundary"][()]:
                keyframe_idxs.append(step_idx)

            frame_dict = {
                "image": image,
                "wrist_image": wrist_image,
                "top_image": top_image,
                "state": state,
                "actions": action_chunk,
                "is_demo": np.array([is_video_demo], dtype=np.bool_),
                "exec_start_idx": np.array([exec_start_idx], dtype=np.int32),
                "step_idx": np.array([step_idx], dtype=np.int32),
                "epis_idx": np.array([global_episode_idx], dtype=np.int32),
                "prompt": task_goal.lower(),
                "simple_subgoal": simple_subgoal.lower(),
                "grounded_subgoal": grounded_subgoal.lower(),
                "simple_subgoal_online": simple_subgoal_online.lower(),
                "grounded_subgoal_online": grounded_subgoal_online.lower(),
            }

            # Stack all available camera views for memory buffer
            if top_image is not None:
                # Stack 3 cameras: front, wrist, top -> shape (1, 3, 224, 224, 3)
                stacked_images = np.stack([image, wrist_image, top_image], axis=0)[None, ...]
            else:
                # Fallback: only 2 cameras, pad with zeros
                stacked_images = np.stack([image, wrist_image, np.zeros_like(image)], axis=0)[None, ...]

            mem_buffer.add_buffer(stacked_images, state[None, ...], [step_idx])
            feat_path = os.path.join(episode_feature_dir, f"token_emb_{step_idx}.npy")
            np.save(feat_path, mem_buffer.get_history_feats(step_idx))

            if not is_video_demo:
                pkl_path = os.path.join(self.data_path, f"{exec_sample_id}.pkl")
                assert not os.path.exists(pkl_path), f"Collision: {pkl_path}"
                with open(pkl_path, "wb") as f:
                    pickle.dump(frame_dict, f)
                exec_sample_id += 1
            total_sample_id += 1

            visualization_videos.append(image.copy())
            # Concatenate all available camera views for recording
            if top_image is not None:
                record_videos.append(np.concatenate([image, wrist_image, top_image], axis=1))
            else:
                record_videos.append(np.concatenate([image, wrist_image], axis=1))

        kept_indices = mem_buffer.get_token_dropping_indices()
        with open(os.path.join(episode_feature_dir, "kept_indices.json"), "w") as f:
            json.dump(kept_indices, f)

        if self.visualize:
            keyframe_idxs = self._remove_redundant_keyframes(keyframe_idxs, exec_start_idx)
            visualize_frame_sampling(record_videos, episode_feature_dir, exec_start_idx, task_goal, keyframe_idxs)
            visualize_token_dropping(kept_indices, visualization_videos, episode_feature_dir, exec_start_idx, task_goal)

        mem_buffer.clear()
        print(
            f"Episode {global_episode_idx}: timesteps={num_timesteps}, exec_start={exec_start_idx}, kept_indices={len(kept_indices)}, task_goal='{task_goal}'",
        )
        return global_episode_idx + 1, mem_buffer, exec_sample_id, total_sample_id