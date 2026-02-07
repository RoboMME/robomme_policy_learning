"""Dataset preprocessing script for MME-VLA-Suite.

This module handles the conversion of raw HDF5 dataset files into preprocessed
formats suitable for training, including feature extraction and memory buffer management.
"""

import os
import shutil
import json
import h5py
import numpy as np
import logging
import pickle
from typing import Any, Callable, Optional

from mme_vla_suite.shared.mem_buffer import MemoryBuffer


logger = logging.getLogger(__name__)


def get_data(
    episode_dataset: h5py.Group,
    idx: int,
    field_name: str,
    lambda_fn: Callable,
    dtype: Optional[type] = None,
) -> Any:
    """Extract and process data from an episode dataset at a specific timestep.

    Args:
        episode_dataset: HDF5 group containing episode data.
        idx: Timestep index.
        field_name: Name of the field to extract.
        lambda_fn: Function to apply to the extracted data.
        dtype: Optional data type for conversion. If str, decodes bytes.

    Returns:
        Processed data after applying lambda_fn.
    """
    timestep_group = episode_dataset[f"record_timestep_{idx}"]
    data = timestep_group[field_name][()]
    if dtype is str:
        data = data.decode()
    elif dtype is not None:
        data = np.asarray(data, dtype=dtype)
    return lambda_fn(data)


def get_state(episode_dataset: h5py.Group, idx: int) -> np.ndarray:
    """Extract state vector from episode dataset.

    Args:
        episode_dataset: HDF5 group containing episode data.
        idx: Timestep index.

    Returns:
        State vector of length 8 (padded if necessary for RouteStick & DrawPattern tasks).
    """
    state = get_data(episode_dataset, idx, "state", lambda x: x[0, :8], dtype=np.float32)
    if len(state) == 8:
        return state
    else:
        # Pad for RouteStick & DrawPattern tasks
        return np.concatenate([state, np.array([0.0])], axis=0, dtype=np.float32)


def get_action(episode_dataset: h5py.Group, idx: int) -> np.ndarray:
    """Extract action vector from episode dataset.

    Args:
        episode_dataset: HDF5 group containing episode data.
        idx: Timestep index.

    Returns:
        Action vector of length 8 (padded if necessary for RouteStick & DrawPattern tasks).
    """
    action = get_data(episode_dataset, idx, "action", lambda x: x, dtype=np.float32)
    if len(action) == 8:
        return action
    else:
        # Pad for RouteStick & DrawPattern tasks
        return np.concatenate([action, np.array([-1.0])], axis=0, dtype=np.float32)


def get_action_chunk(episode_dataset: h5py.Group, idx: int, horizon: int = 20) -> np.ndarray:
    """Extract a chunk of consecutive actions starting from the given index.

    Args:
        episode_dataset: HDF5 group containing episode data.
        idx: Starting timestep index.
        horizon: Number of actions to extract.

    Returns:
        Stacked array of actions with shape (horizon, action_dim).
    """
    action_chunk = []
    last_action = None
    for i in range(horizon):
        try:
            action = get_action(episode_dataset, idx + i)
            action_chunk.append(action)
            last_action = action
        except (KeyError, IndexError):
            # Use last valid action if we run out of timesteps
            action_chunk.append(last_action)
    return np.stack(action_chunk, axis=0)


# For tasks using video-based observation at the initial step,
# the conditioned video frames are considered as is_demo=True
get_is_demo = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "demonstration", lambda x: bool(x)
)
get_image = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "image", lambda x: x, dtype=np.uint8
)
get_wrist_image = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "wrist_image", lambda x: x, dtype=np.uint8
)
get_subgoal = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "subgoal", lambda x: x.decode()
)
get_simple_subgoal = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "simple_subgoal", lambda x: x, dtype=str
)
get_simple_subgoal_online = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "simple_subgoal_online", lambda x: x, dtype=str
)
get_grounded_subgoal = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "grounded_subgoal", lambda x: x, dtype=str
)
get_grounded_subgoal_online = lambda episode_dataset, idx: get_data(
    episode_dataset, idx, "grounded_subgoal_online", lambda x: x, dtype=str
)



class DatasetProcessor:
    """Processes raw HDF5 datasets into preprocessed format for training.

    This class handles the conversion of raw episode data, including feature
    extraction, memory buffer management, and saving preprocessed samples.
    """

    def __init__(self, raw_data_path: str = "data/raw", preprocessed_data_path: str = "data/preprocessed", execution_horizon: int = 16):
        """Initialize the dataset processor.

        Args:
            repo_id: Repository identifier for the dataset.
            execution_horizon: Horizon for execution steps (used for token dropping).
        """
        self.raw_data_path = raw_data_path
        self.dataset_path = preprocessed_data_path

        # Remove existing preprocessed data if it exists
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)
        os.makedirs(self.dataset_path, exist_ok=True)

        # Set up directory structure
        self.feature_path = os.path.join(self.dataset_path, "features")
        self.data_path = os.path.join(self.dataset_path, "data")
        self.meta_path = os.path.join(self.dataset_path, "meta")
        os.makedirs(self.feature_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.meta_path, exist_ok=True)

        self.execution_horizon = execution_horizon


    def run(self) -> None:
        """Process all raw dataset files and generate preprocessed data.

        Iterates through all HDF5 files in the raw data directory, processes
        each episode, and saves preprocessed samples along with metadata.
        """
        global_episode_idx = 0
        mem_buffer = MemoryBuffer(
            num_views=1,
            compute_token_drop_score=True,
            token_drop_stride=self.execution_horizon // 2,
            prepare_buffer=True,
        )
        self.exec_sample_id = 0
        self.total_sample_id = 0

        for file in os.listdir(self.raw_data_path):
            if not file.endswith(".h5"):
                continue
            logger.info(f"Processing file: {file}")
            with h5py.File(os.path.join(self.raw_data_path, file), "r") as data:
                for env_id in data.keys():
                    env_dataset = data[env_id]
                    episode_indices = sorted(
                        int(k.split("_")[1])
                        for k in env_dataset.keys()
                        if k.startswith("episode_")
                    )

                    for episode_idx in episode_indices:
                        global_episode_idx, mem_buffer = (
                            self.process_per_episode(
                                env_dataset,
                                episode_idx,
                                global_episode_idx,
                                mem_buffer,
                            )
                        )

        # Save processing statistics
        stats = {
            "execution_samples": self.exec_sample_id,
            "total_samples": self.total_sample_id,
        }
        with open(os.path.join(self.meta_path, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    
    
    def process_per_episode(
        self,
        env_dataset: h5py.Group,
        episode_idx: int,
        global_episode_idx: int,
        mem_buffer: MemoryBuffer,
    ) -> tuple[int, MemoryBuffer]:
        """Process a single episode and extract all necessary data.

        Args:
            env_dataset: HDF5 group containing environment data.
            episode_idx: Index of the episode within the environment.
            global_episode_idx: Global episode index across all environments.
            mem_buffer: Memory buffer for storing history features.

        Returns:
            Tuple of (updated_global_episode_idx, mem_buffer).
        """
        episode_dataset = env_dataset[f"episode_{episode_idx}"]
        task_goal = episode_dataset["setup"]["language goal"][()].decode()

        timestep_indices = sorted(
            int(k.split("_")[-1])
            for k in episode_dataset.keys()
            if k.startswith("record_timestep_")
        )
        total_indices = list(range(len(timestep_indices)))

        # Find the first non-demonstration timestep (execution start)
        idx = 0
        while get_is_demo(episode_dataset, idx):
            idx += 1
        exec_start_idx = idx

        # Track last subgoals to handle "complete" markers
        last_simple_subgoal = None
        last_grounded_subgoal = None
        last_simple_subgoal_online = None
        last_grounded_subgoal_online = None

        # Create episode feature directory
        episode_feature_dir = os.path.join(
            self.feature_path, f"episode_{global_episode_idx}"
        )
        os.makedirs(episode_feature_dir, exist_ok=True)

        # Process each timestep
        for step_idx, idx in enumerate(total_indices):
            # Extract data for current timestep
            action_chunk = get_action_chunk(episode_dataset, idx, horizon=20)
            state = get_state(episode_dataset, idx)
            image = get_image(episode_dataset, idx)
            wrist_image = get_wrist_image(episode_dataset, idx)
            is_demo = get_is_demo(episode_dataset, idx)
            simple_subgoal = get_simple_subgoal(episode_dataset, idx)
            grounded_subgoal = get_grounded_subgoal(episode_dataset, idx)

            # Extract online subgoals for temporal shift robustness
            # During online evaluation, subtask success criteria may differ from
            # waypoint-based dataset generation, causing online subgoals to complete
            # earlier than standard subgoals. We include both versions to make the
            # model robust to temporal shifts.
            simple_subgoal_online = get_simple_subgoal_online(episode_dataset, idx)
            grounded_subgoal_online = get_grounded_subgoal_online(episode_dataset, idx)

            # Replace "complete" markers with the last valid subgoal
            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal
            if "complete" in simple_subgoal_online:
                simple_subgoal_online = last_simple_subgoal_online
            if "complete" in grounded_subgoal_online:
                grounded_subgoal_online = last_grounded_subgoal_online

            # Validate timestep consistency
            if is_demo:
                assert step_idx < exec_start_idx, (
                    f"step_idx {step_idx} should be < exec_start_idx {exec_start_idx}"
                )
            else:
                assert step_idx >= exec_start_idx, (
                    f"step_idx {step_idx} should be >= exec_start_idx {exec_start_idx}"
                )

            # Compile frame data dictionary
            frame_dict = {
                "image": image,
                "wrist_image": wrist_image,
                "state": state,
                "actions": action_chunk,
                "is_demo": np.array([is_demo], dtype=np.bool_),
                "exec_start_idx": np.array([exec_start_idx], dtype=np.int32),
                "step_idx": np.array([step_idx], dtype=np.int32),
                "epis_idx": np.array([global_episode_idx], dtype=np.int32),
                "prompt": task_goal.lower(),
                "simple_subgoal": simple_subgoal.lower(),
                "grounded_subgoal": grounded_subgoal.lower(),
                "simple_subgoal_online": simple_subgoal_online.lower(),
                "grounded_subgoal_online": grounded_subgoal_online.lower(),
            }

            # Update last subgoals for next iteration
            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal
            last_simple_subgoal_online = simple_subgoal_online
            last_grounded_subgoal_online = grounded_subgoal_online

            # Add to memory buffer and save features
            mem_buffer.add_buffer(image[None, None, ...], state[None, ...], [step_idx])
            feature_path = os.path.join(
                episode_feature_dir, f"token_emb_{step_idx}.npy"
            )
            # with open(feature_path, "wb") as f:
            #     np.save(f, mem_buffer.get_history_feats(step_idx))

            # Save execution samples (non-demo timesteps)
            if not is_demo:
                filename = os.path.join(self.data_path, f"{self.exec_sample_id}.pkl")
                assert not os.path.exists(filename), f"File already exists: {filename}"
                with open(filename, "wb") as f:
                    pickle.dump(frame_dict, f)
                self.exec_sample_id += 1
            self.total_sample_id += 1

        # Save token dropping indices for this episode
        kept_indices = mem_buffer.get_token_dropping_indices()
        kept_indices_path = os.path.join(episode_feature_dir, "kept_indices.json")
        with open(kept_indices_path, "w") as f:
            json.dump(kept_indices, f)

        mem_buffer.clear()

        logger.info(
            f"Episode {global_episode_idx} processed: "
            f"total_timesteps={len(total_indices)}, "
            f"exec_start_idx={exec_start_idx}, "
            f"kept_indices={len(kept_indices)}, "
            f"task_goal='{task_goal}'"
        )

        return global_episode_idx + 1, mem_buffer



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess raw HDF5 dataset files for training"
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="data/raw",
        help="Path to the raw data",
    )
    parser.add_argument(
        "--preprocessed_data_path",
        type=str,
        default="data/preprocessed",
        help="Path to the dataset",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import time
    start_time = time.time()
    preprocessor = DatasetProcessor(raw_data_path=args.raw_data_path, preprocessed_data_path=args.preprocessed_data_path)
    preprocessor.run()
    end_time = time.time()
    logger.info(f"Time taken: {(end_time - start_time) / 60} minutes")