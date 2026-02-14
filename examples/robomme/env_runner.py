"""
RoboMME environment runing wrapper: build envs, get observations, and step with a uniform API.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from robomme.robomme_env import *  # noqa: F401, F403 - env registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder

from utils import TASK_NAME_LIST

np.set_printoptions(precision=4, suppress=True)


def pack_joint_state(state: np.ndarray) -> np.ndarray:
    """Ensure joint state is 8-dimensional; pad with 0 if shorter."""
    if len(state) == 8:
        return state
    return np.concatenate([state, np.array([0.0])], axis=0, dtype=np.float32)


class EnvRunner:
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API, and optional subgoal oracles.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1000) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        self.video_save_dir = video_save_dir

        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
        )
        self.max_steps = max_steps
        self.step_count = 0

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.difficulty: Any = None
        self.task_goal: str = ""

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        self.env, _, self.difficulty = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id

    def get_init_obs(self) -> dict[str, Any]:
        """Reset env and return initial observation dict (images, wrist_images, states, task_goal)."""
        obs, self.info = self.env.reset()
        self.task_goal = self.info["language_goal"][0]

        images = [f.cpu().numpy() for f in obs["front_camera"]]
        wrist_images = [f.cpu().numpy() for f in obs["wrist_camera"]]
        states = [pack_joint_state(s[0,:8].cpu().numpy()) for s in obs["joint_states"]]

        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_goal": self.task_goal,
        }

    def step(self, action: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], bool, str]:
        """
        Execute one step.
        Returns ( (img, wrist_img, state), stop_flag, success_flag ).
        success_flag is one of "success", "fail", "timeout", "unknown".
        """
        obs, _, terminated, truncated, self.info = self.env.step(action)

        img = obs["front_camera"][0].cpu().numpy()
        wrist_img = obs["wrist_camera"][0].cpu().numpy()
        state = pack_joint_state(obs["joint_states"][0].cpu().numpy()[0,:8])

        stop = False
        success_flag = "unknown"

        if truncated[-1] or self.step_count >= self.max_steps:
            stop = True
            success_flag = "timeout"
        elif terminated[-1]:
            stop = True
            if self.info.get("success", False)[-1][-1]: # TODO: to be changed
                success_flag = "success"
            elif self.info.get("fail", False)[-1][-1]:
                success_flag = "fail"
        
        self.step_count += 1
        return (img, wrist_img, state), stop, success_flag

    def close_env(self) -> None:
        """Close and clear the current env."""
        if self.env is not None:
            self.env.close()
            del self.env
            self.env = None
    
    
    @property
    def simple_subgoal_oracle(self) -> str:
        return self.info["subgoal"][-1]
    
    @property
    def grounded_subgoal_oracle(self) -> str:
        return self.info["subgoal_grounded"][-1]