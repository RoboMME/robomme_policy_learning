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


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    # pack into 8-dim state, same as the joint action space
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)

class EnvRunner:
    """
    Wraps RoboMME BenchmarkEnvBuilder for a single task: create env per episode,
    expose initial observation and step API, and optional subgoal oracles.
    """

    def __init__(self, env_id: str, video_save_dir: str, max_steps: int = 1300) -> None:
        if env_id not in TASK_NAME_LIST:
            raise ValueError(f"Environment ID {env_id} not in {TASK_NAME_LIST}")
        self.env_id = env_id
        self.video_save_dir = video_save_dir

        self.env_builder = BenchmarkEnvBuilder(
            env_id=env_id,
            dataset="test",
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )

        # Set after make_env()
        self.env: Any = None
        self.episode_id: int | None = None
        self.task_goal: str = ""

    @property
    def num_episodes(self) -> int:
        return self.env_builder.get_episode_num()

    def make_env(self, episode_id: int) -> None:
        """Build and set the active env for the given episode."""
        self.env = self.env_builder.make_env_for_episode(episode_id)
        self.episode_id = episode_id
        self.difficulty = self.env.unwrapped.difficulty

    def get_init_obs(self) -> dict[str, Any]:
        """Reset env and return initial observation dict (images, wrist_images, states, task_goal)."""
        obs, self.info = self.env.reset()
        if isinstance(self.info["task_goal"], list):
            self.task_goal = self.info["task_goal"][0]
        else:
            self.task_goal = self.info["task_goal"]

        images = [f.cpu().numpy() for f in obs["front_rgb_list"]] # TODO
        wrist_images = [f.cpu().numpy() for f in obs["wrist_rgb_list"]]
        states = [pack_state(joint_state, gripper_state) for joint_state, gripper_state in 
                  zip(obs["joint_state_list"], obs["gripper_state_list"])]

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
        try:
            obs, _, terminated, truncated, self.info = self.env.step(action)
        except Exception as e:
            print(f"Error: {e}")
            return (None, None, None), True, "error"

        img = obs["front_rgb_list"][-1].cpu().numpy()
        wrist_img = obs["wrist_rgb_list"][-1].cpu().numpy()
        joint_state = obs["joint_state_list"][-1]
        gripper_state = obs["gripper_state_list"][-1]
        state = pack_state(joint_state, gripper_state)

        outcome = self.info.get("status", "unknown")
        stop = terminated or truncated
                
        return (img, wrist_img, state), stop, outcome
    
    @property
    def simple_subgoal_oracle(self) -> str:
        return self.info["simple_subgoal_online"][-1]
    
    @property
    def grounded_subgoal_oracle(self) -> str:
        return self.info["grounded_subgoal_online"][-1]
    
    def close_env(self) -> None:
        """Close and clear the current env."""
        if self.env is not None:
            self.env.close()
            del self.env
            self.env = None
            self.episode_id = None
            self.task_goal = None
