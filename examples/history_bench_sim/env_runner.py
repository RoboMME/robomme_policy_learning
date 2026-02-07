from historybench.env_record_wrapper import EpisodeConfigResolver
from utils import TASK_NAME_LIST, TASKS_WITH_STICK_GRIPPER
import numpy as np

from historybench.HistoryBench_env import *

np.set_printoptions(precision=4, suppress=True)

def pack_state(state):
    if len(state) == 8:
        return state
    else:
        return np.concatenate([state, np.array([0.0])], axis=0, dtype=np.float32)
    

class EnvRunner:
    def __init__(self, env_id, video_save_dir: str, render: bool = False, max_steps: int = 1000, is_training_dataset: bool = False):
        assert env_id in TASK_NAME_LIST, f"Environment ID {env_id} not found in {TASK_NAME_LIST}"
        self.metadata_path = "/home/daiyp/openpi/third_party/HistoryBench/dataset_json/record_dataset_{env_id}_metadata.json"
        self.env_id = env_id
        self.video_save_dir = video_save_dir
        self.render = render
        
        self.resolver = EpisodeConfigResolver(
            env_id=env_id,
            dataset=None,
            metadata_path=self.metadata_path.format(env_id=env_id),
            render_mode="human" if render else "rgb_array",
            gui_render=render,
            max_steps_without_demonstration=max_steps,
        )
        
        self.num_episodes = 50 # fix for benchmark
    
    
    def make_env(self, episode_id: int):
        self.env, episode_dataset, seed, difficulty = self.resolver.make_env_for_episode(episode_id)
        self.episode_id = episode_id
        self.difficulty = difficulty
    
    
    def get_init_obs(self):
        self.env.reset()
        demonstration_data = self.env.demonstration_data
        images = demonstration_data.get('frames', [])
        wrist_images = demonstration_data.get('wrist_frames', [])
        states = demonstration_data.get('states', [])
        task_goal = demonstration_data.get('language goal')
        states = [pack_state(state[0, :8]) for state in states]
        self.task_goal = task_goal

        return {
            "images": images,
            "wrist_images": wrist_images,
            "states": states,
            "task_goal": task_goal,
        }
        
    @property
    def simple_subgoal_oracle(self):
        return self.env.subgoal[-1]
    
    @property
    def grounded_subgoal_oracle(self):
        return self.env.subgoal_grounded[-1]
        
    
    def step(self, action: np.ndarray):
        if self.env_id in TASKS_WITH_STICK_GRIPPER:
            action = action[:7]
        
        obs, _, terminated, truncated, info = self.env.step(action)
        
        img = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        wrist_img = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        state = self.env.agent.robot.qpos.cpu().numpy() if hasattr(self.env.agent.robot.qpos, 'cpu') else self.env.agent.robot.qpos
        state = pack_state(state[0, :8]) # only use the first 8 dimensions
        
        if self.render:
            self.env.render()
                    
        stop_flag = False
        success_flag = "unknown"
        
        if truncated:
            print("time limit!")
            stop_flag = True
            success_flag = "timeout"
        
        elif terminated:
            if info.get("success", False):
                success_flag = "success"
            if info.get("fail", False):
                success_flag = "fail"
            stop_flag = True
                    
        return (img, wrist_img, state), stop_flag, success_flag
    
    def close_env(self):
        self.env.close()
        del self.env
    
    
    
    