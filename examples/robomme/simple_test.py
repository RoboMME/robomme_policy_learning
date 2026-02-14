import numpy as np
import imageio
import os

# # Force CPU rendering to avoid Vulkan driver issues
# os.environ['SAPIEN_RENDER_DEVICE'] = 'cpu'
# os.environ['MUJOCO_GL'] = 'osmesa'

from tqdm import tqdm
from env_runner import EnvRunner
from utils import TASK_NAME_LIST



def add_small_noise(
    action: np.ndarray, noise_level: float = 0.0
) -> np.ndarray:
    """Add Gaussian noise to the first `dim` dimensions of the action."""
    noise = np.random.normal(0, noise_level, action.shape)
    noise[..., -1:] = 0.0
    return action + noise


BASE_ACTION = np.array(
    [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
    dtype=np.float32,
)


VIDEO_DIR = "runs/sanity_check_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

for task_name in TASK_NAME_LIST:    
    env_runner = EnvRunner(env_id=task_name, video_save_dir=VIDEO_DIR, max_steps=200)
    num_episodes = env_runner.num_episodes
    print(f"[robomme] Task {task_name} has {num_episodes} episodes to evaluate")
    videos = []
    episode_id = 0
    env_runner.make_env(episode_id)
    env_runner.get_init_obs()
    print(f"[robomme] Episode {episode_id} setup finished")
    for _ in tqdm(range(1000)):
        action = add_small_noise(BASE_ACTION, noise_level=0.01)
        obs, stop_flag, success_flag = env_runner.step(action)
        img, wrist_img, state = obs
        videos.append(np.concatenate([img, wrist_img], axis=1))
        if stop_flag:
            break
    imageio.mimsave(f"{VIDEO_DIR}/{task_name}_{episode_id}.mp4", videos, fps=30)    
    env_runner.close_env()