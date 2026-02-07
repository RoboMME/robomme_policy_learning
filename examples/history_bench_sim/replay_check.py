import numpy as np
import gymnasium as gym
import cv2
import imageio
import h5py
import os

from utils import suppress_warnings

suppress_warnings()


from historybench.env_record_wrapper import DemonstrationWrapper
from historybench.HistoryBench_env import *


def replay(dirpath: str = "data/raw/robomme"):   
    for file in os.listdir(dirpath):
        if not file.endswith(".h5"):
            continue
        if "PickHighlight" not in file:
            continue
        data = h5py.File(os.path.join(dirpath, file), "r")
        for env_id in data.keys():
            env_dataset = data[env_id]
            episode_indexs = sorted(
                int(k.split("_")[1])
                for k in env_dataset.keys()
                if k.startswith("episode_")
            )

            for episode_idx in episode_indexs:
                process_episode(env_dataset, episode_idx, env_id)

def process_episode(env_dataset, episode_idx, env_id, render: bool = False):
    task_name = env_id.replace("env_", "")
    episode_dataset = env_dataset[f"episode_{episode_idx}"]
    seed = episode_dataset["setup"]["seed"][()]
    seed = int(seed)
    task_goal = episode_dataset["setup"]["language goal"][()].decode()
    difficulty = episode_dataset["setup"]["difficulty"][()].decode()
    task_goal = task_goal.replace(" ", "_").replace("/", "_")
    print(f"task_name: {task_name}, episode_idx: {episode_idx}, difficulty: {difficulty}")
    print(f"task_goal: {task_goal}")

    env= gym.make(
        task_name,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode=None,
        reward_mode="dense",
        HistoryBench_seed=seed,
        HistoryBench_difficulty=difficulty,
        max_episode_steps=99999,
    )
    env = DemonstrationWrapper(env, max_steps_without_demonstration=99999, gui_render=False)
    env.reset()
    pre_traj = env.demonstration_data 
    total_images = []        
    demo_length = len(pre_traj['frames'])    
    print("pre_traj length in evaluation", len(pre_traj['frames']))
    
    
    for idx in range(0, len(pre_traj['frames'])):
        concated_image = np.concatenate(
            [pre_traj['frames'][idx], pre_traj['wrist_frames'][idx]],
            axis=1
        )
        
        if len(pre_traj['frames']) > 1 and idx < len(pre_traj['frames']) - 1:
            concated_image = cv2.rectangle(concated_image, (0, 0), 
                                           (concated_image.shape[1], concated_image.shape[0]), (255, 0, 0), 10)
            
        total_images.append(concated_image)        
    
    data_images = []
    step_idx = 0
    

    while episode_dataset[f"record_timestep_{step_idx}"]["demonstration"][()]:
        timestep_group = episode_dataset[f"record_timestep_{step_idx}"]
        image = np.asarray(timestep_group["image"][()], dtype=np.uint8)
        wrist_image = np.asarray(timestep_group["wrist_image"][()], dtype=np.uint8)
        data_images.append(
            np.concatenate(
            [image, wrist_image],
            axis=1
        ))
        step_idx += 1
    print("pre_traj length in data", len(data_images))
        
    step_idx += 1
    action_list = []
    
    total_steps = len([k for k in episode_dataset.keys() if k.startswith("record_timestep_")])    
    
    while step_idx < total_steps:
        timestep_group = episode_dataset[f"record_timestep_{step_idx}"]
        action = np.asarray(timestep_group["action"][()], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        action_list.append(action)        

        image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        wrist_image = obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()
        state = env.agent.robot.qpos.cpu().numpy() if hasattr(env.agent.robot.qpos, 'cpu') else env.agent.robot.qpos
                
        total_images.append(
            np.concatenate(
            [image, wrist_image],
            axis=1
        ))
                
        if render:
            env.render()

        if truncated:
            print("time limit!")
            break
        
        elif terminated:
            if info.get("success", False):
                print("success")
            if info.get("fail", False):
                print("fail")
            break
        
        step_idx += 1
    
    if info.get("success", False):
        flag= "success"
    elif info.get("fail", False):
        flag= "fail"
    else:
        flag= "unknown"
    
    # save video with imageio
    os.makedirs("replay_1206_v3", exist_ok=True)
    imageio.mimsave(f"replay_1206_v3/{flag}_{task_name}_ep{episode_idx}_{task_goal}_step-{demo_length}-{len(total_images)}_v2.mp4", total_images, fps=30)
    env.close()
    
    print(f"--- Finished Running simulation for episode:{episode_idx},env: {task_name} ---")
    
    del env
          
def is_noop(action, prev_action=None, threshold=1e-4):
        if prev_action is None:
            return False

        gripper_action = action[-1]
        prev_gripper_action = prev_action[-1]
        return (
            np.linalg.norm(action[:-1] - prev_action[:-1]) < threshold
            and gripper_action == prev_gripper_action
        )  


if __name__ == "__main__":
    import tyro
    tyro.cli(replay)
