"""
We duplicate keyframe training samples for data balance, otherwise the number of keyframe samples is too few
VLM will not predict correct subgoal changes.
"""

import os
import json
import imageio
import h5py
import numpy as np
import shutil
import re
import cv2


def get_data(episode_dataset, idx, field_name, lambda_fn, dtype=None):
    timestep_group = episode_dataset[f"record_timestep_{idx}"]
    data = timestep_group[field_name][()]
    if dtype is str:
        data = data.decode()
    elif dtype is not None:
        data = np.asarray(data, dtype=dtype)
    return lambda_fn(data)

get_is_demo = lambda episode_dataset, idx: get_data(episode_dataset, idx, "demonstration", lambda x: bool(x))
get_image = lambda episode_dataset, idx: get_data(episode_dataset, idx, "image", lambda x: x, dtype=np.uint8)
get_wrist_image = lambda episode_dataset, idx: get_data(episode_dataset, idx, "wrist_image", lambda x: x, dtype=np.uint8)

get_simple_subgoal = lambda episode_dataset, idx: get_data(episode_dataset, idx, "simple_subgoal", lambda x: x, dtype=str)
get_grounded_subgoal = lambda episode_dataset, idx: get_data(episode_dataset, idx, "grounded_subgoal", lambda x: x, dtype=str)


class DatasetBuilder:
    def __init__(
        self, 
    ):
        self.data_dir = "data/vlm_subgoal_prediction_data/qwenvl_sample_data"
        self.images_dir = os.path.join(self.data_dir, "images")
        
        if os.path.exists(self.images_dir):
            shutil.rmtree(self.images_dir)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.simple_subgoal_train_data_path = os.path.join(self.data_dir, "simple_subgoal_train.jsonl")
        if os.path.exists(self.simple_subgoal_train_data_path):
            os.remove(self.simple_subgoal_train_data_path)
        self.grounded_subgoal_train_data_path = os.path.join(self.data_dir, "grounded_subgoal_train.jsonl")
        if os.path.exists(self.grounded_subgoal_train_data_path):
            os.remove(self.grounded_subgoal_train_data_path)
            
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []

    def build_historybench(self, dirpath: str):
        for file in os.listdir(dirpath):
            if not file.endswith(".h5"):
                continue
            print(f"\nprocessing file: {file}")
            data = h5py.File(os.path.join(dirpath, file), "r")
            for env_id in data.keys():
                env_dataset = data[env_id]
                episode_indexs = sorted(
                    int(k.split("_")[1])
                    for k in env_dataset.keys()
                    if k.startswith("episode_")
                )

                for episode_idx in episode_indexs[:1]:
                    self.process_per_episode(env_dataset, env_id,episode_idx)
    
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
                    
    def make_simple_subgoal_data(self, task_goal, subgoal, image_path, video_path=None) -> dict:
        system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals"
        video_prefix = "<video>" if video_path else ""
        if len(self.history_simple_subgoals) == 0:
            user_prompt = f"{video_prefix}The task goal is: {task_goal}\nThis is the initial turn for prediction\n<image>What's the next language subgoal based on current observation?"
        else:
            user_prompt = f"{video_prefix}The task goal is: {task_goal}\nThe history of previous predicted language subgoals are: {self._wrap_history_subgoals(self.history_simple_subgoals)}\n<image>What's the next language subgoal based on current observation?"
        
        result = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": subgoal
                }
            ],
            "images": [
                image_path
            ],
        }
        if self.history_simple_subgoals:
            if self.history_simple_subgoals[-1] != subgoal:
                self.history_simple_subgoals.append(subgoal)
        else:
            self.history_simple_subgoals.append(subgoal)
        
        if video_path:
            result["videos"] = [video_path]
        
        return result
    
    
    def _preprocess_grounded_subgoal(self, subgoal) -> tuple:
        bbox = []
        # seatch the pattern "at <y, x>"
        matches = re.findall(r"at <(\d+), (\d+)>", subgoal)
        if matches:
            bbox = [[int(float(match[0])), int(float(match[1]))] for match in matches]
        else:
            bbox = []
        # place all "at <y, x>" into  "at <bbox>"
        subgoal = re.sub(r"at <(\d+), (\d+)>", "at <bbox>", subgoal)
        
        return subgoal, bbox

    def _add_noise_to_bbox(self, bbox) -> list:
        y_noise = np.random.randint(-2, 2)
        x_noise = np.random.randint(-2, 2)
        bbox_noisy = []
        for bbox_item in bbox:
            y, x = bbox_item
            bbox_noisy.append([min(max(y + y_noise, 0), 255), min(max(x + x_noise, 0), 255)])
        # print("bbox:", bbox, "->", bbox_noisy)
        return bbox_noisy
    
    def make_grounded_subgoal_data(self, task_goal, subgoal, image_path, video_path=None) -> dict:
        system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of grounded language subgoals"
        video_prefix = "<video>" if video_path else ""
        
        assistant_prompt, bbox = self._preprocess_grounded_subgoal(subgoal)
        
        if len(self.history_grounded_subgoals) == 0:
            user_prompt = f"{video_prefix}The task goal is: {task_goal}\nThis is the initial turn for prediction\n<image>What's the next grounded language subgoal based on current observation?"
        else:            
            user_prompt = f"{video_prefix}The task goal is: {task_goal}\nThe history of previous predicted grounded language subgoals are: {self._wrap_history_subgoals(self.history_grounded_subgoals)}\n<image>What's the next grounded language subgoal based on current observation?"
        
        
        result = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_prompt
                }
            ],
            "objects": {
                "ref": [],
                "bbox": self._add_noise_to_bbox(self.history_grounded_bboxes + bbox)
            },
            "images": [
                image_path
            ],
        }
        
        if video_path:
            result["videos"] = [video_path]
        
        if self.history_grounded_subgoals:
            if self.history_grounded_subgoals[-1] != assistant_prompt:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
        else:
            self.history_grounded_subgoals.append(assistant_prompt)
            self.history_grounded_bboxes.extend(bbox)
        
        return result

    
    def combine_image_and_wrist_image(self, image, wrist_image, simple_subgoal) -> np.ndarray:
        output = np.concatenate([image, wrist_image], axis=1)
        # add simple_subgoal text on the top of the image
        output = cv2.putText(output, simple_subgoal, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return output
    
    def process_per_episode(
        self,
        env_dataset: h5py.File,
        env_id: str,
        episode_idx: int
    ):
        print(f"processing episode {episode_idx} of {env_id}...")
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []
        
        episode_dataset = env_dataset[f"episode_{episode_idx}"]
        task_goal = episode_dataset["setup"]["language goal"][()].decode().lower()

        timestep_indexs = sorted(
            int(k.split("_")[-1])
            for k in episode_dataset.keys()
            if k.startswith("record_timestep_")
        )
                
        # get the execution start idx
        idx = 0
        while get_is_demo(episode_dataset, idx):
            idx += 1
        exec_start_idx = idx
        
        # get all the transition frame idxs
        
        transition_idxs = []
        if "PatternLock" in env_id:
            transition_idxs = np.arange(exec_start_idx, len(timestep_indexs), 32).astype(np.int32).tolist()
        else:
            last_simple_subgoal = None
            idx = exec_start_idx
            while idx < len(timestep_indexs):
                simple_subgoal = get_simple_subgoal(episode_dataset, idx)            
                if "complete" in simple_subgoal:
                    simple_subgoal = last_simple_subgoal
                if simple_subgoal != last_simple_subgoal:
                    transition_idxs.append(idx)
                last_simple_subgoal = simple_subgoal
                idx += 1
            

        transition_idxs.append(len(timestep_indexs)-1)
        print('transition_idxs: ', transition_idxs)
        
        duplicate_idxs = {}
        
        select_idxs = []
        if "StopCube" in env_id:
            stride = 32
        else:
            stride = 16
    
        for start_idx, end_idx in zip(transition_idxs[:-1], transition_idxs[1:]):            
            mid_number = (end_idx - start_idx) // stride
            mid_idxs = np.linspace(start_idx, end_idx, mid_number, endpoint=False).astype(np.int32).tolist()
            select_idxs.extend(mid_idxs)
            
            duplicate_idxs[end_idx] = max(mid_number - 1, 0)
            
        duplicate_idxs.pop(len(timestep_indexs)-1)
            
        select_idxs.append(len(timestep_indexs)-1)
        select_idxs = sorted(list(set(select_idxs)))
                
        print('select_idxs: ', select_idxs)
        if exec_start_idx > 0:
            video_frames = []            
            for i in range(exec_start_idx):
                video_frames.append(get_image(episode_dataset, i))
            video_path = os.path.join(self.images_dir, f"{env_id}_ep{episode_idx}_video.mp4")
            imageio.mimsave(video_path, video_frames, fps=30)
        else:
            video_path = None
            

        last_simple_subgoal = None
        last_grounded_subgoal = None
        
        save_images = []
        visualization_video_path = os.path.join(os.path.dirname(self.images_dir), "visualization")
        os.makedirs(visualization_video_path, exist_ok=True)
        
        for idx in select_idxs:
            image = get_image(episode_dataset, idx)
            # wrist_image = get_wrist_image(episode_dataset, idx)  # don't use wrist image for subgoal prediction
            simple_subgoal = get_simple_subgoal(episode_dataset, idx).lower()
            grounded_subgoal = get_grounded_subgoal(episode_dataset, idx).lower()
            
            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal
                
            image_path = os.path.join(self.images_dir, f"{env_id}_ep{episode_idx}_step{idx}.png")
            imageio.imwrite(image_path, image)
            
            simple_subgoal_data = self.make_simple_subgoal_data(task_goal, simple_subgoal, image_path, video_path)
            grounded_subgoal_data = self.make_grounded_subgoal_data(task_goal, grounded_subgoal, image_path, video_path)

                        
            with open(self.simple_subgoal_train_data_path, "a") as f:
                f.write(json.dumps(simple_subgoal_data) + "\n")
            with open(self.grounded_subgoal_train_data_path, "a") as f:
                f.write(json.dumps(grounded_subgoal_data) + "\n")
                
            image_copy = image.copy()
            image_copy = cv2.putText(image_copy, f"Step {idx}: {simple_subgoal}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            save_images.append(image_copy)  
            
            if idx in duplicate_idxs and duplicate_idxs[idx] > 0:
                print(f"duplicate {idx} for {duplicate_idxs[idx]} more times")
                for _ in range(duplicate_idxs[idx]):
                    with open(self.simple_subgoal_train_data_path, "a") as f:
                        f.write(json.dumps(simple_subgoal_data) + "\n")
                    with open(self.grounded_subgoal_train_data_path, "a") as f:
                        f.write(json.dumps(grounded_subgoal_data) + "\n")
                    
                    image_copy = image.copy()
                    image_copy = cv2.putText(image_copy, f"Duplicate: {simple_subgoal}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    save_images.append(image_copy)  
        
            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal

        imageio.mimsave(os.path.join(visualization_video_path, f"{env_id}_ep{episode_idx}_save_images.mp4"), save_images, fps=1)

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_historybench(dirpath="data/robomme_h5_data")