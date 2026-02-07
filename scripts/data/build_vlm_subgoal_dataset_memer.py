import os
import json
import imageio
import h5py
import numpy as np
import shutil
import re
import cv2

np.set_printoptions(precision=4, suppress=True)

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
get_action = lambda episode_dataset, idx: get_data(episode_dataset, idx, "action", lambda x: x, dtype=np.float32)
get_state = lambda episode_dataset, idx: get_data(episode_dataset, idx, "state", lambda x: x, dtype=np.float32)[0, :8]

get_simple_subgoal = lambda episode_dataset, idx: get_data(episode_dataset, idx, "simple_subgoal", lambda x: x, dtype=str)
get_grounded_subgoal = lambda episode_dataset, idx: get_data(episode_dataset, idx, "grounded_subgoal", lambda x: x, dtype=str)


def find_local_minima_simple(values, threshold=0.001, min_distance=5):
    minima_indices = []
    n = len(values)
    
    for i in range(1, n - 1):  # Skip first and last points
        # Check if value is below threshold
        if values[i] >= threshold:
            continue
        
        # Check if it's smaller than both neighbors
        if values[i] >= values[i - 1] or values[i] >= values[i + 1]:
            continue
        
        # Check distance from last found minimum
        if minima_indices and (i - minima_indices[-1] < min_distance):
            continue
        
        # All conditions met - this is a local minimum
        minima_indices.append(i)
    
    return minima_indices


def wrap_text_opencv(text, font, font_scale, max_width, thickness=1):
    """Wrap text to fit within max_width pixels."""
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def get_middle_point(frm_0, frm_1, num=1) -> int:
    return np.linspace(frm_0, frm_1, num+2)[1:-1].astype(int).tolist()
    

def put_wrapped_text(image, text, org, font, font_scale, color, thickness=1, line_spacing=20):
    """Put wrapped text on image."""
    x, y = org
    max_width = image.shape[1] - x - 10  # leave some margin
    lines = wrap_text_opencv(text, font, font_scale, max_width, thickness)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (x, y + i * line_spacing), font, font_scale, color, thickness)
    return image, y + len(lines) * line_spacing  # return next y position

def merge_transition_idxs_with_local_minimum_idxs(transition_idxs: list, local_minimum_idxs: list, exec_start_idx: int) -> list:
    merged_idxs = transition_idxs + local_minimum_idxs
    merged_idxs = sorted(list(set(merged_idxs)))
    merged_idxs = [idx for idx in merged_idxs if idx >= exec_start_idx]
    output = [merged_idxs[0]]
    for i in range(1, len(merged_idxs)):
        if abs(merged_idxs[i] - output[-1]) < 10:
            output[-1] = (merged_idxs[i] + output[-1]) // 2
        else:
            output.append(merged_idxs[i])
    
    return output

class DatasetBuilder:
    def __init__(
        self, 
    ):
        self.data_dir = "data/vlm_subgoal_prediction_data/memer_sample_data"
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
        max_frame_len = 0
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

                for episode_idx in episode_indexs:
                    frame_len = self.process_per_episode(env_dataset, env_id,episode_idx)
                    max_frame_len = max(max_frame_len, frame_len)
        print(f"max_frame_len: {max_frame_len}")
    
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
    
    def _wrap_keyframes(self, key_frame_paths) -> str:
        return "; ".join([f"Past Keyframe {i+1}: <image>" for i in range(len(key_frame_paths))])
    
    def _wrap_execution_frames(self, execution_frame_paths) -> str:
        return "; ".join([f"Executed Frame {i+1}: <image>" for i in range(len(execution_frame_paths))])
    
    def _wrap_images(self, image_paths) -> str:
        if len(image_paths) == 0:
            return "[]"
        return "[" + ", ".join(["<image>" for _ in image_paths]) + "]"
                    
    def make_simple_subgoal_data(self, task_goal, subgoal, execution_frame_paths, key_frame_paths, video_path=None, candidate_frame_idx=None) -> dict:
        system_prompt = "You are a robot program that predicts actions. The current input images from the front-view camera shows the most recent actions the robot has executed. The past keyframes are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else. Some tasks may have a video input for initial setup, some may not.\n\nReturn a JSON with:\n- current_subtask: the action that should be executed at the current timestep\n- keyframe_positions: list of frame positions (1-indexed) from the current input images where actions change"
        
        video_prefix = "The task has a video input for initial setup: <video>\n" if video_path else "" 
        
        user_prompt = f"{video_prefix}The task goal is: {task_goal}\nHere are the selected frames from the entirety of the full execution that are of particular importance:{self._wrap_images(key_frame_paths)}\nHere is current input image list from the front-view camera: {self._wrap_images(execution_frame_paths)}\n\nWhat subtask should the robot execute and what is the keyframe position?"
        
        all_image_paths = key_frame_paths + execution_frame_paths
        
        if candidate_frame_idx is not None:
            assistant_response = f"""{{"current_subtask": "{subgoal}", "keyframe_positions": [{candidate_frame_idx}]}}"""
        else:
            assistant_response = f"""{{"current_subtask":"{subgoal}", "keyframe_positions": []}}"""
                    
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
                    "content": assistant_response
                }
            ],
            "images": all_image_paths,
        }
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
    
    def make_grounded_subgoal_data(self, task_goal, subgoal, execution_frame_paths, key_frame_paths, video_path=None, candidate_frame_idx=None) -> dict:
        system_prompt = "You are a robot program that predicts actions. The current input images from the front-view camera shows the most recent actions the robot has executed. The past keyframes are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else. Some tasks may have a video input for initial setup, some may not.\n\nReturn a JSON with:\n- current_subtask: the action that should be executed at the current timestep\n- keyframe_positions: list of frame positions (1-indexed) from the current input images where actions change"
        
        video_prefix = "The task has a video input for initial setup: <video>\n" if video_path else "" 
        
        user_prompt = f"{video_prefix}The task goal is: {task_goal}\nHere are the selected frames from the entirety of the full execution that are of particular importance:{self._wrap_images(key_frame_paths)}\nHere is current input image list from the front-view camera: {self._wrap_images(execution_frame_paths)}\n\nWhat subtask should the robot execute and what is the keyframe position?"
        
        all_image_paths = key_frame_paths + execution_frame_paths
        
        assistant_prompt, bbox = self._preprocess_grounded_subgoal(subgoal)
                
        if candidate_frame_idx is not None:
            assistant_response = f"""{{"current_subtask": "{assistant_prompt}", "keyframe_positions": [{candidate_frame_idx}]}}"""
        else:
            assistant_response = f"""{{"current_subtask":"{assistant_prompt}", "keyframe_positions": []}}"""
                    
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
                    "content": assistant_response
                }
            ],
            "objects": {
                "ref": [],
                "bbox": self._add_noise_to_bbox(self.history_grounded_bboxes + bbox)
            },
            "images": all_image_paths,
        }
        
        if video_path:
            result["videos"] = [video_path]
        
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
        
        delta_actions = []
        for idx in timestep_indexs:
            state = get_state(episode_dataset, idx)
            action = get_action(episode_dataset, idx)
            delta_action = action - state
            delta_actions.append(np.linalg.norm(delta_action))
            
        delta_actions = np.array(delta_actions)        
        local_minimum_idxs = find_local_minima_simple(delta_actions)
                        
        # get the execution start idx
        idx = 0
        while get_is_demo(episode_dataset, idx):
            idx += 1
        exec_start_idx = idx
        
        # get all the transition frame idxs
        transition_idxs = []
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
        
        if env_id in ["env_PatternLock", "env_RouteStick"]:
            key_frame_idxs = merge_transition_idxs_with_local_minimum_idxs(transition_idxs, local_minimum_idxs, exec_start_idx)
        else:
            key_frame_idxs = transition_idxs
            
        if env_id == "env_PickHighlight":
            # add two more key frames between 0 and 1
            frm_0 = key_frame_idxs[0]
            frm_1 = key_frame_idxs[1]
            key_frame_idxs.extend(get_middle_point(frm_0, frm_1, 1))
            key_frame_idxs = sorted(key_frame_idxs)
                            
        if env_id in "env_ButtonUnmaskSwap":
            # add one more key frame between 0 and 1 and 1 and 2
            frm_0 = key_frame_idxs[0]
            frm_1 = key_frame_idxs[1]
            frm_2 = key_frame_idxs[2]
            key_frame_idxs.extend(get_middle_point(frm_0, frm_1, 2))
            key_frame_idxs.extend(get_middle_point(frm_1, frm_2, 2))
            key_frame_idxs = sorted(key_frame_idxs)
        
        
        key_frame_idxs.pop()
        print('key_frame_idxs: ', key_frame_idxs)
                
        if exec_start_idx > 0:
            video_frames = []            
            for i in range(exec_start_idx):
                video_frames.append(get_image(episode_dataset, i))
            init_video_path = os.path.join(self.images_dir, f"{env_id}_ep{episode_idx}_video.mp4")
            imageio.mimsave(init_video_path, video_frames, fps=30)
        else:
            init_video_path = None
            
        last_simple_subgoal = None
        last_grounded_subgoal = None
        
        save_images = []
        visualization_video_path = os.path.join(os.path.dirname(self.images_dir), "visualization")
        os.makedirs(visualization_video_path, exist_ok=True)
        
        memory_frames = []
        execution_frames = []
        candidate_frame_idx = None
        step = 0
        max_frame_len = 0
        for step, idx in enumerate(range(exec_start_idx, len(timestep_indexs), 2)):
            simple_subgoal = get_simple_subgoal(episode_dataset, idx).lower()
            grounded_subgoal = get_grounded_subgoal(episode_dataset, idx).lower()
            if "complete" in simple_subgoal:
                simple_subgoal = last_simple_subgoal
            if "complete" in grounded_subgoal:
                grounded_subgoal = last_grounded_subgoal
            
            # save dense images
            image = get_image(episode_dataset, idx)
            image_path = os.path.join(self.images_dir, f"{env_id}_ep{episode_idx}_step{idx}.png")
            imageio.imwrite(image_path, image)
            execution_frames.append(image_path)
            

            # check if it's key frame
            if candidate_frame_idx is None and key_frame_idxs and abs(idx - key_frame_idxs[0]) < 2:
                candidate_frame_idx = len(execution_frames)
                candidate_frame_image_path = image_path
                
            if step % 8 == 0 or idx in [len(timestep_indexs) - 1, len(timestep_indexs) - 2]:
                # save data
                simple_subgoal_data = self.make_simple_subgoal_data(task_goal, simple_subgoal, execution_frames, memory_frames, init_video_path, candidate_frame_idx)
                grounded_subgoal_data = self.make_grounded_subgoal_data(task_goal, grounded_subgoal, execution_frames, memory_frames, init_video_path, candidate_frame_idx)
                
                max_frame_len = max(len(memory_frames)+len(execution_frames), max_frame_len)
                            
                with open(self.simple_subgoal_train_data_path, "a") as f:
                    f.write(json.dumps(simple_subgoal_data) + "\n")
                with open(self.grounded_subgoal_train_data_path, "a") as f:
                    f.write(json.dumps(grounded_subgoal_data) + "\n")
                
                                
                top_image = np.zeros((100, 256*8, 3), dtype=np.uint8)
                top_image, _ = put_wrapped_text(top_image, grounded_subgoal_data["messages"][1]["content"], (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                top_image, _ = put_wrapped_text(top_image, grounded_subgoal_data["messages"][2]["content"], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                middle_image = np.zeros((256, 256*8, 3), dtype=np.uint8)
                si=0
                for i, key_image in enumerate(memory_frames):
                    # only take the last 8 key frames
                    if i < len(memory_frames) - 8:
                        continue
                    key_image = imageio.imread(key_image).copy() # 256x256x3
                    key_image = cv2.putText(key_image, f"KeyFrame {i+1}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    middle_image[:, si*256:(si+1)*256, :] = key_image
                    si += 1
                bottom_image = np.zeros((256, 256*8, 3), dtype=np.uint8)
                for i, execution_image in enumerate(execution_frames):
                    execution_image = imageio.imread(execution_image).copy() # 256x256x3 
                    execution_image = cv2.putText(execution_image, f"Frame {i+1}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # add point of bbox on the image
                    for bbox in grounded_subgoal_data["objects"]["bbox"]:
                            x = int(bbox[1])
                            y = int(bbox[0])
                            execution_image = cv2.circle(execution_image, (x, y), 5, (255, 0, 255), -1)
                    
                    if i+1 == candidate_frame_idx:
                        # put a red frames 
                        execution_image = cv2.rectangle(execution_image, (0, 0), (255, 255), (255, 0, 0), 10)
                    bottom_image[:, i*256:(i+1)*256, :] = execution_image
                big_image = np.concatenate([top_image, middle_image, bottom_image], axis=0)
                save_images.append(big_image)
                
                    
                
                if candidate_frame_idx is not None:
                    memory_frames.append(candidate_frame_image_path)
                    key_frame_idxs.pop(0)
                
                candidate_frame_idx = None
                execution_frames = []

            last_simple_subgoal = simple_subgoal
            last_grounded_subgoal = grounded_subgoal
            
                        
        imageio.mimsave(os.path.join(visualization_video_path, f"{env_id}_ep{episode_idx}_save_images.mp4"), save_images, fps=1)
        return max_frame_len

if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build_historybench(dirpath="data/robomme_h5_data")