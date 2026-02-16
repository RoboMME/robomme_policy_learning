import os
import shutil
from h5py._hl.dataset import sel
import numpy as np
import imageio
from typing import List
import os
import re
import pprint
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'

import json
from swift.llm import PtEngine, InferRequest, RequestConfig

class Qwen3VLModelMemER:
    
    def __init__(self, adapter_path: str):
        self.model_name = "memer"
        self.subgoal_type = "grounded_subgoal"
        self.image_size = (256, 256)
        
        self.system_prompt = "You are a robot program that predicts actions. The current input images from the front-view camera shows the most recent actions the robot has executed. The past keyframes are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else. Some tasks may have a video input for initial setup, some may not.\n\nReturn a JSON with:\n- current_subtask: the action that should be executed at the current timestep\n- keyframe_positions: list of frame positions (1-indexed) from the current input images where actions change"
        
        self.engine = PtEngine(
            model_id_or_path='Qwen/Qwen3-VL-4B-Instruct',
            adapters=[adapter_path],
            attn_impl='flash_attention_2' #'sdpa'
        )
        
        
        
    def _parse_box_patterns(self, subgoal: str, replacement: str = "scaled_coords", return_bbox: bool = False):
        """
        Parse box patterns from subgoal and replace them.
        
        Args:
            subgoal: The subgoal string containing box patterns
            replacement: Either "scaled_coords" to replace with <x, y> or "bbox" to replace with <bbox>
            return_bbox: If True, also return the list of bbox coordinates
        
        Returns:
            If return_bbox is False: modified subgoal string
            If return_bbox is True: tuple of (modified subgoal string, bbox list)
        """
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        
        qwen3_vl_image_size = (1000, 1000)
        
        if len(matches) == 0:
            if return_bbox:
                return subgoal, []
            return subgoal
        
        # Extract bbox coordinates (scaled)
        bbox = [[int(float(match[0])/qwen3_vl_image_size[1]*self.image_size[1]), 
                 int(float(match[1])/qwen3_vl_image_size[0]*self.image_size[0])] for match in matches]
        
        # Replace based on replacement type
        if replacement == "scaled_coords":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                lambda m: f'<{int(int(m.group(1)) * self.image_size[1] / qwen3_vl_image_size[1])}, {int(int(m.group(2)) * self.image_size[0] / qwen3_vl_image_size[0])}>',
                subgoal
            )
        elif replacement == "bbox":
            response = re.sub(
                r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
                '<bbox>',
                subgoal
            )
        else:
            raise ValueError(f"Invalid replacement type: {replacement}")
        
        if return_bbox:
            return response, bbox
        return response
    
    def _parse_subgoal_for_vla(self, subgoal: str) -> str:
        """Parse subgoal and replace box patterns with scaled coordinates for VLA."""
        subgoal = json.loads(subgoal)["current_subtask"]
        return self._parse_box_patterns(subgoal, replacement="scaled_coords", return_bbox=False)
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        """Preprocess grounded subgoal by replacing box patterns with <bbox> and extracting bbox coordinates."""
        return self._parse_box_patterns(subgoal, replacement="bbox", return_bbox=True)
    
    def start_new_episode(self, save_dir: str, video_query: List[np.ndarray]| None, task_goal: str = None, 
) -> dict:
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        ep_name = os.path.basename(save_dir)
        self.save_json_path = os.path.join(os.path.dirname(save_dir), f"{ep_name}_MemER_log.jsonl")
                
        if video_query is not None and len(video_query) > 0:
            imageio.mimsave(os.path.join(self.save_dir, f"step_0_video.mp4"), video_query, fps=30)
            self.video_path = os.path.join(self.save_dir, f"step_0_video.mp4")
        else:
            self.video_path = None
        self.task_goal = task_goal
        self.conversation_history = []
        self.total_images = []
        self.subgoals = []
        
        self.key_frame_paths = {} # {key_frame_id: image_path}
        self.execution_frame_paths = []
        
        self.last_response = None
     
    def _wrap_history_subgoals(self, subgoals) -> str:
        return "; ".join([f"{i+1}. {subgoal}" for i, subgoal in enumerate(subgoals)])
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        bbox = []
        # seatch the pattern "at <y, x>"
        matches = re.findall(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', subgoal)
        if matches:
            bbox = [[int(float(match[0])/1000*self.image_size[1]), int(float(match[1])/1000*self.image_size[0])] for match in matches]
        else:
            bbox = []        
        response = re.sub(
            r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>',
            '<bbox>',
            subgoal
        )
        
        return response, bbox
    
    
    def merge_key_frame_paths(self, dist: int = 8):
        key_frame_ids = list(self.key_frame_paths.keys())
        # merge the key frames, if the distance between two key frames is less than 10, merge them into one group, then select the midian idx as the key frame
        nums = sorted(key_frame_ids)

        groups = []
        cur = [nums[0]]

        for x in nums[1:]:
            if x - cur[-1] <= dist:
                cur.append(x)
            else:
                groups.append(cur)
                cur = [x]
        groups.append(cur)

        # median per group (for even size, pick "upper median" as int, tweak if you want average)
        meds = []
        for g in groups:
            meds.append(g[len(g)//2])
        
        self.key_frame_paths = {m: self.key_frame_paths[m] for m in meds}
        print("merged:", key_frame_ids, "->", meds)
    
    
    def update_history_subgoals(self, subgoal: str):
        response = json.loads(subgoal)
        current_subtask = response["current_subtask"]
        keyframe_positions = response["keyframe_positions"]
        
        if len(keyframe_positions) > 0:
            for key_id in keyframe_positions:
                path_str = self.current_execution_frame_paths[key_id-1]
                int_idx = int(re.search(r"step_(\d+)_image.png", path_str).group(1))
                self.key_frame_paths[int_idx] = path_str
        self.merge_key_frame_paths()
        return current_subtask
        
    
    
    def _wrap_images(self, image_paths) -> str:
        if len(image_paths) == 0:
            return "[]"
        return "[" + ", ".join(["<image>" for _ in image_paths]) + "]"
    
    
    def add_execution_frame(self, image_query: np.ndarray):
        image_path = os.path.join(self.save_dir, f"step_{len(self.execution_frame_paths)}_image.png")
        imageio.imwrite(image_path, image_query)
        self.execution_frame_paths.append(image_path)
        
    
    def _get_current_execution_frame_paths(self) -> list:
        if len(self.execution_frame_paths) == 1:
            return self.execution_frame_paths
        else:
            idx = -1 
            count = 8
            paths = []
            while count > 0:
                paths.insert(0, self.execution_frame_paths[idx])
                idx -= 2
                count -= 1
            return paths
        
    
    def prepare_infer_request(self) -> dict:
        
        video_prefix = "The task has a video input for initial setup: <video>\n" if self.video_path else "" 
        
        
        self.current_execution_frame_paths = self._get_current_execution_frame_paths()
        key_frame_paths = [key_path for key_path in self.key_frame_paths.values()]
        
        user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nHere are the selected frames from the entirety of the full execution that are of particular importance:{self._wrap_images(key_frame_paths)}\nHere is current input image list from the front-view camera: {self._wrap_images(self.current_execution_frame_paths)}\n\nWhat subtask should the robot execute and what is the keyframe position?"
        
        all_image_paths = key_frame_paths + self.current_execution_frame_paths
               
        infer_request_dict = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "images": all_image_paths
        }
        
        if self.video_path is not None:
            infer_request_dict["videos"] = [self.video_path]
        
        print("\n\n")
        pprint.pprint(infer_request_dict, width=800)
        
        with open(self.save_json_path, "a") as f:
            json.dump(infer_request_dict, f)
            f.write("\n")

        return InferRequest(**infer_request_dict)
    
    
    def call(self) -> str:        
        infer_request = self.prepare_infer_request()
        response = self.engine.infer([infer_request], request_config=RequestConfig(max_tokens=128, temperature=0))
        response = response[0].choices[0].message.content        
        print("Response: ", response)

        with open(self.save_json_path, "a") as f:
            json.dump({"response": response}, f)
            f.write("\n")

        try:
            self.update_history_subgoals(response)
            subgoal = self._parse_subgoal_for_vla(response)
        except Exception as e:
            print(f"Error updating history subgoals: {e}")
            subgoal = self.subgoals[-1]
                    
        return subgoal