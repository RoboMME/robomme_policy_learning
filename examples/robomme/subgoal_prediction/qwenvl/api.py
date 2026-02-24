import os
import shutil
import numpy as np
import imageio
from typing import List
import os
import re
import pprint
os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
os.environ['FPS_MAX_FRAMES'] = '10'

from swift.llm import PtEngine, InferRequest, RequestConfig

class Qwen3VLModel:
    
    def __init__(self, 
        adapter_path: str,
        subgoal_type: str = "simple_subgoal", 
    ):
        self.model_name = "qwenvl"
        self.subgoal_type = subgoal_type
        self.image_size = (256, 256)
        
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal"]
        
        # Load appropriate prompt dictionary
        if subgoal_type == "simple_subgoal":
            self.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of language subgoals"
        elif subgoal_type == "grounded_subgoal":
            self.system_prompt = "You are a helpful assistant to help guide the robot to complete the task by predicting a sequence of grounded language subgoals"
        else:
            raise ValueError(f"Invalid subgoal type: {subgoal_type}")
        
        self.engine = PtEngine(
            model_id_or_path="/nfs/turbo/coe-chaijy-unreplicated/pre-trained-weights/Qwen3-VL-4B-Instruct", #'Qwen/Qwen3-VL-4B-Instruct',
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
        return self._parse_box_patterns(subgoal, replacement="scaled_coords", return_bbox=False)
    
    def _parse_grounded_subgoal(self, subgoal) -> tuple:
        """Preprocess grounded subgoal by replacing box patterns with <bbox> and extracting bbox coordinates."""
        return self._parse_box_patterns(subgoal, replacement="bbox", return_bbox=True)
    
    def start_new_episode(self, save_dir: str, video_query: List[np.ndarray]| None, task_goal: str = None) -> dict:
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        if video_query is not None and len(video_query) > 0:
            imageio.mimsave(os.path.join(self.save_dir, f"step_0_video.mp4"), video_query, fps=30)
            self.video_path = os.path.join(self.save_dir, f"step_0_video.mp4")
        else:
            self.video_path = None
        self.task_goal = task_goal
        self.conversation_history = []
        self.total_images = []
        self.subgoals = []
        self.history_simple_subgoals = []
        self.history_grounded_subgoals = []
        self.history_grounded_bboxes = []
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
    
    def update_history_subgoals(self, subgoal: str):
        if self.subgoal_type == "simple_subgoal":
            if self.history_simple_subgoals:
                if self.history_simple_subgoals[-1] != subgoal:
                    self.history_simple_subgoals.append(subgoal)
            else:
                self.history_simple_subgoals.append(subgoal)
        else:
            assistant_prompt, bbox = self._parse_grounded_subgoal(subgoal)
            if self.history_grounded_subgoals:
                if self.history_grounded_subgoals[-1] != assistant_prompt:
                    self.history_grounded_subgoals.append(assistant_prompt)
                    self.history_grounded_bboxes.extend(bbox)
            else:
                self.history_grounded_subgoals.append(assistant_prompt)
                self.history_grounded_bboxes.extend(bbox)
    
    def prepare_infer_request(self, image_query: np.ndarray, step_idx: int) -> dict:
        
        image_path = os.path.join(self.save_dir, f"step_{step_idx}_image.png")
        imageio.imwrite(image_path, image_query)
        video_prefix = "<video>" if self.video_path else ""
        
        if self.subgoal_type == "simple_subgoal":            
            if len(self.history_simple_subgoals) == 0:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThis is the initial turn for prediction\n<image>What's the next language subgoal based on current observation?"
            else:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThe history of previous predicted language subgoals are: {self._wrap_history_subgoals(self.history_simple_subgoals)}\n<image>What's the next language subgoal based on current observation?"
                    
        else:        
            if len(self.history_grounded_subgoals) == 0:
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThis is the initial turn for prediction\n<image>What's the next grounded language subgoal based on current observation?"
            else:            
                user_prompt = f"{video_prefix}The task goal is: {self.task_goal}\nThe history of previous predicted grounded language subgoals are: {self._wrap_history_subgoals(self.history_grounded_subgoals)}\n<image>What's the next grounded language subgoal based on current observation?"
        
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
            "images": [image_path]
        }
        
        if self.video_path is not None:
            infer_request_dict["videos"] = [self.video_path]
            
        if self.subgoal_type == "grounded_subgoal":
            infer_request_dict["objects"] = {"ref": [], "bbox": self.history_grounded_bboxes}
        
        print("\n\n")
        pprint.pprint(infer_request_dict)

        return InferRequest(**infer_request_dict)
    
    
    def call(self, image_query: np.ndarray, step_idx: int, keep_period: int = 0) -> str:        
        if step_idx <= keep_period and self.last_response is not None:
            # some tasks that require press button, qwen models always skip
            # add some hard-coded rules to fix it
            response = self.last_response
        else:
            infer_request = self.prepare_infer_request(image_query, step_idx)
            response = self.engine.infer([infer_request], request_config=RequestConfig(max_tokens=128, temperature=0))
            response = response[0].choices[0].message.content
        
        print("Response: ", response)
        self.last_response = response
        self.update_history_subgoals(response)
        return self._parse_subgoal_for_vla(response)