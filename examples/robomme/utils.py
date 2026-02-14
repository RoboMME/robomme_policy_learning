import os
import numpy as np
import cv2
import imageio
import re
import collections
from typing import Tuple, Optional



TASK_WITH_VIDEO_DEMO = [
    "VideoUnmask", "VideoUnmaskSwap", "VideoPlaceButton", "VideoPlaceOrder",
    "VideoRepick", "MoveCube", "InsertPeg", "PatternLock", "RouteStick"
]

TASK_NAME_LIST=  [      
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick"
]

SUBGOAL_TYPES = ("simple_subgoal", "grounded_subgoal")



def pack_buffer(image_buffer, state_buffer, exec_start_idx=0):
    image_output = np.stack(image_buffer, axis=0).astype(np.uint8)[:, None]
    state_output = np.stack(state_buffer, axis=0).astype(np.float32)
    return {
        "images": image_output,
        "state": state_output,
        "add_buffer": True,
        "exec_start_idx": exec_start_idx,
    }
    
def check_args(args):
    assert args.subgoal_type in ["simple_subgoal", "grounded_subgoal", None] and args.obs_horizon == 16
    if args.use_memer:
        args.subgoal_type = "grounded_subgoal"



class EpisodeState:
    def __init__(self):
        self.image_buffer = []
        self.wrist_image_buffer = []
        self.state_buffer = []
        self.action_plan = collections.deque()
        self.count = 0
        self.exec_start_idx = 0

    def add_observation(self, img: np.ndarray, wrist_img: np.ndarray, state: np.ndarray):
        self.image_buffer.append(img.copy())
        self.wrist_image_buffer.append(wrist_img.copy())
        self.state_buffer.append(state.copy())

    def clear_buffers(self):
        self.image_buffer.clear()
        self.wrist_image_buffer.clear()
        self.state_buffer.clear()
        self.exec_start_idx = 0

    def get_current_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.image_buffer[-1], self.wrist_image_buffer[-1], self.state_buffer[-1]





class RolloutRecorder:
    def __init__(self, save_dir: str, task_goal: str, fps: int = 30):
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        self.total_images = []
        self.fps = fps
        self.task_goal = task_goal
        
    def _extract_points(self, subgoal: str):
        match = re.findall(r'<(\d+), (\d+)>', subgoal)
        points = []
        for m in match:
            points.append((int(m[0]), int(m[1])))
        return points
        
    def record(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, action: np.ndarray=None, is_video_demo: bool=False, subgoal: Optional[str] = None):
        
        concat_image = np.concatenate([image, wrist_image], axis=1)
        if is_video_demo: # add a red border
            concat_image = cv2.rectangle(concat_image, (0, 0), (concat_image.shape[1], concat_image.shape[0]), (255, 0, 0), 10)
        
        frame_text = "Frame: " + str(len(self.total_images))
        frame_text_area = self.add_text_area(frame_text, concat_image.shape)
        
        goal_text = "Task Goal: " + self.task_goal
        goal_text_area = self.add_text_area(goal_text, concat_image.shape)
        
        if subgoal is not None:
            subgoal_text = "Subgoal: " + subgoal
            subgoal_text_area = self.add_text_area(subgoal_text, concat_image.shape)
            
            if self._extract_points(subgoal) is not None:
                for point in self._extract_points(subgoal):
                    concat_image = cv2.circle(concat_image, point[::-1], 5, (255, 255, 0), -1)
                    
            concat_image = np.concatenate([subgoal_text_area, concat_image], axis=0)
        
        state_text = "State: " + ','.join([f"{i:.4f}" for i in state])
        state_text_area = self.add_text_area(state_text, concat_image.shape)
        
        action_text = 'Action: ' + ','.join([f"{i:.4f}" for i in action]) if action is not None else "Action:None"
        action_text_area = self.add_text_area(action_text, concat_image.shape)
        # Concatenate text area on top of image
        concat_image = np.concatenate([frame_text_area, goal_text_area, action_text_area, state_text_area, concat_image], axis=0)
        self.total_images.append(concat_image)
    
    def add_text_area(self, text: str, concat_image_shape: tuple):        
        # Calculate text wrapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        max_width = concat_image_shape[1] - 20  # Leave 10px margin on each side
        lines = []
        words = text.replace(',', ' ').split()
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + ' ' + word
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        
        # Create text area with dynamic height
        line_height = 20
        text_area_height = max(50, len(lines) * line_height + 10)
        text_area = np.zeros((text_area_height, concat_image_shape[1], 3), dtype=np.uint8)
        
        # Draw each line
        for i, line in enumerate(lines):
            y_position = 15 + i * line_height
            text_area = cv2.putText(text_area, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
        
        return text_area
             
    def save_video(self, filename: str):
        imageio.mimsave(os.path.join(self.save_dir, filename), self.total_images, fps=self.fps)