import os
import json
import time
import imageio
import re
import shutil
import cv2

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, List
import google.generativeai as genai

from subgoal_prediction.gemini.prompts import *



def parse_markdown_json(text):
    """Parse JSON that may be wrapped in markdown code fences."""
    text = text.strip()
    match = re.search(r'["`]subgoal[`"]\s*:(.*?)\n', text, re.DOTALL)

    if match:
        subgoal_str = match.group(1).strip()
        subgoal_str = re.search(r'"(.*?)"', subgoal_str).group(1)
        out = {"subgoal": subgoal_str}
    else:
        try: 
            subgoal = json.loads(text)
            out = {"subgoal": subgoal["subgoal"]}
        except Exception as e:
            return None
    return out



class BaseModel(ABC):
    """Base class for vision-language models"""
    
    def __init__(self, save_dir: str = None, task_id: str = None, 
                 model_name: str = None, task_goal: str = None, 
                 subgoal_type: str = "simple_subgoal", 
                 image_size: tuple = (256, 256)):
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        self.task_id = task_id
        self.conversation_history = []
        self.model_name = model_name
        self.subgoal_type = subgoal_type
        self.image_size = image_size
        
        assert subgoal_type in ["simple_subgoal", "grounded_subgoal"]
        
        # Load appropriate prompt dictionary
        if subgoal_type == "simple_subgoal":
            _prompt_dict = prompt_dict_simple
        elif subgoal_type == "grounded_subgoal":
            _prompt_dict = prompt_dict_grounded
        else:
            raise ValueError(f"Invalid subgoal type: {subgoal_type}")
        
        system_prompt = _prompt_dict[task_id].replace("<task_goal>", task_goal)
        print(f"\n\n\"\"\"\nSystem prompt:\n{system_prompt}\n\"\"\"\n\n")
        
        self.init_model(system_prompt, model_name)
        
        self.total_images = []
        self.subgoals = []
        self.last_subgoal = None
    
    
    def _normalize_point(self, point: tuple, image_size: tuple | None) -> tuple:
        if image_size is None:
            return point
        # return is [0,1000], normalized to the image size
        x = int(point[1] * image_size[1] / 1000)
        x = np.clip(x, 0, image_size[1])
        y = int(point[0] * image_size[0] / 1000)
        y = np.clip(y, 0, image_size[0])
        return y, x


    def normalize_point_in_response(self, response: dict) -> dict:
        points = []
        if self.subgoal_type == "grounded_subgoal":
            subgoal = response['subgoal']
            matches = re.findall(r'<(\d+), (\d+)>', subgoal)
            for match in matches:
                y = int(match[0])
                x = int(match[1])
                point = self._normalize_point((y, x), self.image_size)
                points.append(point)
                original = f'<{match[0]}, {match[1]}>'
                subgoal = subgoal.replace(original, f'<{point[0]}, {point[1]}>')
            response['subgoal'] = subgoal
            
        return response, points
    
    @abstractmethod
    def init_model(self, system_prompt: str, model_name: str):
        """Initialize the model with system prompt"""
        pass
    
    def call(self, input_data: dict) -> Optional[str]:
        """Main entry point for processing inputs"""
        image_path = input_data.get('image_path')
        video_path = input_data.get('video_path')
        text_query = input_data.get('text_query', '')
        
        try:
            print('--------------------------------')
            if image_path is not None:
                print(f"Processing image: {image_path}")
                print(f"Text query: {text_query}")
                response = self._process_image(image_path, text_query)
            elif video_path is not None:
                print(f"Processing video: {video_path}")
                print(f"Text query: {text_query}")
                response = self._process_video(video_path, text_query)
            else:
                print(f"Text query: {text_query}")
                response = self._process_text(text_query)
                
            print(f"\nResponse:\n{response}")
            print('--------------------------------')
            response = parse_markdown_json(response)
            response, points = self.normalize_point_in_response(response)
            self.last_subgoal = response['subgoal']
            return response, points
        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            self.last_subgoal = None
            return None, None
    
    def prepare_input_data(self, image_query: Union[np.ndarray, List[np.ndarray]], text_query: str, step_idx: int) -> dict:
        """Prepare input data for the model"""
        if not isinstance(image_query, list) or (isinstance(image_query, list) and len(image_query) <= 3):
            # will consider it as a single image, take the last one
            imageio.imwrite(os.path.join(self.save_dir, f"step_{step_idx}_image.png"), image_query[-1])
            input_data = {
                "image_path": os.path.join(self.save_dir, f"step_{step_idx}_image.png"),
                "video_path": None,
                "text_query": text_query,
            }
            self.total_images.append(image_query[-1])
            self.subgoals.append(self.last_subgoal)
            return input_data
        else:
            self.total_images.extend(image_query)
            self.subgoals.extend([self.last_subgoal] * len(image_query))
            imageio.mimsave(os.path.join(self.save_dir, f"step_{step_idx}_video.mp4"), image_query, fps=30)
            input_data = {
                "image_path": None,
                "video_path": os.path.join(self.save_dir, f"step_{step_idx}_video.mp4"),
                "text_query": text_query,
            }
            return input_data
    
    def save_conversation(self):
        """Save conversation history to JSON"""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "conversation.json"), "w") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)

    def _wrap_text(self, text: str, font, font_scale, thickness, max_width):
        """Wrap text to fit within max_width pixels"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            
            if text_width <= max_width - 20:  # 20 pixels margin
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    def save_final_video(self, video_name: str):
        if len(self.total_images) != len(self.subgoals):
            raise ValueError(f"Total images and subgoals must have the same length")
        final_images = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 25
        
        max_text_height = 80
        last_subgoal = None
        for i, (image, subgoal) in enumerate(zip(self.total_images, self.subgoals)):
            if subgoal is None:
                subgoal = "[initializing...]"
            
            if isinstance(subgoal, dict):
                subgoal = subgoal['action'] if subgoal['point'] is None else f"{subgoal['action']} <{subgoal['point'][0]}, {subgoal['point'][1]}>"
            
            # Wrap text to fit image width
            lines = self._wrap_text(subgoal, font, font_scale, thickness, image.shape[1])
            
            # Create black image with the maximum text height for all frames
            black_image = np.zeros((image.shape[0] + max_text_height, image.shape[1], 3), dtype=np.uint8)
                        
            # Add each line of text
            for i, line in enumerate(lines):
                y_position = (i + 1) * line_height
                black_image = cv2.putText(black_image, line, (10, y_position), font, font_scale, (255, 255, 255), thickness)
            
            black_image[max_text_height:, :] = image
            final_images.append(black_image)
            if subgoal != last_subgoal:
                final_images.extend([black_image] * 30)
            last_subgoal = subgoal
        imageio.mimsave(os.path.join(self.save_dir, video_name), final_images, fps=45)
        
        
    def add_frame_hold(self, image: np.ndarray, hold_len: int = 10):
        for i in range(hold_len):
            self.total_images.append(image.copy())
            self.subgoals.append(self.last_subgoal)
    
class GeminiModel(BaseModel):
    """Gemini model implementation"""
    
    def init_model(self, system_prompt: str, model_name: str):
        self.model = genai.GenerativeModel(
            model_name=model_name or 'gemini-2.5-flash-lite',
            system_instruction=system_prompt
        )
        self.chat = self.model.start_chat(history=[])
        
        self.all_uploaded_file_names = []
    
    def _process_image(self, image_path: Union[str, List[str]], 
                      text_query: str = "What should the robot do in this situation?") -> str:
        if isinstance(image_path, list):
            image_files = [genai.upload_file(path=path) for path in image_path]
            self.all_uploaded_file_names.extend([image_file.name for image_file in image_files])
            while all(file.state.name == "PROCESSING" for file in image_files):
                time.sleep(0.1)
                image_files = [genai.get_file(file.name) for file in image_files]
            if any(file.state.name == "FAILED" for file in image_files):
                raise ValueError(f"Image processing failed")
            
            response = self.chat.send_message([text_query, *image_files])
            
            self.conversation_history.append({
                "turn": len(self.conversation_history) + 1,
                "type": "image",
                "path": [os.path.basename(path) for path in image_path],
                "query": text_query,
                "response": response.text
            })
        else:
            image_file = genai.upload_file(path=image_path)
            self.all_uploaded_file_names.append(image_file.name)
            
            while image_file.state.name == "PROCESSING":
                time.sleep(0.1)
                image_file = genai.get_file(image_file.name)
            
            if image_file.state.name == "FAILED":
                raise ValueError(f"Image processing failed")
            
            response = self.chat.send_message([text_query, image_file])
            
            self.conversation_history.append({
                "turn": len(self.conversation_history) + 1,
                "type": "image",
                "path": os.path.basename(image_path),
                "query": text_query,
                "response": response.text
            })
        
        return response.text
    
    def _process_video(self, video_path: str, 
                      text_query: str = "What should the robot do based on this video?") -> str:
        video_file = genai.upload_file(path=video_path)
        self.all_uploaded_file_names.append(video_file.name)
        
        while video_file.state.name == "PROCESSING":
            time.sleep(0.1)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed")
        
        response = self.chat.send_message([text_query, video_file])
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "video",
            "path": os.path.basename(video_path),
            "query": text_query,
            "response": response.text
        })
        
        return response.text
    
    def _process_text(self, user_query: str) -> str:
        response = self.chat.send_message(user_query)
        
        self.conversation_history.append({
            "turn": len(self.conversation_history) + 1,
            "type": "text",
            "query": user_query,
            "response": response.text
        })
        
        return response.text
    
    def clear_uploaded_files(self):
        try:
            for file_name in self.all_uploaded_file_names:
                genai.delete_file(file_name)
            print(f"Cleared {len(self.all_uploaded_file_names)} uploaded files")
            self.all_uploaded_file_names = []
        except Exception as e:
            print(f"Error clearing uploaded files: {e}")