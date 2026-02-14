from typing import Optional, Tuple, Any
from pathlib import Path

import os
import shutil
from env_runner import EnvRunner
from utils import EpisodeState, SUBGOAL_TYPES, TASK_WITH_VIDEO_DEMO

from subgoal_prediction.gemini.api import GeminiModel
from subgoal_prediction.gemini.prompts import (
    DEMO_TEXT_QUERY,
    IMAGE_TEXT_QUERY,
    VIDEO_TEXT_QUERY,
)

from subgoal_prediction.qwenvl.api import Qwen3VLModel
from subgoal_prediction.qwenvl.api_memer import Qwen3VLModelMemER


LONG_FIRST_ACTION_TASKS = [
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    
    "PickHighlight",
    "VideoRepick",
    
    "VideoPlaceButton",
    "VideoPlaceOrder",
    
    "MoveCube",
    "InsertPeg"
] # For Gemini only. Due to we found Gemini is very inconsistent for incremental video understanding, hard code to make it work better



class SubgoalPredictorBase:
    def __init__(
        self,
        args,
        save_dir: Path,
    ):
        self.args = args
        self.save_dir = save_dir
        self.video_buffer = []
        self.episode_dir: Optional[str] = None
        
        self.setup_api()

    def setup_api(self) -> None:
        pass

    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        self.env_name = env_runner.env_id
        self.episode_id = env_runner.episode_id
        self.task_goal = env_runner.task_goal
        self.env_runner = env_runner

    def step(self, epstate: EpisodeState) -> None:
        pass

    def maybe_extend_video(self, images: list) -> None:
        pass

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # return (subgoal_str, has_api_error)
        raise NotImplementedError

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        pass


class NullSubgoalPredictor(SubgoalPredictorBase):
    def get_subgoal(self, *args, **kwargs) -> Tuple[Optional[str], bool]:
        return None, False
    

class GeminiSubgoalPredictor(SubgoalPredictorBase):
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.api = GeminiModel(
            save_dir=os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}"),
            task_id=self.env_name,
            model_name=self.args.gemini_model_name,
            task_goal=self.task_goal,
            subgoal_type=self.args.subgoal_type,
        )
        self.video_buffer.extend(epstate.image_buffer[:-1])
        print(f"[robomme] Gemini agent for {self.args.subgoal_type}, task {self.env_name}, episode {self.episode_id}, setup finished")

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if not self._should_call(count):
            return current_subgoal, False

        text_query = self._get_text_query(count)
        input_data = self.api.prepare_input_data(self.video_buffer, text_query, count)
        response, _ = self.api.call(input_data)
        self.video_buffer.clear()

        if response is None:
            return None, True

        subgoal = response['subgoal']
        if "is complete" in subgoal or "is finished" in subgoal: # avoid using these subgoals as the final subgoal
            subgoal = last_subgoal
        return subgoal, False

    def end_episode(self, epstate: EpisodeState, success_flag: str) -> None:
        if not self.api:
            return
        self.api.save_conversation()
        self.api.prepare_input_data(
            epstate.image_buffer,
            self._get_text_query(epstate.count),
            epstate.count,
        )
        self.api.save_final_video(f"{success_flag}_ep{self.episode_id}_{self.task_goal}.mp4")
        self.api.clear_uploaded_files()
        del self.api

    def _get_text_query(self, count: int) -> str:
        if count == 0:
            if self.env_name in TASK_WITH_VIDEO_DEMO:
                template = DEMO_TEXT_QUERY
            else:
                template = IMAGE_TEXT_QUERY
        else:
            template = VIDEO_TEXT_QUERY
        return template.format(task_goal=self.task_goal)

    def _should_call(self, count: int) -> bool:
        if count == 0:
            return True
        if self.env_name in LONG_FIRST_ACTION_TASKS and count < 75:
            return False # avoid changing the first action too early
        return count % 48 == 0


class QwenVLSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        self.api = Qwen3VLModel(
            adapter_path=self.args.qwenvl_simpleSG_adapter_path if self.args.subgoal_type == "simple_subgoal" else self.args.qwenvl_groundSG_adapter_path,
            subgoal_type=self.args.subgoal_type,
        )
        print(f"[robomme] QwenVL {self.args.subgoal_type} agent setup finished")
        
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.video_buffer.append(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        # some tricks. QwenVL sometimes thinks the button has been pressed
        if self.env_name in ["ButtonUnmask", "PickHighlight"]:
            keep_period = 90
        elif self.env_name == "ButtonUnmaskSwap":
            if last_subgoal and "press the first button" in last_subgoal:
                keep_period = 100
            elif last_subgoal and "press the second button" in last_subgoal:
                keep_period = 250
            else:
                keep_period = 0
        else:
            keep_period = 0

        response = self.api.call(self.video_buffer[-1], count, keep_period)
        self.video_buffer.clear()
        return response, False


class MemERSubgoalPredictor(SubgoalPredictorBase):
    def setup_api(self) -> None:
        self.api = Qwen3VLModelMemER(adapter_path=self.args.memer_adapter_path)
        print("[robomme] MemER agent setup finished")
    
    def start_episode(self, epstate: EpisodeState, env_runner: EnvRunner) -> None:
        super().start_episode(epstate, env_runner)
        self.episode_dir = os.path.join(self.save_dir, self.env_name, f"ep{self.episode_id}")
        self.api.start_new_episode(self.episode_dir, epstate.image_buffer[:-1], self.task_goal)

    def step(self, epstate: EpisodeState) -> None:
        self.api.add_execution_frame(epstate.image_buffer[-1])

    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        response = self.api.call()
        return response, False

    def finalize(self, epstate: EpisodeState, success_flag: str) -> None:
        if self.episode_dir:
            shutil.rmtree(self.episode_dir) # save some space, you can comment this function out to keep all video frames


class OracleSubgoalPredictor(SubgoalPredictorBase):
    
    def setup_api(self) -> None:
        print("[robomme] Oracle agent setup finished")
    
    def get_subgoal(
        self,
        count: int,
        current_subgoal: Optional[str],
        last_subgoal: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        if self.args.subgoal_type == "simple_subgoal":
            return self.env_runner.simple_subgoal_oracle, False
        else:
            return self.env_runner.grounded_subgoal_oracle, False


def build_subgoal_predictor(
    args,
    save_dir: Path,
) -> SubgoalPredictorBase:
    if args.use_gemini:
        return GeminiSubgoalPredictor(args, save_dir)
    if args.use_qwenvl:
        return QwenVLSubgoalPredictor(args, save_dir)
    if args.use_memer:
        return MemERSubgoalPredictor(args, save_dir)
    if args.use_oracle:
        return OracleSubgoalPredictor(args, save_dir)
    
    return NullSubgoalPredictor(args, save_dir)



