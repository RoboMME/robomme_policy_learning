import os
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import re

from openpi.training import config as _config
from openpi.training.data_loader import Dataset
from mme_vla_suite.models.config.schema import HistoryConfig, PastFramesConfig
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
import pickle

random.seed(0)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode cache — mmap-friendly per-episode npy arrays built by
# ``scripts/build_episode_cache.py``.
# ---------------------------------------------------------------------------

class EpisodeCache:
    """Lazy-loading, mmap-backed per-episode arrays.

    Produced by ``scripts/build_episode_cache.py``.  Structure on disk::

        episode_cache/episode_{E}/images.npy
                                  wrist_images.npy
                                  states.npy
                                  actions.npy
                                  meta.json   # step_to_local / local_to_step
    """

    def __init__(self, cache_root: Path):
        self._root = cache_root
        self._meta: dict[int, dict] = {}
        self._arrays: dict[int, dict[str, np.ndarray]] = {}

    @property
    def available(self) -> bool:
        return self._root.is_dir()

    def _ensure_episode(self, epis_idx: int) -> None:
        if epis_idx in self._meta:
            return
        ep_dir = self._root / f"episode_{epis_idx}"
        with open(ep_dir / "meta.json", encoding="utf-8") as f:
            self._meta[epis_idx] = json.load(f)
        self._arrays[epis_idx] = {
            "images": np.load(ep_dir / "images.npy", mmap_mode="r"),
            "wrist_images": np.load(ep_dir / "wrist_images.npy", mmap_mode="r"),
            "states": np.load(ep_dir / "states.npy", mmap_mode="r"),
            "actions": np.load(ep_dir / "actions.npy", mmap_mode="r"),
        }

    def get(
        self, epis_idx: int, step_idx: int, action_horizon: int,
    ) -> dict[str, np.ndarray] | None:
        """Return (image, wrist_image, state, actions) for one step, or *None* if missing."""
        self._ensure_episode(epis_idx)
        local = self._meta[epis_idx]["step_to_local"].get(str(step_idx))
        if local is None:
            return None
        arrs = self._arrays[epis_idx]
        return {
            "image": np.array(arrs["images"][local]),
            "wrist_image": np.array(arrs["wrist_images"][local]),
            "state": np.array(arrs["states"][local], dtype=np.float32),
            "actions": np.array(arrs["actions"][local][:action_horizon], dtype=np.float32),
        }


def load_episode_step_index(dataset_path: str | Path) -> dict[tuple[int, int], int]:
    """Load ``meta/episode_step_to_sample_idx.json`` built by ``scripts/build_episode_step_index.py``."""
    path = Path(dataset_path) / "meta" / "episode_step_to_sample_idx.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run: uv run scripts/build_episode_step_index.py {dataset_path}"
        )
    with open(path, encoding="utf-8") as f:
        raw: dict[str, int] = json.load(f)
    out: dict[tuple[int, int], int] = {}
    for k, v in raw.items():
        parts = k.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Bad key in episode_step index: {k!r}")
        out[(int(parts[0]), int(parts[1]))] = int(v)
    return out


def load_vector_file(vector_path: str, step_idx: int) -> tuple[dict, int]:
    with open(vector_path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx
    
    

class SampleDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.stats = json.load(open(os.path.join(self.dataset_path, "meta", "stats.json")))
    
    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        else:
            return self.stats["total_samples"]
    
    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path,  "data", f"{idx}.pkl"), "rb") as f:
            return pickle.load(f)


class RoboMMEDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_config: _config.DataConfig,
        history_config: HistoryConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
        past_frames_config: PastFramesConfig | None = None,
    ):
        self.history_config = history_config
        self.past_frames_config = past_frames_config

        self.action_horizon = action_horizon
        self.dataset = SampleDataset(dataset_path)

        self._episode_cache: EpisodeCache | None = None
        self._epis_step_to_idx: dict[tuple[int, int], int] | None = None
        if past_frames_config is not None:
            cache = EpisodeCache(Path(self.dataset.dataset_path) / "episode_cache")
            if cache.available:
                self._episode_cache = cache
                logger.info("Using mmap episode cache for past frames")
            else:
                self._epis_step_to_idx = load_episode_step_index(self.dataset.dataset_path)
                logger.info("Episode cache not found — falling back to per-pkl loading")

        self.feature_dir = Path(self.dataset.dataset_path) / "features"

        if self.history_config is not None:
            self.img_emb_dim = self.history_config.memory_feature.img.input_dim
            self.pos_emb_dim = self.history_config.memory_feature.pos.input_dim
            self.state_emb_dim = self.history_config.memory_feature.state.input_dim
            self.num_views = self.history_config.num_views
            self.streaming_obs_horizon = self.history_config.streaming_obs_horizon
        
            if self.history_config.representation_type == "perceptual":
                self.mem_buffer = MemoryBuffer(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    token_drop_stride=self.streaming_obs_horizon // 2,
                )
            elif self.history_config.representation_type == "recurrent":
                self.mem_buffer = MemoryBufferRecurrent(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    input_obs_horizon=self.streaming_obs_horizon,
                    max_recur_steps=self.history_config.recurrent_memory.max_recur_steps,
                    max_video_steps=self.history_config.recurrent_memory.max_pretraj_steps,
                )
            else:
                # symbolic memory does not need mem_buffer
                self.mem_buffer = None
        else:
            logger.info("=== Do not use history ===")
        
        self.compute_norm_stats = compute_norm_stats
        
        if not compute_norm_stats:
            self.state_norm_stats = data_config.norm_stats['state']
            self.use_quantiles = data_config.use_quantile_norm
        
    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        history_feats = {}
        history_paths = []
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(
                os.path.join(self.feature_dir, f"episode_{epis_idx}", f"token_emb_{idx}.npy")
            )
        
        if self.history_config.representation_type == "recurrent":
            # load_vector_fn = load_vector_file_lru4096 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(48, max(4, len(history_paths)))
        else:
            # load_vector_fn = load_vector_file_lru1024 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(36, max(4, len(history_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_vector_fn, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats
    
    
    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_indices = json.load(open(os.path.join(self.feature_dir, f"episode_{epis_idx}", "kept_indices.json")))
        
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat, 
            kept_indices=kept_indices, epis_idx=epis_idx)
        

    def prepare_frame_sampling(self,  epis_idx,  step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget

        return self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat, 
            epis_idx=epis_idx)


    
    def prepare_token_recurrent(self, epis_idx, step_idx, exec_start_idx):      
        self.mem_buffer.exec_start_idx = exec_start_idx
        
        return self.mem_buffer.prepare_token_recurrent(
            step_idx, exec_start_idx, self._gather_history_feat,
            epis_idx=epis_idx)
        
    
    def __len__(self):
        return len(self.dataset)
    
    
    def _normalize_state(self, state):
        if self.compute_norm_stats:
            return state
        else:
            if self.use_quantiles:
                return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
            else:
                return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)
    
    def _truncated_gaussian_noise(self, max_val: int, std: float = None) -> int:
        """Generate truncated Gaussian noise centered at 0, clamped to [-max_val, max_val]."""
        if std is None:
            std = max_val / 2.5  # ~95% of samples within range before clamping
        noise = random.gauss(0, std)
        noise = max(-max_val, min(max_val, noise))  # clamp to [-max_val, max_val]
        return int(round(noise))

    def add_grounding_augmentation(self, subgoal: str, noise_range: int = 8) -> str:
        if not subgoal:
            return subgoal
        
        matches = re.findall(r'at <(\d+), (\d+)>', subgoal)
        
        if len(matches) == 0 or len(matches) > 1:
            return subgoal
        
        x, y = matches[0]
        x = int(x)
        y = int(y)
        noise_x = self._truncated_gaussian_noise(noise_range)
        noise_y = self._truncated_gaussian_noise(noise_range)

        new_subgoal = subgoal.replace(f'at <{x}, {y}>', f'at <{x + noise_x}, {y + noise_y}>')
        return new_subgoal

    def _gather_past_frames(self, data: dict) -> dict[str, np.ndarray]:
        """Load N past steps at stride (see ``PastFramesConfig``).

        Uses the mmap episode cache when available, otherwise falls back to
        per-pkl loading (slower).  Pads with zeros where the step doesn't exist.
        """
        cfg = self.past_frames_config
        assert cfg is not None
        epis_idx = int(data["epis_idx"].item())
        step_idx = int(data["step_idx"].item())
        N = cfg.num_past
        stride = cfg.stride

        img_ref = np.asarray(data["image"])
        wrist_ref = np.asarray(data["wrist_image"])
        state_ref = np.asarray(data["state"])
        act_ref = np.asarray(data["actions"])

        past_image = np.zeros((N, *img_ref.shape), dtype=img_ref.dtype)
        past_wrist = np.zeros((N, *wrist_ref.shape), dtype=wrist_ref.dtype)
        past_state = np.zeros((N,) + state_ref.shape, dtype=np.float32)
        past_actions = np.zeros((N,) + act_ref.shape, dtype=act_ref.dtype)
        mask = np.zeros((N,), dtype=np.bool_)

        use_cache = self._episode_cache is not None
        data_root = Path(self.dataset.dataset_path) / "data" if not use_cache else None

        for i in range(N):
            t = step_idx - (i + 1) * stride
            if use_cache:
                past = self._episode_cache.get(epis_idx, t, self.action_horizon)
            else:
                sample_idx = self._epis_step_to_idx.get((epis_idx, t))
                if sample_idx is None:
                    past = None
                else:
                    pkl_path = data_root / f"{sample_idx}.pkl"
                    with open(pkl_path, "rb") as f:
                        raw = pickle.load(f)
                    past = {
                        "image": raw["image"],
                        "wrist_image": raw["wrist_image"],
                        "state": np.asarray(raw["state"], dtype=np.float32),
                        "actions": np.asarray(raw["actions"])[:self.action_horizon],
                    }
            if past is None:
                continue
            past_image[i] = past["image"]
            past_wrist[i] = past["wrist_image"]
            past_state[i] = past["state"]
            past_actions[i] = past["actions"]
            mask[i] = True

        return {
            "past_image": past_image,
            "past_wrist_image": past_wrist,
            "past_state": past_state,
            "past_actions": past_actions,
            "past_frame_mask": mask,
        }

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        data["actions"] = data["actions"][:self.action_horizon]
        
        # During online evaluation, the ground-truth subgoal may change earlier than when it was recorded.
        # To make the model robust to this temporal shift and avoid train/test distribution mismatch,
        # we randomly sample from either subgoal or subgoal_online (early change).
        if self.history_config is not None \
            and self.history_config.representation_type == "symbolic" \
            and self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"] \
            and random.random() < 0.5: 
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")

        if self.past_frames_config is not None:
            data.update(self._gather_past_frames(data))
        
        if self.history_config is not None and self.history_config.representation_type == "symbolic":
            data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
            data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)
 
        
        if self.history_config is not None:
            # use history
            epis_idx = data["epis_idx"].item()
            step_idx = data["step_idx"].item()
            exec_start_idx = data["exec_start_idx"].item()

            if self.history_config.representation_type == "perceptual":
                if self.history_config.perceptual_memory.type == "token_dropping":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # >=64
                    ) = self.prepare_token_drop(epis_idx, step_idx)
                else:
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # slow >=64, mid >=16,fast >=4
                    ) = self.prepare_frame_sampling(epis_idx, step_idx)
                
                data["static_image_emb"] = static_img_emb
                data["static_pos_emb"] = static_pos_emb
                data["static_state_emb"] = self._normalize_state(static_state_emb)
                data["static_mask"] = static_mask

            elif self.history_config.representation_type == "recurrent":             
                recur_img_emb, recur_pos_emb, recur_state_emb, recur_mask = (
                    self.prepare_token_recurrent(
                        epis_idx, step_idx, exec_start_idx)
                )

                data["recur_image_emb"] = recur_img_emb
                data["recur_pos_emb"] = recur_pos_emb
                data["recur_state_emb"] = self._normalize_state(recur_state_emb)
                data["recur_mask"] = recur_mask  # 1 ~ max_exec_steps

            elif self.history_config.representation_type == "symbolic":
                pass
            else:
                raise ValueError(
                    f"Not supported representation type: {self.history_config.representation_type}"
                )
        
        
        for key in [
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            
            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",
            
            "simple_subgoal",
            "grounded_subgoal",
            "prompt"
        ]:
            if key not in data:
                data[key] = None        

        return data