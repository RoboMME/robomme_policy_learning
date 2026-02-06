"""
Use one layer of recurrent models to condense the history tokens into fixed sized tokens.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops
import time
from typing import Any
import numpy as np
from omegaconf import OmegaConf, DictConfig

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.recur_mem import RecurrentMemory


from tqdm import tqdm
cfg = OmegaConf.load(
    "/home/daiyp/openpi/src/mme_vla_suite/models/config/robomme/bg512-input-recurrent-ttt.yaml")

rngs = nnx.Rngs(0)
block = RecurrentMemory(cfg, rngs)
v = cfg.num_views
t = cfg.recurrent_memory.max_recur_steps
p = cfg.token_per_image
d1 = cfg.memory_feature.img.input_dim
d2 = cfg.memory_feature.pos.input_dim
d3 = cfg.memory_feature.state.input_dim

batch_size = 1


all_states = None

mapping = jax.jit(block.__call__)

for _ in tqdm(range(1000)):
    recur_image_emb_np = np.random.normal(0, 1, (batch_size, t, v, p, d1))
    recur_pos_emb_np = np.random.normal(0, 1, (batch_size, t, v, p, d2))
    recur_state_emb_np = np.random.normal(0, 1, (batch_size, t, d3))
    recur_mask_np = np.random.randint(0, 2, (batch_size, t))
    recur_mask = jnp.array(recur_mask_np, dtype=jnp.bool_)
    # 0:60 are padding, 60:62 pre-loaded, 62:64 execution
    recur_pos_emb = jnp.array(recur_pos_emb_np, dtype=jnp.float32)
    recur_state_emb = jnp.array(recur_state_emb_np, dtype=jnp.float32)
    recur_image_emb = jnp.array(recur_image_emb_np, dtype=jnp.float32)
    
    time_taken = time.time()
    (tokens_ref, input_mask_ref), all_states_new_ref, stats_ref = mapping(
        recur_image_emb, recur_mask, recur_pos_emb, recur_state_emb, all_states
    )
    time_taken = time.time() - time_taken
    print(f"Time taken: {time_taken}")


# # test online evaluation
# all_states_online = None

# preload_image_emb_np = recur_image_emb_np[:, 60:62]
# preload_pos_emb_np = recur_pos_emb_np[:, 60:62]
# preload_state_emb_np = recur_state_emb_np[:, 60:62]
# preload_mask_np = recur_mask_np[:, 60:62]

# from mme_vla_suite.shared.data_utils import left_padding_token_emb

# preload_image_emb, preload_pos_emb, preload_state_emb, preload_mask = left_padding_token_emb(
#     preload_image_emb_np[0], preload_pos_emb_np[0], preload_state_emb_np[0], preload_mask_np[0], 64
# )
# preload_image_emb = jnp.array(preload_image_emb, dtype=jnp.float32)
# preload_pos_emb = jnp.array(preload_pos_emb, dtype=jnp.float32)
# preload_state_emb = jnp.array(preload_state_emb, dtype=jnp.float32)
# preload_mask = jnp.array(preload_mask, dtype=jnp.bool_)


# (tokens_online, input_mask_online), all_states_online, stats_online = mapping(
#     preload_image_emb[None, ...], preload_mask[None, ...], preload_pos_emb[None, ...], preload_state_emb[None, ...], all_states_online
# )

# # all_states_online = jax.tree_util.tree_map(lambda x: x[0], all_states_online_batch[0]), all_states_online_batch[1], all_states_online_batch[2]

# for i in range(62, 64):
#     this_image_emb_np = recur_image_emb_np[:, i]
#     this_pos_emb_np = recur_pos_emb_np[:, i]
#     this_state_emb_np = recur_state_emb_np[:, i]
#     this_mask_np = recur_mask_np[:, i]
#     this_image_emb = jnp.array(this_image_emb_np, dtype=jnp.float32)
#     this_pos_emb = jnp.array(this_pos_emb_np, dtype=jnp.float32)
#     this_state_emb = jnp.array(this_state_emb_np, dtype=jnp.float32)
#     this_mask = jnp.array(this_mask_np, dtype=jnp.bool_)
    

#     (tokens_online, input_mask_online), all_states_online_batch, stats_online = mapping(
#         this_image_emb[None, ...], this_mask[None, ...], this_pos_emb[None, ...], this_state_emb[None, ...], all_states_online)
#     all_states_online = jax.tree_util.tree_map(lambda x: x[0], all_states_online_batch[0]), all_states_online_batch[1], all_states_online_batch[2]
    

# import pdb; pdb.set_trace()


