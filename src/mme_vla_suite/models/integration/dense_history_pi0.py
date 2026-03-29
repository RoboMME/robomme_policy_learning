"""π₀.₅ with dense past-frame history and per-position action supervision.

Standalone model based directly on ``openpi.models.pi0.Pi0`` — no dependency
on ``HistoryPi0`` or any RoboMME memory encoder.

Sequence layout (training)::

    prefix:  [img_t-N | … | img_t-1 | img_cur | wrist_cur | lang]   ← bidirectional
    suffix:  [act_t-N | … | act_t-1 | act_cur]                      ← block-causal

Each ``act_tk`` block is ``action_horizon`` tokens.  Block-causal means
``act_tk`` attends to all prefix tokens + ``act_t0 … act_tk`` but **not**
to future action blocks.

At inference only the current action chunk is decoded (parent ``sample_actions``
logic with a longer prefix).
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask, posemb_sincos
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at

from mme_vla_suite.models.integration.history_observation import (
    HistAugObservation,
    preprocess_observation,
)

logger = logging.getLogger("dense-history-pi0")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DenseHistoryPi0Config(Pi0Config):
    """Config for dense-history π₀.₅.

    Identical to ``Pi0Config`` (same backbone, same weights) plus
    ``num_past_frames`` which controls how many past observation / action
    positions appear during training.
    """

    num_past_frames: int = 10

    @override
    def create(self, rng: at.KeyArrayLike) -> "DenseHistoryPi0":
        return DenseHistoryPi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        base_obs, base_act = super().inputs_spec(batch_size=batch_size)
        return base_obs, base_act


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DenseHistoryPi0(Pi0):
    """Dense-history variant of π₀.₅.

    Re-uses every layer from ``Pi0`` (SigLIP, Gemma LLM, action projection,
    time MLP).  Only ``compute_loss`` is overridden to:

    1. Encode N past front-camera images via the **same frozen SigLIP**.
    2. Place them before the current images in the prefix.
    3. Stack N+1 action chunks in the suffix with block-causal masking.
    4. Supervise flow-matching at every position.

    ``sample_actions`` delegates to the parent with an extended prefix.
    """

    def __init__(self, config: DenseHistoryPi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.num_past_frames = config.num_past_frames

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _encode_past_images(
        self,
        past_image: at.Float[at.Array, "b n h w c"],
        past_mask: at.Bool[at.Array, "b n"],
    ) -> tuple[at.Float[at.Array, "b tokens emb"], at.Bool[at.Array, "b tokens"]]:
        """Run SigLIP on each past front-camera image and flatten to (b, N*P, emb)."""
        b, n, h, w, c = past_image.shape
        flat = einops.rearrange(past_image, "b n h w c -> (b n) h w c")
        flat_tokens, _ = self.PaliGemma.img(flat, train=False)
        P = flat_tokens.shape[1]
        tokens = einops.rearrange(flat_tokens, "(b n) p d -> b (n p) d", b=b, n=n)

        per_frame_mask = einops.repeat(past_mask, "b n -> b (n p)", p=P)
        return tokens, per_frame_mask

    # ------------------------------------------------------------------
    # embed_prefix — extended with past images
    # ------------------------------------------------------------------

    def embed_prefix_with_history(
        self,
        obs: HistAugObservation,
        past_image: at.Float[at.Array, "b n h w c"] | None,
        past_mask: at.Bool[at.Array, "b n"] | None,
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Array]:
        tokens = []
        input_mask = []
        ar_mask: list[bool] = []

        if past_image is not None and past_mask is not None:
            hist_tokens, hist_mask = self._encode_past_images(past_image, past_mask)
            tokens.append(hist_tokens)
            input_mask.append(hist_mask)
            ar_mask += [False] * hist_tokens.shape[1]

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            lang_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(lang_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * lang_tokens.shape[1]

        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.array(ar_mask),
        )

    # ------------------------------------------------------------------
    # embed_suffix — (N+1) action chunks, block-causal
    # ------------------------------------------------------------------

    def embed_suffix_dense(
        self,
        noisy_actions_stack: at.Float[at.Array, "b k ah ad"],
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Array, at.Float[at.Array, "b emb"] | None]:
        """Embed K = N+1 action chunks into one suffix with block-causal ar_mask."""
        b, K, ah, ad = noisy_actions_stack.shape
        flat_actions = einops.rearrange(noisy_actions_stack, "b k ah ad -> b (k ah) ad")
        action_tokens = self.action_in_proj(flat_actions)

        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=K * ah)
            action_time = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_tokens = self.action_time_mlp_in(action_time)
            action_tokens = nnx.swish(action_tokens)
            action_tokens = self.action_time_mlp_out(action_tokens)
            adarms_cond = None

        mask = jnp.ones((b, K * ah), dtype=jnp.bool_)

        ar_mask_list: list[bool] = []
        for _ in range(K):
            ar_mask_list += [True] + [False] * (ah - 1)

        return action_tokens, mask, jnp.array(ar_mask_list), adarms_cond

    # ------------------------------------------------------------------
    # compute_loss — multi-position flow matching
    # ------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        past_image = observation.past_image
        past_actions = observation.past_actions
        past_mask = observation.past_frame_mask

        has_history = past_actions is not None and past_mask is not None

        if has_history:
            all_actions = jnp.concatenate(
                [past_actions, actions[:, None, :, :]],
                axis=1,
            )
            K = all_actions.shape[1]
        else:
            all_actions = actions[:, None, :, :]
            K = 1

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, all_actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * all_actions
        u_t = noise - all_actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_history(
            observation, past_image, past_mask
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_dense(
            x_t, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        v_t = self.action_out_proj(suffix_out)
        v_t = einops.rearrange(v_t, "b (k ah) ad -> b k ah ad", k=K, ah=self.action_horizon)
        per_pos_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        if has_history:
            loss_mask = jnp.concatenate(
                [past_mask, jnp.ones((*batch_shape, 1), dtype=jnp.bool_)],
                axis=-1,
            )
            masked_loss = per_pos_loss * loss_mask[..., None]
            num_valid = jnp.maximum(loss_mask.sum(axis=-1, keepdims=True), 1.0)
            per_step_loss = masked_loss.sum(axis=1) / num_valid
        else:
            per_step_loss = per_pos_loss[:, 0]

        stats = {}
        return per_step_loss, stats

    # ------------------------------------------------------------------
    # sample_actions — inference uses parent with extended prefix
    # ------------------------------------------------------------------

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: HistAugObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        past_image = observation.past_image
        past_mask = observation.past_frame_mask

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_history(
            observation, past_image, past_mask
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cross_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([cross_mask, suffix_attn_mask], axis=-1)
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
