"""Verify that enlarging MemoryBuffer.max_steps preserves pos_emb values.

PosEmb3D is pure sincos (no learnable params, no sequence-length-dependent
normalization). The pre-computed pos_emb_dict in MemoryBuffer is just
`np.array(pos_embedder(jnp.arange(max_steps), spatial))`, so enlarging max_steps
should leave positions [0, old_max_steps) byte-for-byte unchanged.

Run:  uv run python -m mme_vla_suite.shared.test_pos_emb_consistency
"""

import jax.numpy as jnp
import numpy as np

from mme_vla_suite.shared.posemb_3d import PosEmb3D


def _build_pos_emb_dict(max_steps: int, pos_emb_dim: int = 768) -> dict:
    """Replicate the pos_emb_dict allocation in MemoryBuffer.__init__."""
    pos_embedder = PosEmb3D(dim=pos_emb_dim)
    ranges = jnp.arange(max_steps)
    return {
        "8x8": np.array(pos_embedder(ranges, 8)),
        "4x4": np.array(pos_embedder(ranges, 4)),
        "2x2": np.array(pos_embedder(ranges, 2)),
    }


def _check_pair(small_max_steps: int, big_max_steps: int) -> None:
    assert big_max_steps > small_max_steps
    dict_small = _build_pos_emb_dict(max_steps=small_max_steps)
    dict_big = _build_pos_emb_dict(max_steps=big_max_steps)

    for key in ("8x8", "4x4", "2x2"):
        small = dict_small[key]
        big = dict_big[key]
        assert small.shape[0] == small_max_steps and big.shape[0] == big_max_steps
        np.testing.assert_array_equal(
            big[: small.shape[0]],
            small,
            err_msg=(
                f"{key}: pos_emb at positions [0, {small.shape[0]}) "
                f"differs between max_steps={small_max_steps} and max_steps={big_max_steps}"
            ),
        )
        print(
            f"  [OK] {key}: small={small.shape}, big={big.shape}, "
            f"first {small.shape[0]} positions byte-identical"
        )


def main() -> None:
    cases = [
        (4096, 8192),
        (8192, 20480),
        (4096, 20480),
    ]
    for small, big in cases:
        print(f"\n--- pair (max_steps={small}) vs (max_steps={big}) ---")
        _check_pair(small, big)

    print("\nPASS: pos_emb_dict positions in [0, small_max_steps) are byte-for-byte identical across all tested max_steps pairs")


if __name__ == "__main__":
    main()
