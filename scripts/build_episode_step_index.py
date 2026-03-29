#!/usr/bin/env python3
"""Build and save (epis_idx, step_idx) -> global sample index for data/*.pkl.

Writes ``meta/episode_step_to_sample_idx.json`` with string keys ``\"{epis_idx}_{step_idx}\"``.

Usage:
  uv run scripts/build_episode_step_index.py data/robomme_preprocessed_data
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Preprocessed dataset root (contains data/ and meta/).",
    )
    args = parser.parse_args()

    data_dir = args.dataset_path / "data"
    meta_dir = args.dataset_path / "meta"
    if not data_dir.is_dir():
        raise SystemExit(f"Missing data dir: {data_dir}")

    meta_dir.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, int] = {}
    pkl_files = sorted(data_dir.glob("*.pkl"), key=lambda p: int(p.stem))

    for p in tqdm(pkl_files, desc="episode_step index"):
        sample_idx = int(p.stem)
        with open(p, "rb") as f:
            d = pickle.load(f)
        epis = int(d["epis_idx"])
        step = int(d["step_idx"])
        key = f"{epis}_{step}"
        if key in mapping:
            raise ValueError(
                f"Duplicate (epis_idx, step_idx)=({epis}, {step}): "
                f"existing sample_idx={mapping[key]}, new={sample_idx}"
            )
        mapping[key] = sample_idx

    out_path = meta_dir / "episode_step_to_sample_idx.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"Wrote {len(mapping)} entries to {out_path}")


if __name__ == "__main__":
    main()
