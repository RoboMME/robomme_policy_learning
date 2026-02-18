#!/usr/bin/env python3
import argparse
import multiprocessing as mp
from pathlib import Path
from zipfile import ZipFile, BadZipFile

from tqdm import tqdm


def unzip_one(zip_path: Path, overwrite: bool = False) -> None:
    zip_path = zip_path.resolve()
    out_dir = zip_path.with_suffix("")
    stem = zip_path.stem

    if out_dir.exists() and not overwrite:
        print(f"[skip] {zip_path} -> {out_dir} (already exists)")
        return

    try:
        out_dir.mkdir(exist_ok=True, parents=True)
        with ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                name = member.filename

                # Skip directory entries; we'll create directories as needed
                if name.endswith("/"):
                    continue

                internal_path = Path(name)
                parts = internal_path.parts

                # Strip everything up to and including the first occurrence
                # of the zip's stem (e.g., "79999") so that contents end up
                # directly under the output folder.
                if stem in parts:
                    idx = parts.index(stem)
                    rel_parts = parts[idx + 1 :]
                    if rel_parts:
                        dest_rel = Path(*rel_parts)
                    else:
                        # If the entry is exactly the stem directory, just use its name
                        dest_rel = Path(internal_path.name)
                else:
                    # Fallback: just use the basename to avoid recreating long paths
                    dest_rel = Path(internal_path.name)

                dest_path = out_dir / dest_rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member, "r") as src, open(dest_path, "wb") as dst:
                    dst.write(src.read())
        print(f"[ok]   {zip_path} -> {out_dir}")
    except BadZipFile:
        print(f"[bad]  {zip_path} is not a valid zip file")
    except Exception as e:
        print(f"[err]  Failed on {zip_path}: {e}")


def find_zip_files(root: Path):
    return [p for p in root.rglob("*.zip") if p.is_file()]


def _worker(args):
    """Helper for multiprocessing.imap_unordered."""
    return unzip_one(*args)


def main():
    parser = argparse.ArgumentParser(
        description="Unzip all .zip files under a directory using multiprocessing."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="runs/ckpts",
        help="Root directory to search for .zip files (default: runs/ckpts)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=0,
        help="Number of processes (default: CPU count)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite (re-extract) even if output folder already exists.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}")
        return

    zips = find_zip_files(root)
    if not zips:
        print(f"No .zip files found under {root}")
        return

    print(f"Found {len(zips)} zip files under {root}")
    procs = args.processes or mp.cpu_count()

    # Prepare argument tuples for workers
    tasks = [(zp, args.overwrite) for zp in zips]

    with mp.Pool(processes=procs) as pool:
        for _ in tqdm(
            pool.imap_unordered(_worker, tasks),
            total=len(tasks),
            desc="Unzipping",
        ):
            # tqdm updates as each task completes
            pass


if __name__ == "__main__":
    main()