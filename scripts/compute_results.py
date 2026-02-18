"""
Automatically gather the results from all ckpt_list and seed_list, and compute the final results (mean + std)

Example usage:
For perceptual or recurrent memory models:
uv run scripts/compute_results.py --model_dir perceptual-tokendrop-modul --ckpt_list ckpt60000,ckpt70000,ckpt79999 --seed_list seed0,seed42,seed7

For symbolic memory models:
uv run scripts/compute_results.py --model_dir symbolic-grounded-subgoal --ckpt_list ckpt60000,ckpt70000,ckpt79999 --seed_list seed0,seed42,seed7 --symbolic_type oracle
"""

import pandas as pd
from pathlib import Path
import json
from typing import List

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="perceptual-tokendrop-modul")
parser.add_argument("--ckpt_list", type=str, default="ckpt60000,ckpt70000,ckpt79999")
parser.add_argument("--seed_list", type=str, default="seed0,seed42,seed7")
parser.add_argument("--symbolic_type", type=str, default="") # oracle / gemini / qwenvl / memer or empty
args = parser.parse_args()
DIR = Path("runs/evaluation")

MODEL_DIR = args.model_dir
SYMBOLIC_TYPE = args.symbolic_type

if "symbolic" in MODEL_DIR:
    assert SYMBOLIC_TYPE in ["oracle", "gemini", "qwenvl", "memer"], "Invalid symbolic type"
else:
    assert SYMBOLIC_TYPE == "", "Symbolic type is not supported for this model type"

CKPT_LIST = args.ckpt_list.split(",")
SEED_LIST = args.seed_list.split(",")

TASK_NAME_LIST = [      
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    
    "VideoUnmask",
    "ButtonUnmask",
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

# Task suites
TASK_SUITES = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Persistent": ["ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"],
    "Referential": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Behavior": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}


def load_results(model_dir: str, ckpt_list: List[str], seed_list: List[str]) -> pd.DataFrame:
    """Load all results from specified checkpoints into a DataFrame.
    
    Args:
        ckpt_name: Name of the model directory
        checkpoint_filter: List of checkpoint names to include (e.g., ["ckpt60000", "ckpt70000"])
    """
    data = []
    ckpt_base_dir = DIR / model_dir
    
    if not ckpt_base_dir.exists():
        print(f"Warning: Base directory not found: {ckpt_base_dir}")
        return pd.DataFrame()
    
    for ckpt in ckpt_list:
        ckpt_dir = ckpt_base_dir / ckpt
        
        if not ckpt_dir.exists():
            print(f"Warning: Checkpoint directory not found: {ckpt_dir}")
            continue
        
        for seed in seed_list:
            seed_name = ckpt_dir / seed
            if not seed_name.is_dir():
                continue
                
            try:
                # Determine log path based on symbolic_type
                if SYMBOLIC_TYPE:
                    log_path = seed_name / SYMBOLIC_TYPE / "log.json"
                else:
                    log_path = seed_name / "log.json"
                
                if not log_path.exists():
                    print(f"Warning: Log file not found: {log_path}")
                    continue
                
                results = json.load(open(log_path))
                success_rate = results["success_rate"]
                
                # Create row with metadata
                row = {
                    "checkpoint": ckpt,
                    "seed": seed,
                }
                
                # Add task results (convert to percentage)
                for task_name in TASK_NAME_LIST:
                    row[task_name] = success_rate.get(task_name, 0) * 100
                
                data.append(row)
                
            except Exception as e:
                print(f"Error loading {seed_name}: {e}")
                continue
    
    return pd.DataFrame(data)


def calculate_suite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add suite-level scores to the DataFrame."""
    df_copy = df.copy()
    
    for suite_name, tasks in TASK_SUITES.items():
        # Calculate mean across tasks in the suite for each row
        df_copy[suite_name] = df_copy[tasks].mean(axis=1)
    
    # Calculate overall mean across all tasks
    df_copy["Overall"] = df_copy[TASK_NAME_LIST].mean(axis=1)
    
    return df_copy


def format_mean_std(mean: float, std: float, separator: str = " ± ") -> str:
    """Format mean and std string with custom separator.
    
    Args:
        mean: Mean value
        std: Standard deviation
        separator: String to separate mean and std (e.g., " ± ", " & ", " | ", "\t")
    """
    return f"{mean:.2f}{separator}{std:.2f}"


def print_results(df: pd.DataFrame):
    """Print results in a formatted way."""
    # Get all numeric columns (exclude 'checkpoint' and 'seed')
    numeric_cols = [col for col in df.columns if col not in ['checkpoint', 'seed']]
    
    # Group by checkpoint and calculate statistics only for numeric columns
    stats = df.groupby("checkpoint")[numeric_cols].agg(['mean', 'std'])
    
    # Calculate overall statistics across all checkpoints
    overall_stats = df[numeric_cols].agg(['mean', 'std'])
    
    # Individual tasks
    print("=" * 100)
    print("INDIVIDUAL TASKS")
    print("=" * 100)
    
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"\n{MODEL_DIR}-{checkpoint}")
        print(f"Number of seeds: {len(df[df['checkpoint'] == checkpoint])}")
        print("-" * 100)
        
        for task in TASK_NAME_LIST:
            mean = stats.loc[checkpoint, (task, 'mean')]
            std = stats.loc[checkpoint, (task, 'std')]
            print(f"{task:20s}: {format_mean_std(mean, std)}")
    
    # Print overall across all checkpoints
    print(f"\n{MODEL_DIR}-ALL")
    print(f"Number of seeds: {len(df)} (across all checkpoints)")
    print("-" * 100)
    for task in TASK_NAME_LIST:
        mean = overall_stats.loc['mean', task]
        std = overall_stats.loc['std', task]
        print(f"{task:20s}: {format_mean_std(mean, std)}")
    
    # Task suites
    print("\n" + "=" * 100)
    print("TASK SUITES")
    print("=" * 100)
    
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"\n{MODEL_DIR}-{checkpoint}")
        print("-" * 100)
        
        for suite_name in TASK_SUITES.keys():
            mean = stats.loc[checkpoint, (suite_name, 'mean')]
            std = stats.loc[checkpoint, (suite_name, 'std')]
            print(f"{suite_name:20s}: {format_mean_std(mean, std)}")
        
        # Overall
        mean = stats.loc[checkpoint, ('Overall', 'mean')]
        std = stats.loc[checkpoint, ('Overall', 'std')]
        print(f"{'Overall':20s}: {format_mean_std(mean, std)}")
    
    # Print overall across all checkpoints
    print(f"\n{MODEL_DIR}-ALL")
    print("-" * 100)
    for suite_name in TASK_SUITES.keys():
        mean = overall_stats.loc['mean', suite_name]
        std = overall_stats.loc['std', suite_name]
        print(f"{suite_name:20s}: {format_mean_std(mean, std)}")
    mean = overall_stats.loc['mean', 'Overall']
    std = overall_stats.loc['std', 'Overall']
    print(f"{'Overall':20s}: {format_mean_std(mean, std)}")


def print_compact_table(df: pd.DataFrame):
    """Print a compact table with all checkpoints."""
    # Get all numeric columns
    numeric_cols = [col for col in df.columns if col not in ['checkpoint', 'seed']]
    stats = df.groupby("checkpoint")[numeric_cols].agg(['mean', 'std'])
    
    # Calculate overall statistics across all checkpoints
    overall_stats = df[numeric_cols].agg(['mean', 'std'])
        
    print("\n" + "=" * 150)
    print("COMPACT SUMMARY TABLE")
    print("=" * 150)
    
    # Header
    print(f"{'Checkpoint':<15}", end=" & ")
    for task in TASK_NAME_LIST[:8]:
        print(f"{task:<10}", end=" & ")
    print()
    print("-" * (15 + (20 + 3) * len(TASK_NAME_LIST[:8])))
    
    # Rows
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"{checkpoint:<10}", end=" & ")
        for task in TASK_NAME_LIST[:8]:
            mean = stats.loc[checkpoint, (task, 'mean')]
            std = stats.loc[checkpoint, (task, 'std')]
            print(f"{format_mean_std(mean, std):<10}", end=" & ")
        print()
    
    # Overall row
    print(f"{'ALL':<10}", end=" & ")
    for task in TASK_NAME_LIST[:8]:
        mean = overall_stats.loc['mean', task]
        std = overall_stats.loc['std', task]
        print(f"{format_mean_std(mean, std):<10}", end=" & ")
    print()
    
    
    print("\n" + "=" * 150)
    print("\n" + "=" * 150)
    
    print(f"{'Checkpoint':<15}", end=" & ")
    for task in TASK_NAME_LIST[8:]:
        print(f"{task:<10}", end=" & ")
    print()
    print("-" * (15 + (20 + 3) * len(TASK_NAME_LIST[8:])))
    
    # Rows
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"{checkpoint:<10}", end=" & ")
        for task in TASK_NAME_LIST[8:]:
            mean = stats.loc[checkpoint, (task, 'mean')]
            std = stats.loc[checkpoint, (task, 'std')]
            print(f"{format_mean_std(mean, std):<10}", end=" & ")
        print()
    
    # Overall row
    print(f"{'ALL':<10}", end=" & ")
    for task in TASK_NAME_LIST[8:]:
        mean = overall_stats.loc['mean', task]
        std = overall_stats.loc['std', task]
        print(f"{format_mean_std(mean, std):<10}", end=" & ")
    print()
    
    # Suite summary
    print("\n" + "=" * 150)
    print("SUITE SUMMARY")
    print("=" * 150)
    
    # Header
    print(f"{'Checkpoint':<15}", end=" & ")
    for suite in list(TASK_SUITES.keys()) + ["Overall"]:
        print(f"{suite:<20}", end=" & ")
    print()
    print("-" * (15 + (20 + 3) * (len(TASK_SUITES) + 1)))
    
    # Rows
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"{checkpoint:<15}", end=" & ")
        for suite in list(TASK_SUITES.keys()) + ["Overall"]:
            mean = stats.loc[checkpoint, (suite, 'mean')]
            std = stats.loc[checkpoint, (suite, 'std')]
            print(f"{format_mean_std(mean, std):<20}", end=" & ")
        print()
    
    # Overall row
    print(f"{'ALL':<15}", end=" & ")
    for suite in list(TASK_SUITES.keys()) + ["Overall"]:
        mean = overall_stats.loc['mean', suite]
        std = overall_stats.loc['std', suite]
        print(f"{format_mean_std(mean, std):<20}", end=" & ")
    print()


def print_mean_only_table(df: pd.DataFrame):
    """Print a compact table showing only means (no std)."""
    numeric_cols = [col for col in df.columns if col not in ['checkpoint', 'seed']]
    stats = df.groupby("checkpoint")[numeric_cols].agg(['mean', 'std'])
    
    # Calculate overall statistics across all checkpoints
    overall_stats = df[numeric_cols].agg(['mean', 'std'])
    
    print("\n" + "=" * 150)
    print("MEAN ONLY TABLE (for quick comparison)")
    print("=" * 150)
    
    # Header
    print(f"{'Checkpoint':<15}", end=" & ")
    for task in TASK_NAME_LIST:
        print(f"{task:<8}", end=" & ")
    for suite in list(TASK_SUITES.keys()) + ["Overall"]:
        print(f"{suite:<12}", end=" & ")
    print()
    print("-" * 200)
    
    # Rows
    for checkpoint in CKPT_LIST:  # Use the order from ckpt_dirs
        if checkpoint not in df["checkpoint"].values:
            continue
        print(f"{checkpoint:<15}", end=" & ")
        for task in TASK_NAME_LIST:
            mean = stats.loc[checkpoint, (task, 'mean')]
            print(f"{mean:>8.2f}", end=" & ")
        for suite in list(TASK_SUITES.keys()) + ["Overall"]:
            mean = stats.loc[checkpoint, (suite, 'mean')]
            print(f"{mean:>12.2f}", end=" & ")
        print()
    
    # Overall row
    print(f"{'ALL':<15}", end=" & ")
    for task in TASK_NAME_LIST:
        mean = overall_stats.loc['mean', task]
        print(f"{mean:>8.2f}", end=" & ")
    for suite in list(TASK_SUITES.keys()) + ["Overall"]:
        mean = overall_stats.loc['mean', suite]
        print(f"{mean:>12.2f}", end=" & ")
    print()



def compute_final_results():
    """Evaluate one model across specified checkpoints and seeds."""
    ckpt_dir = DIR / MODEL_DIR
    
    if not ckpt_dir.exists():
        print(f"Directory not found: {ckpt_dir}")
        return
    
    # Print configuration
    print(f"Model: {MODEL_DIR}")
    print(f"Checkpoints: {CKPT_LIST}")
    print(f"Symbolic type: '{SYMBOLIC_TYPE}' (empty means default)")
    print()
    
    # Load all results with checkpoint filtering
    df = load_results(MODEL_DIR, CKPT_LIST, SEED_LIST)
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(df)} results from {len(df['checkpoint'].unique())} checkpoints")
    print()
    
    # Calculate suite scores
    df = calculate_suite_scores(df)
    
    # Print detailed results
    print_results(df)
    
    # Print compact table
    print_compact_table(df)
    
    # Print mean-only table for quick comparison
    print_mean_only_table(df)
    

if __name__ == "__main__":
    compute_final_results()