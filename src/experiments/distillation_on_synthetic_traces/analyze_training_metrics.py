"""
Analyze training metrics from all branches and plot comparison graphs.
Reads trainer_state.json from each checkpoint and extracts accuracy metrics.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_metrics_from_checkpoints(model_dir: Path, branch_name: str) -> pd.DataFrame:
    """
    Extract metrics from all checkpoints in a model directory.
    
    Args:
        model_dir: Path to model directory containing checkpoints
        branch_name: Name of the branch (A, B, C)
    
    Returns:
        DataFrame with columns: epoch, step, eval_accuracy, train_loss, branch
    """
    checkpoints = sorted(model_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    
    metrics_data = []
    for ckpt in checkpoints:
        state_file = ckpt / "trainer_state.json"
        if not state_file.exists():
            continue
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        epoch = state.get("epoch", 0)
        step = state.get("global_step", 0)
        
        # Extract eval metrics
        log_history = state.get("log_history", [])
        for entry in log_history:
            if "eval_accuracy" in entry:
                metrics_data.append({
                    "branch": branch_name,
                    "epoch": epoch,
                    "step": step,
                    "eval_accuracy": entry["eval_accuracy"],
                    "checkpoint": ckpt.name,
                })
            elif "loss" in entry and "eval_loss" not in entry:
                # Training loss
                if metrics_data and metrics_data[-1]["step"] == entry.get("step"):
                    metrics_data[-1]["train_loss"] = entry["loss"]
    
    return pd.DataFrame(metrics_data)


def analyze_branches(
    output_dir: Path = Path("data/out/models"),
    save_dir: Path = Path("data/out/analysis/branches"),
) -> Dict[str, pd.DataFrame]:
    """
    Analyze all branches and generate comparison reports.
    
    Args:
        output_dir: Directory containing branch_a, branch_b, branch_c models
        save_dir: Directory to save analysis results
    
    Returns:
        Dictionary with branch metrics DataFrames
    """
    output_dir = Path(output_dir).resolve()
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    branches = {}
    for branch_name in ["A", "B", "C"]:
        branch_dir = output_dir / f"branch_{branch_name.lower()}"
        if not branch_dir.exists():
            logging.warning(f"Branch {branch_name} directory not found: {branch_dir}")
            continue
        
        logging.info(f"Analyzing Branch {branch_name}...")
        df = extract_metrics_from_checkpoints(branch_dir, branch_name)
        branches[branch_name] = df
        
        # Save individual branch metrics
        df.to_csv(save_dir / f"branch_{branch_name.lower()}_metrics.csv", index=False)
        logging.info(f"Branch {branch_name}: {len(df)} metric entries")
    
    # Combine all branches
    if branches:
        combined_df = pd.concat(branches.values(), ignore_index=True)
        combined_df.to_csv(save_dir / "all_branches_metrics.csv", index=False)
        logging.info(f"Saved combined metrics: {len(combined_df)} total entries")
    
    return branches


def plot_accuracy_comparison(
    branches: Dict[str, pd.DataFrame],
    save_path: Path = Path("data/out/analysis/branches/accuracy_comparison.png"),
):
    """
    Plot accuracy comparison across branches by epoch.
    
    Args:
        branches: Dictionary with branch metrics DataFrames
        save_path: Path to save the plot
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for branch_name, df in branches.items():
        if df.empty or "eval_accuracy" not in df.columns:
            continue
        
        # Group by epoch and take the last eval_accuracy for each epoch
        epoch_metrics = df.groupby("epoch")["eval_accuracy"].last()
        plt.plot(epoch_metrics.index, epoch_metrics.values, marker='o', label=f"Branch {branch_name}")
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Comparison Across Branches", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved accuracy plot: {save_path}")
    plt.close()


def plot_training_curves(
    branches: Dict[str, pd.DataFrame],
    save_path: Path = Path("data/out/analysis/branches/training_curves.png"),
):
    """
    Plot training loss curves for all branches.
    
    Args:
        branches: Dictionary with branch metrics DataFrames
        save_path: Path to save the plot
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy
    for branch_name, df in branches.items():
        if df.empty or "eval_accuracy" not in df.columns:
            continue
        epoch_metrics = df.groupby("epoch")["eval_accuracy"].last()
        ax1.plot(epoch_metrics.index, epoch_metrics.values, marker='o', label=f"Branch {branch_name}")
    
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Eval Accuracy by Epoch", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    for branch_name, df in branches.items():
        if df.empty or "train_loss" not in df.columns:
            continue
        train_metrics = df[df["train_loss"].notna()]
        if not train_metrics.empty:
            ax2.plot(train_metrics["step"], train_metrics["train_loss"], alpha=0.7, label=f"Branch {branch_name}")
    
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Training Loss by Step", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved training curves: {save_path}")
    plt.close()


def analyze_by_complexity(
    model_dir: Path,
    branch_name: str,
    save_dir: Path,
) -> pd.DataFrame:
    """Analyze incorrect answers by complexity levels."""
    incorrect_file = model_dir / "incorrect_answers.tsv"
    if not incorrect_file.exists():
        logging.warning(f"No incorrect_answers.tsv found for branch {branch_name}")
        return pd.DataFrame()
    
    df = pd.read_csv(incorrect_file, sep="\t")
    if "complexity" not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Bin complexity into levels
    df["complexity_level"] = pd.cut(
        df["complexity"],
        bins=[0, 50, 100, 150, 200, float("inf")],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    
    # Count errors by complexity
    complexity_stats = df.groupby("complexity_level").size().reset_index(name="error_count")
    complexity_stats["branch"] = branch_name
    
    # Save complexity analysis
    complexity_stats.to_csv(save_dir / f"branch_{branch_name.lower()}_complexity.csv", index=False)
    
    return complexity_stats


def plot_complexity_analysis(
    branches_complexity: Dict[str, pd.DataFrame],
    save_path: Path = Path("data/out/analysis/branches/complexity_analysis.png"),
):
    """Plot error distribution by complexity levels."""
    if not branches_complexity:
        return
    
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Combine all branch data
    all_data = pd.concat(branches_complexity.values(), ignore_index=True)
    
    # Plot grouped bar chart
    complexity_levels = ["Very Low", "Low", "Medium", "High", "Very High"]
    x = range(len(complexity_levels))
    width = 0.25
    
    for i, (branch_name, df) in enumerate(branches_complexity.items()):
        df_full = pd.DataFrame({"complexity_level": complexity_levels})
        df_full = df_full.merge(df[["complexity_level", "error_count"]], on="complexity_level", how="left")
        df_full["error_count"] = df_full["error_count"].fillna(0)
        
        offset = width * (i - 1)
        ax.bar([xi + offset for xi in x], df_full["error_count"], width, label=f"Branch {branch_name}")
    
    ax.set_xlabel("Complexity Level", fontsize=12)
    ax.set_ylabel("Number of Errors", fontsize=12)
    ax.set_title("Error Distribution by Question Complexity", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(complexity_levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logging.info(f"Saved complexity analysis plot: {save_path}")
    plt.close()


def print_summary_table(branches: Dict[str, pd.DataFrame]):
    """Print summary statistics for all branches."""
    print("\n" + "="*80)
    print("BRANCH COMPARISON SUMMARY")
    print("="*80)
    
    summary_data = []
    for branch_name, df in branches.items():
        if df.empty:
            continue
        
        final_accuracy = df["eval_accuracy"].iloc[-1] if "eval_accuracy" in df.columns else None
        max_accuracy = df["eval_accuracy"].max() if "eval_accuracy" in df.columns else None
        num_checkpoints = df["checkpoint"].nunique() if "checkpoint" in df.columns else 0
        
        summary_data.append({
            "Branch": branch_name,
            "Final Accuracy": f"{final_accuracy:.4f}" if final_accuracy else "N/A",
            "Best Accuracy": f"{max_accuracy:.4f}" if max_accuracy else "N/A",
            "Checkpoints": num_checkpoints,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "../../../../data/out/models"
    save_dir = Path(__file__).parent / "../../../../data/out/analysis/branches"
    
    # Analyze all branches
    branches = analyze_branches(
        output_dir=output_dir,
        save_dir=save_dir,
    )
    
    if branches:
        # Generate plots
        plot_accuracy_comparison(branches)
        plot_training_curves(branches)
        
        # Analyze by complexity
        branches_complexity = {}
        for branch_name in ["A", "B", "C"]:
            branch_dir = output_dir / f"branch_{branch_name.lower()}"
            if branch_dir.exists():
                complexity_df = analyze_by_complexity(branch_dir, branch_name, save_dir)
                if not complexity_df.empty:
                    branches_complexity[branch_name] = complexity_df
        
        if branches_complexity:
            plot_complexity_analysis(branches_complexity)
            logging.info("Complexity analysis complete!")
        
        # Print summary
        print_summary_table(branches)
        
        logging.info("Analysis complete!")
    else:
        logging.warning("No branch data found to analyze.")
