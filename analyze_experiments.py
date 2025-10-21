"""
analyze_experiments.py

Usage:
    python analyze_experiments.py <path_to_experiment_results.csv>

This script:
  - Loads experiment_results.csv from the DiT experiments sweep
  - Cleans and aggregates results
  - Generates report-ready comparative visualizations:
      * avg loss vs. patch size, depth, heads, timesteps
      * FID vs. configuration parameters (if available)
      * parameter count vs. avg loss (efficiency)
      * heatmaps for loss trends across hyperparameters
  - Saves plots to an 'analysis' subfolder next to the CSV
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-darkgrid")

def main(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiments from {csv_path}")

    # Clean numeric columns
    for col in ["avg_epoch_loss", "fid", "params"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Output folder for analysis
    analysis_dir = csv_path.parent / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    print(f"Saving analysis plots to: {analysis_dir}")

    # --- Basic summary ---
    print("\nSummary of completed experiments:")
    print(df[["varied_param", "patch_size", "depth", "num_heads", "timesteps", "avg_epoch_loss"]].describe())

    # --- Helper plotting function ---
    def save_plot(fig, name):
        fig.tight_layout()
        fig.savefig(analysis_dir / f"{name}.png", dpi=300)
        plt.close(fig)

    # ================== LOSS PLOTS ==================
    # Create a 2x2 grid of plots for the four varied parameters
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 1. Average Loss vs Patch Size ---
    patch_data = df[df["varied_param"] == "patch_size"].sort_values("patch_size")
    if len(patch_data) > 0:
        axes[0, 0].plot(patch_data["patch_size"], patch_data["avg_epoch_loss"], 
                       marker='o', linewidth=2, markersize=8, color='#2E86AB')
        axes[0, 0].set_xlabel("Patch Size", fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel("Avg Epoch Loss", fontsize=11, fontweight='bold')
        axes[0, 0].set_title("Effect of Patch Size", fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xscale('log', base=2)
        
        # Add value labels
        for x, y in zip(patch_data["patch_size"], patch_data["avg_epoch_loss"]):
            axes[0, 0].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=9)

    # --- 2. Average Loss vs Depth ---
    depth_data = df[df["varied_param"] == "depth"].sort_values("depth")
    if len(depth_data) > 0:
        axes[0, 1].plot(depth_data["depth"], depth_data["avg_epoch_loss"], 
                       marker='s', linewidth=2, markersize=8, color='#A23B72')
        axes[0, 1].set_xlabel("Model Depth (# DiT Blocks)", fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel("Avg Epoch Loss", fontsize=11, fontweight='bold')
        axes[0, 1].set_title("Effect of Model Depth", fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(depth_data["depth"], depth_data["avg_epoch_loss"]):
            axes[0, 1].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=9)

    # --- 3. Average Loss vs Num Heads ---
    heads_data = df[df["varied_param"] == "num_heads"].sort_values("num_heads")
    if len(heads_data) > 0:
        axes[1, 0].plot(heads_data["num_heads"], heads_data["avg_epoch_loss"], 
                       marker='^', linewidth=2, markersize=8, color='#F18F01')
        axes[1, 0].set_xlabel("Number of Attention Heads", fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel("Avg Epoch Loss", fontsize=11, fontweight='bold')
        axes[1, 0].set_title("Effect of Attention Heads", fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(heads_data["num_heads"], heads_data["avg_epoch_loss"]):
            axes[1, 0].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=9)

    # --- 4. Average Loss vs Timesteps ---
    timesteps_data = df[df["varied_param"] == "timesteps"].sort_values("timesteps")
    if len(timesteps_data) > 0:
        axes[1, 1].plot(timesteps_data["timesteps"], timesteps_data["avg_epoch_loss"], 
                       marker='D', linewidth=2, markersize=8, color='#C73E1D')
        axes[1, 1].set_xlabel("Diffusion Timesteps", fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel("Avg Epoch Loss", fontsize=11, fontweight='bold')
        axes[1, 1].set_title("Effect of Diffusion Timesteps", fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(timesteps_data["timesteps"], timesteps_data["avg_epoch_loss"]):
            axes[1, 1].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=9)

    fig.suptitle("DiT Hyperparameter Sensitivity Analysis - Training Loss\n(One Parameter Varied at a Time)", 
                 fontsize=14, fontweight='bold', y=0.995)
    save_plot(fig, "hyperparameter_trends_loss_combined")
    
    # ================== FID PLOTS ==================
    # Check if we have FID data
    has_fid = df["fid"].notna().sum() > 0
    
    if has_fid:
        print(f"\nFound FID scores for {df['fid'].notna().sum()} experiments. Generating FID plots...")
        
        # Create a 2x2 grid of FID plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # --- 1. FID vs Patch Size ---
        if len(patch_data) > 0 and patch_data["fid"].notna().sum() > 0:
            valid_patch = patch_data[patch_data["fid"].notna()]
            axes[0, 0].plot(valid_patch["patch_size"], valid_patch["fid"], 
                           marker='o', linewidth=2, markersize=8, color='#2E86AB')
            axes[0, 0].set_xlabel("Patch Size", fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel("FID Score (lower is better)", fontsize=11, fontweight='bold')
            axes[0, 0].set_title("Effect of Patch Size", fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log', base=2)
            
            # Add value labels
            for x, y in zip(valid_patch["patch_size"], valid_patch["fid"]):
                axes[0, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=9)
        
        # --- 2. FID vs Depth ---
        if len(depth_data) > 0 and depth_data["fid"].notna().sum() > 0:
            valid_depth = depth_data[depth_data["fid"].notna()]
            axes[0, 1].plot(valid_depth["depth"], valid_depth["fid"], 
                           marker='s', linewidth=2, markersize=8, color='#A23B72')
            axes[0, 1].set_xlabel("Model Depth (# DiT Blocks)", fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel("FID Score (lower is better)", fontsize=11, fontweight='bold')
            axes[0, 1].set_title("Effect of Model Depth", fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(valid_depth["depth"], valid_depth["fid"]):
                axes[0, 1].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=9)
        
        # --- 3. FID vs Num Heads ---
        if len(heads_data) > 0 and heads_data["fid"].notna().sum() > 0:
            valid_heads = heads_data[heads_data["fid"].notna()]
            axes[1, 0].plot(valid_heads["num_heads"], valid_heads["fid"], 
                           marker='^', linewidth=2, markersize=8, color='#F18F01')
            axes[1, 0].set_xlabel("Number of Attention Heads", fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel("FID Score (lower is better)", fontsize=11, fontweight='bold')
            axes[1, 0].set_title("Effect of Attention Heads", fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(valid_heads["num_heads"], valid_heads["fid"]):
                axes[1, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=9)
        
        # --- 4. FID vs Timesteps ---
        if len(timesteps_data) > 0 and timesteps_data["fid"].notna().sum() > 0:
            valid_timesteps = timesteps_data[timesteps_data["fid"].notna()]
            axes[1, 1].plot(valid_timesteps["timesteps"], valid_timesteps["fid"], 
                           marker='D', linewidth=2, markersize=8, color='#C73E1D')
            axes[1, 1].set_xlabel("Diffusion Timesteps", fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel("FID Score (lower is better)", fontsize=11, fontweight='bold')
            axes[1, 1].set_title("Effect of Diffusion Timesteps", fontsize=12, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for x, y in zip(valid_timesteps["timesteps"], valid_timesteps["fid"]):
                axes[1, 1].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=9)
        
        fig.suptitle("DiT Hyperparameter Sensitivity Analysis - FID Score\n(One Parameter Varied at a Time)", 
                     fontsize=14, fontweight='bold', y=0.995)
        save_plot(fig, "hyperparameter_trends_fid_combined")
        
        # Save individual FID plots for better detail
        for param_name, data, color, marker in [
            ("patch_size", patch_data, '#2E86AB', 'o'),
            ("depth", depth_data, '#A23B72', 's'),
            ("num_heads", heads_data, '#F18F01', '^'),
            ("timesteps", timesteps_data, '#C73E1D', 'D')
        ]:
            if len(data) > 0 and data["fid"].notna().sum() > 0:
                valid_data = data[data["fid"].notna()]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(valid_data[param_name], valid_data["fid"], 
                       marker=marker, linewidth=2.5, markersize=10, color=color)
                
                xlabel_map = {
                    "patch_size": "Patch Size",
                    "depth": "Model Depth (# DiT Blocks)",
                    "num_heads": "Number of Attention Heads",
                    "timesteps": "Diffusion Timesteps"
                }
                title_map = {
                    "patch_size": "Effect of Patch Size on Image Quality (FID)",
                    "depth": "Effect of Model Depth on Image Quality (FID)",
                    "num_heads": "Effect of Attention Heads on Image Quality (FID)",
                    "timesteps": "Effect of Diffusion Timesteps on Image Quality (FID)"
                }
                
                ax.set_xlabel(xlabel_map[param_name], fontsize=12, fontweight='bold')
                ax.set_ylabel("FID Score (lower is better)", fontsize=12, fontweight='bold')
                ax.set_title(title_map[param_name], fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                if param_name == "patch_size":
                    ax.set_xscale('log', base=2)
                
                # Add value labels
                for x, y in zip(valid_data[param_name], valid_data["fid"]):
                    ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                              xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
                
                save_plot(fig, f"fid_vs_{param_name}")
    else:
        print("\n⚠️  No FID scores found in data. Skipping FID plots.")
    
    # Also save individual loss plots for better detail
    for param_name, data, color, marker in [
        ("patch_size", patch_data, '#2E86AB', 'o'),
        ("depth", depth_data, '#A23B72', 's'),
        ("num_heads", heads_data, '#F18F01', '^'),
        ("timesteps", timesteps_data, '#C73E1D', 'D')
    ]:
        if len(data) > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(data[param_name], data["avg_epoch_loss"], 
                   marker=marker, linewidth=2.5, markersize=10, color=color)
            
            xlabel_map = {
                "patch_size": "Patch Size",
                "depth": "Model Depth (# DiT Blocks)",
                "num_heads": "Number of Attention Heads",
                "timesteps": "Diffusion Timesteps"
            }
            title_map = {
                "patch_size": "Effect of Patch Size on Training Loss",
                "depth": "Effect of Model Depth on Training Loss",
                "num_heads": "Effect of Attention Heads on Training Loss",
                "timesteps": "Effect of Diffusion Timesteps on Training Loss"
            }
            
            ax.set_xlabel(xlabel_map[param_name], fontsize=12, fontweight='bold')
            ax.set_ylabel("Average Epoch Loss", fontsize=12, fontweight='bold')
            ax.set_title(title_map[param_name], fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if param_name == "patch_size":
                ax.set_xscale('log', base=2)
            
            # Add value labels
            for x, y in zip(data[param_name], data["avg_epoch_loss"]):
                ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                          xytext=(0,12), ha='center', fontsize=10, fontweight='bold')
            
            save_plot(fig, f"loss_vs_{param_name}")

    # --- Parameter count comparison ---
    if df["params"].notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each experiment group
        for param_name, color in [
            ("patch_size", '#2E86AB'),
            ("depth", '#A23B72'),
            ("num_heads", '#F18F01'),
            ("timesteps", '#C73E1D')
        ]:
            param_df = df[df["varied_param"] == param_name]
            if len(param_df) > 0:
                ax.scatter(param_df["params"], param_df["avg_epoch_loss"], 
                          label=f"Varying {param_name}", s=100, alpha=0.7, color=color)
        
        ax.set_xlabel("Model Parameters", fontsize=12, fontweight='bold')
        ax.set_ylabel("Average Epoch Loss", fontsize=12, fontweight='bold')
        ax.set_title("Model Efficiency: Parameters vs Loss", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        save_plot(fig, "params_vs_loss")

    # --- Summary table ---
    summary_data = []
    for param in ["patch_size", "depth", "num_heads", "timesteps"]:
        param_df = df[df["varied_param"] == param]
        if len(param_df) > 0:
            best_idx = param_df["avg_epoch_loss"].idxmin()
            best_row = param_df.loc[best_idx]
            summary_data.append({
                "Varied Parameter": param,
                "Best Value": best_row[param],
                "Best Loss": f"{best_row['avg_epoch_loss']:.4f}",
                "Num Params": f"{int(best_row['params']):,}" if pd.notna(best_row['params']) else "N/A"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*60)
    print("Best Configuration for Each Varied Parameter:")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv(analysis_dir / "best_configs_summary.csv", index=False)

    print("\n✅ All plots saved successfully in", analysis_dir)
    print("\nGenerated plots:")
    print("\nLoss Analysis:")
    print(" - hyperparameter_trends_loss_combined.png: 2x2 grid showing all loss trends")
    print(" - loss_vs_patch_size.png: Detailed patch size analysis")
    print(" - loss_vs_depth.png: Detailed depth analysis")
    print(" - loss_vs_num_heads.png: Detailed attention heads analysis")
    print(" - loss_vs_timesteps.png: Detailed timesteps analysis")
    if has_fid:
        print("\nFID (Image Quality) Analysis:")
        print(" - hyperparameter_trends_fid_combined.png: 2x2 grid showing all FID trends")
        print(" - fid_vs_patch_size.png: Detailed patch size FID analysis")
        print(" - fid_vs_depth.png: Detailed depth FID analysis")
        print(" - fid_vs_num_heads.png: Detailed attention heads FID analysis")
        print(" - fid_vs_timesteps.png: Detailed timesteps FID analysis")
    print("\nOther:")
    print(" - params_vs_loss.png: Model efficiency comparison")
    print(" - best_configs_summary.csv: Summary table of best values")
    
    print("\n📊 Key Observations:")
    print(" • Patch Size: Smaller patches capture finer details but increase computation")
    print(" • Depth: Deeper models learn better but may overfit with limited data")
    print(" • Attention Heads: Multi-head attention improves contextual understanding")
    print(" • Timesteps: More steps allow gradual denoising but increase inference time")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_experiments.py <path_to_experiment_results.csv>")
        sys.exit(1)
    main(sys.argv[1])
