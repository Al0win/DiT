"""
Experiment driver for DiT scratch-model configurations.

Place this file in the same directory as script1.py (the uploaded code).
It imports classes/functions from script1.py:
  - DiT, NoiseScheduler, train_n_epochs, visualize_loss, visualize_epoch_loss
  - LandscapeDataset, transform (or dataset_path)
  - compute_fid_score (optional; expensive)

See script1.py for those definitions. :contentReference[oaicite:1]{index=1}
"""

import os
import csv
import time
import itertools
from pathlib import Path
from datetime import datetime

import torch
import random
import numpy as np

from tqdm import tqdm

# Import definitions from dit_models module (clean module without execution code)
from dit_models import (
    DiT,
    NoiseScheduler,
    train_n_epochs,
    visualize_loss,
    visualize_epoch_loss,
    LandscapeDataset,
    transform,
    compute_fid_score,   # optional
)

from torch.utils.data import DataLoader

# -------------------------
# Experiment configuration
# -------------------------
# Default baseline configuration
DEFAULT_CONFIG = {
    "patch_size": 4,
    "depth": 8,
    "num_heads": 6,
    "timesteps": 1000,
    "hidden_size": 384,
}

# Hyperparameter variations (one at a time from baseline)
EXPERIMENT_CONFIG = {
    # Vary patch size (keep others at default)
    "patch_size_experiments": [4, 8, 16, 32],
    
    # Vary depth (keep others at default)
    "depth_experiments": [4, 6, 8, 10],
    
    # Vary attention heads (keep others at default)
    "num_heads_experiments": [1, 2, 4, 8],
    
    # Vary timesteps (keep others at default)
    "timesteps_experiments": [100, 250, 500, 1000],
    
    # Fixed training parameters
    "hidden_size": 384,
    "batch_size": 16,
    "epochs_per_run": 5,
    "lr": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "compute_fid": True,  # Disabled for faster experiments
    "fid_num_samples": 32,
    "fid_batch_size": 32
}

# -------------------------
# Utilities
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_output_dir(base="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out

# -------------------------
# Build dataloader
# -------------------------
def build_dataloader(dataset_root, batch_size, shuffle=True):
    ds = LandscapeDataset(root_dir=dataset_root, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    return dl

# -------------------------
# Experiment runner
# -------------------------
def run_single_experiment(cfg, dataset_root, outdir, exp_id, config_dict):
    """
    - cfg: experiment config (global)
    - dataset_root: path to images (same as dataset_path in script1.py)
    - outdir: Path object where to save artifacts for this experiment
    - exp_id: string id for this run
    - config_dict: dict with keys: patch_size, depth, num_heads, timesteps
    """
    device = cfg["device"]
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # Create model with chosen config
    model = DiT(
        img_size=128,
        patch_size=config_dict["patch_size"],
        in_channels=3,
        hidden_size=cfg["hidden_size"],
        depth=config_dict["depth"],
        num_heads=config_dict["num_heads"],
    ).to(device)

    # Scheduler
    noise_scheduler = NoiseScheduler(num_timesteps=config_dict["timesteps"], schedule_type='linear').to(device)
    # Dataloader
    dataloader = build_dataloader(dataset_root, batch_size=cfg["batch_size"], shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # Train for specified epochs
    epochs = cfg["epochs_per_run"]
    print(f"[{exp_id}] Starting training: patch={config_dict['patch_size']} depth={config_dict['depth']} heads={config_dict['num_heads']} timesteps={config_dict['timesteps']} epochs={epochs}")

    epoch_losses, all_losses = train_n_epochs(model, dataloader, optimizer, noise_scheduler, device=device, num_epochs=epochs)

    # Save losses, plots, checkpoint
    outdir.mkdir(parents=True, exist_ok=True)
    # Save checkpoint
    ckpt_path = outdir / f"{exp_id}_checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict,
        "epoch_losses": epoch_losses,
        "all_losses": all_losses
    }, ckpt_path)

    # Save loss plots
    all_loss_plot_path = outdir / f"{exp_id}_all_batches_loss.png"
    epoch_loss_plot_path = outdir / f"{exp_id}_epoch_loss.png"
    try:
        visualize_loss(all_losses, save_path=str(all_loss_plot_path))
        visualize_epoch_loss(epoch_losses, save_path=str(epoch_loss_plot_path))
    except Exception as e:
        print(f"[{exp_id}] Warning: could not create plots: {e}")

    # Optionally compute FID (this is slow and requires many generated samples)
    fid_score = None
    if cfg["compute_fid"]:
        try:
            print(f"[{exp_id}] Computing FID (this will be slow)...")
            # Prepare a small real dataloader for FID reference
            real_loader = build_dataloader(dataset_root, batch_size=cfg["fid_batch_size"], shuffle=False)
            fid_score = compute_fid_score(
                model=model,
                noise_scheduler=noise_scheduler,
                real_dataloader=real_loader,
                num_samples=cfg["fid_num_samples"],
                batch_size=cfg["fid_batch_size"],
                device=device
            )
            print(f"[{exp_id}] FID = {fid_score:.4f}")
        except Exception as e:
            print(f"[{exp_id}] FID computation failed: {e}")
            fid_score = None

    # Summary metrics
    summary = {
        "exp_id": exp_id,
        "patch_size": config_dict["patch_size"],
        "depth": config_dict["depth"],
        "num_heads": config_dict["num_heads"],
        "timesteps": config_dict["timesteps"],
        "params": sum(p.numel() for p in model.parameters()),
        "avg_epoch_loss": float(epoch_losses[-1]) if len(epoch_losses) > 0 else None,
        "fid": float(fid_score) if fid_score is not None else None,
        "checkpoint": str(ckpt_path),
        "all_loss_png": str(all_loss_plot_path),
        "epoch_loss_png": str(epoch_loss_plot_path),
    }

    return summary

# -------------------------
# Master sweep
# -------------------------
def sweep_and_run(global_cfg, dataset_root):
    """
    Run experiments varying one hyperparameter at a time while keeping others at default.
    This allows us to observe trends for each hyperparameter independently.
    """
    out_root = make_output_dir()
    csv_path = out_root / "experiment_results.csv"
    fieldnames = ["exp_id", "varied_param", "patch_size", "depth", "num_heads", "timesteps", 
                  "params", "avg_epoch_loss", "fid", "checkpoint", "all_loss_png", "epoch_loss_png"]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        exp_ix = 0
        
        # Experiment 1: Vary patch size
        print("\n" + "="*60)
        print("EXPERIMENT SET 1: Varying Patch Size")
        print("="*60)
        for ps in global_cfg["patch_size_experiments"]:
            exp_ix += 1
            exp_id = f"exp_{exp_ix:03d}_p{ps}_d{DEFAULT_CONFIG['depth']}_h{DEFAULT_CONFIG['num_heads']}_T{DEFAULT_CONFIG['timesteps']}"
            run_out = out_root / exp_id
            config_dict = {
                "patch_size": ps,
                "depth": DEFAULT_CONFIG["depth"],
                "num_heads": DEFAULT_CONFIG["num_heads"],
                "timesteps": DEFAULT_CONFIG["timesteps"]
            }
            
            try:
                summary = run_single_experiment(global_cfg, dataset_root, run_out, exp_id, config_dict)
                summary["varied_param"] = "patch_size"
                writer.writerow({k: summary.get(k, "") for k in fieldnames})
                f.flush()
            except Exception as e:
                print(f"[{exp_id}] Experiment failed: {e}")
                writer.writerow({
                    "exp_id": exp_id, "varied_param": "patch_size",
                    "patch_size": ps, "depth": DEFAULT_CONFIG["depth"], 
                    "num_heads": DEFAULT_CONFIG["num_heads"], "timesteps": DEFAULT_CONFIG["timesteps"]
                })
                f.flush()
        
        # Experiment 2: Vary depth
        print("\n" + "="*60)
        print("EXPERIMENT SET 2: Varying Model Depth")
        print("="*60)
        for depth in global_cfg["depth_experiments"]:
            exp_ix += 1
            exp_id = f"exp_{exp_ix:03d}_p{DEFAULT_CONFIG['patch_size']}_d{depth}_h{DEFAULT_CONFIG['num_heads']}_T{DEFAULT_CONFIG['timesteps']}"
            run_out = out_root / exp_id
            config_dict = {
                "patch_size": DEFAULT_CONFIG["patch_size"],
                "depth": depth,
                "num_heads": DEFAULT_CONFIG["num_heads"],
                "timesteps": DEFAULT_CONFIG["timesteps"]
            }
            
            try:
                summary = run_single_experiment(global_cfg, dataset_root, run_out, exp_id, config_dict)
                summary["varied_param"] = "depth"
                writer.writerow({k: summary.get(k, "") for k in fieldnames})
                f.flush()
            except Exception as e:
                print(f"[{exp_id}] Experiment failed: {e}")
                writer.writerow({
                    "exp_id": exp_id, "varied_param": "depth",
                    "patch_size": DEFAULT_CONFIG["patch_size"], "depth": depth,
                    "num_heads": DEFAULT_CONFIG["num_heads"], "timesteps": DEFAULT_CONFIG["timesteps"]
                })
                f.flush()
        
        # Experiment 3: Vary num_heads
        print("\n" + "="*60)
        print("EXPERIMENT SET 3: Varying Attention Heads")
        print("="*60)
        for heads in global_cfg["num_heads_experiments"]:
            exp_ix += 1
            exp_id = f"exp_{exp_ix:03d}_p{DEFAULT_CONFIG['patch_size']}_d{DEFAULT_CONFIG['depth']}_h{heads}_T{DEFAULT_CONFIG['timesteps']}"
            run_out = out_root / exp_id
            config_dict = {
                "patch_size": DEFAULT_CONFIG["patch_size"],
                "depth": DEFAULT_CONFIG["depth"],
                "num_heads": heads,
                "timesteps": DEFAULT_CONFIG["timesteps"]
            }
            
            try:
                summary = run_single_experiment(global_cfg, dataset_root, run_out, exp_id, config_dict)
                summary["varied_param"] = "num_heads"
                writer.writerow({k: summary.get(k, "") for k in fieldnames})
                f.flush()
            except Exception as e:
                print(f"[{exp_id}] Experiment failed: {e}")
                writer.writerow({
                    "exp_id": exp_id, "varied_param": "num_heads",
                    "patch_size": DEFAULT_CONFIG["patch_size"], "depth": DEFAULT_CONFIG["depth"],
                    "num_heads": heads, "timesteps": DEFAULT_CONFIG["timesteps"]
                })
                f.flush()
        
        # Experiment 4: Vary timesteps
        print("\n" + "="*60)
        print("EXPERIMENT SET 4: Varying Diffusion Timesteps")
        print("="*60)
        for tsteps in global_cfg["timesteps_experiments"]:
            exp_ix += 1
            exp_id = f"exp_{exp_ix:03d}_p{DEFAULT_CONFIG['patch_size']}_d{DEFAULT_CONFIG['depth']}_h{DEFAULT_CONFIG['num_heads']}_T{tsteps}"
            run_out = out_root / exp_id
            config_dict = {
                "patch_size": DEFAULT_CONFIG["patch_size"],
                "depth": DEFAULT_CONFIG["depth"],
                "num_heads": DEFAULT_CONFIG["num_heads"],
                "timesteps": tsteps
            }
            
            try:
                summary = run_single_experiment(global_cfg, dataset_root, run_out, exp_id, config_dict)
                summary["varied_param"] = "timesteps"
                writer.writerow({k: summary.get(k, "") for k in fieldnames})
                f.flush()
            except Exception as e:
                print(f"[{exp_id}] Experiment failed: {e}")
                writer.writerow({
                    "exp_id": exp_id, "varied_param": "timesteps",
                    "patch_size": DEFAULT_CONFIG["patch_size"], "depth": DEFAULT_CONFIG["depth"],
                    "num_heads": DEFAULT_CONFIG["num_heads"], "timesteps": tsteps
                })
                f.flush()

    total_experiments = (len(global_cfg["patch_size_experiments"]) + 
                        len(global_cfg["depth_experiments"]) + 
                        len(global_cfg["num_heads_experiments"]) + 
                        len(global_cfg["timesteps_experiments"]))
    
    print("\n" + "="*60)
    print(f"✅ All {total_experiments} experiments completed!")
    print(f"Results saved to: {csv_path}")
    print("="*60)
    
    return out_root

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    cfg = EXPERIMENT_CONFIG.copy()
    dataset_root = os.environ.get("LANDSCAPE_DATASET_PATH", None)
    
    # If dataset_path not set via environment variable, try to auto-detect
    if dataset_root is None:
        # Try kagglehub cache location (standard path)
        import kagglehub
        try:
            dataset_root = kagglehub.dataset_download("arnaud58/landscape-pictures")
            print(f"Dataset found/downloaded at: {dataset_root}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise RuntimeError("Set LANDSCAPE_DATASET_PATH env var or ensure kagglehub can download the dataset.")
    
    print(f"Using dataset from: {dataset_root}")
    print(f"Number of images: {len([f for f in os.listdir(dataset_root) if f.endswith('.jpg')])}")

    # Run sweep (will create experiments/<timestamp>/ directory)
    out_root = sweep_and_run(cfg, dataset_root)
    print("Experiment artifacts stored under:", out_root)
