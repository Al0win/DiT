import os
import csv
import time
import torch
import pandas as pd
import kagglehub
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse  # <-- Import argparse

# Import definitions from your dit_models module
from dit_models import (
    DiT,
    NoiseScheduler,
    LandscapeDataset,
    transform,
    VAEWrapper,
    compute_fid_score
)

# Set cache directory to /mnt/local/hf to avoid filling up /home/jovyan
CACHE_DIR = "/mnt/local/hf/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables for HuggingFace cache (MUST be set before loading models)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR

# --- CONFIGURATION ---

# IMPORTANT: Set this to the path of your experiment directory
# This is the folder that contains your `experiment_results.csv`
EXPERIMENT_DIR = "experiments/20251021_053158"

# VAE and Model settings (must match your training)
VAE_MODEL_NAME = "stabilityai/sd-vae-ft-ema"
IMAGE_SIZE = 128       # Original image size
HIDDEN_SIZE = 384      # Fixed hidden size from your config
BATCH_SIZE = 256       # <-- EASY WIN: Increased batch size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------

def build_real_dataloader(dataset_root, batch_size):
    """Builds a dataloader for the full real dataset."""
    ds = LandscapeDataset(root_dir=dataset_root, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Loaded real dataset with {len(ds)} images.")
    return dl, len(ds)

def find_dataset_root():
    """Finds the dataset path, same as your experiment script."""
    dataset_root = os.environ.get("LANDSCAPE_DATASET_PATH", None)
    if dataset_root is None:
        print("LANDSCAPE_DATASET_PATH not set, trying kagglehub...")
        try:
            dataset_root = kagglehub.dataset_download("arnaud58/landscape-pictures")
            print(f"Dataset found/downloaded at: {dataset_root}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise RuntimeError("Set LANDSCAPE_DATASET_PATH env var or ensure kagglehub can download the dataset.")
    return dataset_root

if __name__ == "__main__":
    # --- NEW: Add argument parser for parallel runs ---
    parser = argparse.ArgumentParser(description="Parallel FID Evaluator")
    parser.add_argument('--range', type=str, default="0-14", 
                        help="Range of CSV rows to process (e.g., '0-7')")
    parser.add_argument('--output', type=str, default="experiment_results_new_fid.csv", 
                        help="Output CSV file name")
    args = parser.parse_args()

    # Parse the range
    try:
        start_row, end_row = map(int, args.range.split('-'))
    except ValueError:
        print("Error: --range must be in 'start-end' format (e.g., '0-7')")
        exit()
    # --- END NEW ---

    if not os.path.exists(EXPERIMENT_DIR):
        print(f"Error: Experiment directory not found at {EXPERIMENT_DIR}")
        print("Please update the EXPERIMENT_DIR variable in this script.")
    else:
        # --- 1. Setup Dataset and VAE (once) ---
        print(f"Using device: {DEVICE}")
        dataset_root = find_dataset_root()
        real_dataloader, num_real_samples = build_real_dataloader(dataset_root, BATCH_SIZE)
        
        NEW_FID_NUM_SAMPLES = num_real_samples

        print("Initializing VAE (this may take a moment)...")
        vae_wrapper = VAEWrapper(vae_model_name=VAE_MODEL_NAME, device=DEVICE)
        latent_size = vae_wrapper.get_latent_size(IMAGE_SIZE)
        latent_channels = vae_wrapper.get_latent_channels()
        print(f"VAE initialized. Latent size: {latent_size}, Channels: {latent_channels}")

        # --- 2. Load CSV ---
        csv_path = os.path.join(EXPERIMENT_DIR, "experiment_results.csv")
        try:
            df = pd.read_csv(csv_path)
            total_rows = len(df)
            # --- NEW: Slice the DataFrame based on the --range argument ---
            df = df.iloc[start_row:end_row + 1]
            print(f"Loaded {csv_path}. Processing rows {start_row} to {end_row} (total {len(df)} models).")
            # --- END NEW ---
        except FileNotFoundError:
            print(f"Error: {csv_path} not found!")
            exit()
        
        new_fid_scores = []
        torch.backends.cudnn.benchmark = True
        
        # --- 3. Iterate and Re-compute FID ---
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Computing FIDs (Rows {start_row}-{end_row})"):
            exp_id = row['exp_id']
            try:
                # 1. Re-create model from row info
                model = DiT(
                    img_size=latent_size,
                    patch_size=int(row['patch_size']),
                    in_channels=latent_channels,
                    hidden_size=HIDDEN_SIZE,
                    depth=int(row['depth']),
                    num_heads=int(row['num_heads']),
                    use_latent_space=bool(row['latent_space'])
                ).to(DEVICE)
                
                # 2. Load checkpoint
                checkpoint_path = row['checkpoint']
                # Check if path in CSV is relative and fix if needed
                if not os.path.exists(checkpoint_path) and os.path.exists(os.path.join(EXPERIMENT_DIR, '..', checkpoint_path)):
                    checkpoint_path = os.path.join(EXPERIMENT_DIR, '..', checkpoint_path)
                
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # 3. Re-create scheduler
                noise_scheduler = NoiseScheduler(num_timesteps=int(row['timesteps'])).to(DEVICE)
                
                # 4. Compute new FID with *full* sample count
                fid_score = compute_fid_score(
                    model=model,
                    noise_scheduler=noise_scheduler,
                    real_dataloader=real_dataloader,
                    num_samples=NEW_FID_NUM_SAMPLES, # The crucial change
                    batch_size=BATCH_SIZE,
                    device=DEVICE,
                    vae_wrapper=vae_wrapper,
                    latent_size=latent_size,
                    latent_channels=latent_channels
                )
                
                tqdm.write(f"  [{exp_id}] Old FID: {row.get('fid', 'N/A')}, New FID: {fid_score:.4f}")
                new_fid_scores.append(fid_score)

            except Exception as e:
                tqdm.write(f"  [Failed] {exp_id}: {e}")
                new_fid_scores.append(None)

        # --- 4. Save New CSV ---
        # Overwrite the old 'fid' column with the new, reliable scores
        df['fid'] = new_fid_scores
        
        # --- NEW: Use the output file argument ---
        new_csv_path = os.path.join(EXPERIMENT_DIR, args.output)
        df.to_csv(new_csv_path, index=False)
        
        print("\n" + "="*60)
        print(f"✅ FID re-computation complete for rows {start_row}-{end_row}!")
        print(f"New results saved to: {new_csv_path}")
        print("="*60)