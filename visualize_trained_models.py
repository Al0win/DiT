"""
Visualize Evolution from Trained DiT Models

This script loads trained DiT model checkpoints and generates 10x10 grids
showing the evolution of the denoising process from noise to clean images.
Similar to the pretrained model visualization, but for custom-trained models.
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys

# Import model classes from dit_models.py
from dit_models import DiT, NoiseScheduler, VAEWrapper

def load_trained_model(checkpoint_path, device='cuda'):
    """Load a trained DiT model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters from checkpoint
    config = checkpoint.get('config', {})
    patch_size = config.get('patch_size', 2)
    depth = config.get('depth', 8)
    num_heads = config.get('num_heads', 6)
    num_timesteps = config.get('num_timesteps', 1000)
    use_vae = config.get('use_vae', False)
    latent_mode = config.get('latent_mode', False)
    
    print(f"  Config: patch={patch_size}, depth={depth}, heads={num_heads}, T={num_timesteps}, VAE={use_vae}")
    
    # Determine dimensions based on latent mode
    if latent_mode or use_vae:
        img_size = 32  # Latent space is 32x32
        in_channels = 4  # VAE latents have 4 channels
    else:
        img_size = 128  # Pixel space is 128x128
        in_channels = 3  # RGB has 3 channels
    
    # Initialize model
    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=384,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type='linear'
    ).to(device)
    
    # Initialize VAE if needed
    vae_wrapper = None
    if use_vae or latent_mode:
        vae_wrapper = VAEWrapper(device=device)
    
    return model, noise_scheduler, vae_wrapper, config


@torch.no_grad()
def generate_evolution_grid(model, noise_scheduler, vae_wrapper=None, 
                           num_steps=100, device='cuda', seed=42):
    """
    Generate a 10x10 grid showing evolution from noise to clean image.
    
    Similar to the pretrained model visualization, captures x_0 prediction
    at each step during the denoising process.
    """
    torch.manual_seed(seed)
    
    # Determine image/latent dimensions
    if vae_wrapper is not None:
        img_size = 32  # Latent space
        channels = 4   # VAE latents
    else:
        img_size = 128  # Pixel space
        channels = 3    # RGB
    
    # Initialize random noise
    x = torch.randn(1, channels, img_size, img_size, device=device)
    
    # Sample timesteps uniformly
    step_size = noise_scheduler.num_timesteps // num_steps
    timesteps = list(range(0, noise_scheduler.num_timesteps, step_size))[:num_steps]
    timesteps.reverse()
    
    # Storage for x_0 predictions at each step
    x0_predictions = []
    
    print(f"Generating evolution with {num_steps} steps...")
    print(f"  Image size: {img_size}x{img_size}, Channels: {channels}")
    print(f"  Using VAE: {vae_wrapper is not None}")
    
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Predict x_0 (clean image/latent)
        alpha_t_cumprod = noise_scheduler.alphas_cumprod[t]
        pred_x0 = (x - torch.sqrt(1 - alpha_t_cumprod) * predicted_noise) / torch.sqrt(alpha_t_cumprod)
        
        # Clamp x_0 prediction (only for pixel space, not latents)
        if vae_wrapper is None:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Decode from latent space if using VAE
        if vae_wrapper is not None:
            pred_x0_decoded = vae_wrapper.decode(pred_x0)
        else:
            pred_x0_decoded = pred_x0
        
        # Convert to PIL image
        image = (pred_x0_decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        # Save x_0 prediction
        x0_predictions.append(image)
        
        # Update x for next iteration (DDPM step)
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_t_cumprod_prev = noise_scheduler.alphas_cumprod[t_prev]
            
            # Compute posterior mean
            posterior_mean_coef1 = noise_scheduler.posterior_mean_coef1[t]
            posterior_mean_coef2 = noise_scheduler.posterior_mean_coef2[t]
            x = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x
            
            # Add noise if not at the last step
            if t_prev > 0:
                noise = torch.randn_like(x)
                posterior_variance = noise_scheduler.posterior_variance[t]
                x = x + torch.sqrt(posterior_variance) * noise
    
    return x0_predictions


def visualize_single_experiment(exp_dir, output_dir, device='cuda', seed=42):
    """Visualize evolution for a single experiment."""
    exp_name = os.path.basename(exp_dir)
    checkpoint_path = os.path.join(exp_dir, f"{exp_name}_checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing: {exp_name}")
    print(f"{'='*80}")
    
    # Load model
    model, noise_scheduler, vae_wrapper, config = load_trained_model(checkpoint_path, device)
    
    # Generate evolution grid
    x0_predictions = generate_evolution_grid(
        model, noise_scheduler, vae_wrapper, 
        num_steps=100, device=device, seed=seed
    )
    
    # Create 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize=(30, 30))
    
    for i in range(10):
        for j in range(10):
            idx = i * 10 + j
            if idx < len(x0_predictions):
                axes[i, j].imshow(x0_predictions[idx])
                # Calculate actual timestep
                timestep_num = int((idx / 99) * (config.get('num_timesteps', 1000) - 1))
                axes[i, j].set_title(f"Step {timestep_num}", fontsize=8, pad=2)
                
                # Highlight final image
                if idx == len(x0_predictions) - 1:
                    for spine in axes[i, j].spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)
            axes[i, j].axis('off')
    
    # Add title with experiment details
    config_str = f"patch={config.get('patch_size', 2)}, depth={config.get('depth', 8)}, heads={config.get('num_heads', 6)}, T={config.get('num_timesteps', 1000)}"
    plt.suptitle(
        f"Evolution of x_0 Prediction: Noise → Clean (100 Steps)\n"
        f"Experiment: {exp_name}\nConfig: {config_str}",
        fontsize=20, y=0.995, fontweight='bold'
    )
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{exp_name}_evolution_10x10.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def visualize_all_experiments(experiments_root, output_dir, device='cuda', seed=42):
    """Visualize evolution for all experiments in a directory."""
    # Find all experiment directories
    exp_dirs = []
    for item in os.listdir(experiments_root):
        item_path = os.path.join(experiments_root, item)
        if os.path.isdir(item_path) and item.startswith('exp_'):
            checkpoint_path = os.path.join(item_path, f"{item}_checkpoint.pth")
            if os.path.exists(checkpoint_path):
                exp_dirs.append(item_path)
    
    exp_dirs.sort()
    
    print(f"\nFound {len(exp_dirs)} experiments to visualize")
    
    for exp_dir in exp_dirs:
        try:
            visualize_single_experiment(exp_dir, output_dir, device, seed)
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # Configuration
    EXPERIMENTS_ROOT = "/mnt/local/hf/DiT/experiments/20251021_053158"
    OUTPUT_DIR = "/mnt/local/hf/DiT/visualizations/trained_models"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    print("="*80)
    print("Visualizing Evolution from Trained DiT Models")
    print("="*80)
    print(f"Experiments directory: {EXPERIMENTS_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    
    # Visualize all experiments
    visualize_all_experiments(EXPERIMENTS_ROOT, OUTPUT_DIR, DEVICE, SEED)
    
    print("\n" + "="*80)
    print("✅ All visualizations complete!")
    print("="*80)
    print(f"Output saved to: {OUTPUT_DIR}")
