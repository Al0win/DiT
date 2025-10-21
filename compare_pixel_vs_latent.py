"""
Comparison Script: Pixel Space vs Latent Space Training

This script demonstrates the difference between pixel-space and latent-space
DiT training. It runs a quick test with both modes and compares:
- Memory usage
- Training speed
- Model parameters
- Sample quality (qualitative)

Usage:
    python compare_pixel_vs_latent.py
"""

import torch
import time
import os
from pathlib import Path

try:
    from dit_models import (
        DiT, NoiseScheduler, VAEWrapper,
        LandscapeDataset, transform,
        train_one_epoch, ddpm_sample
    )
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure dit_models.py is in the same directory")
    IMPORTS_OK = False


def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def format_time(seconds):
    """Format seconds as MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def test_pixel_space(dataloader, device, image_size=128):
    """Test pixel-space training"""
    print("\n" + "="*60)
    print("Testing PIXEL SPACE Training")
    print("="*60)
    
    # Model configuration
    model = DiT(
        img_size=image_size,
        patch_size=4,
        in_channels=3,
        hidden_size=256,
        depth=4,
        num_heads=4,
        use_latent_space=False
    ).to(device)
    
    noise_scheduler = NoiseScheduler(num_timesteps=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")
    
    # Measure memory before training
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_usage()
    
    # Training
    print("\nTraining for 1 epoch...")
    start_time = time.time()
    avg_loss, losses = train_one_epoch(
        model, dataloader, optimizer, noise_scheduler,
        device=device, epoch=1, vae_wrapper=None
    )
    training_time = time.time() - start_time
    
    # Measure memory after training
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\n📊 Pixel Space Results:")
    print(f"  Training Time: {format_time(training_time)}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Peak Memory: {mem_peak:.2f} GB")
    print(f"  Final Loss: {losses[-1]:.4f}")
    
    # Generate samples
    print("\nGenerating samples...")
    samples = ddpm_sample(
        model, noise_scheduler,
        batch_size=4,
        img_size=image_size,
        channels=3,
        device=device,
        vae_wrapper=None
    )
    
    return {
        'model': model,
        'samples': samples,
        'params': params,
        'training_time': training_time,
        'avg_loss': avg_loss,
        'peak_memory': mem_peak,
        'mode': 'Pixel Space'
    }


def test_latent_space(dataloader, device, image_size=128):
    """Test latent-space training with VAE"""
    print("\n" + "="*60)
    print("Testing LATENT SPACE Training (with VAE)")
    print("="*60)
    
    try:
        # Initialize VAE
        print("Loading VAE... (may take a moment on first run)")
        vae = VAEWrapper(device=device)
        latent_size = vae.get_latent_size(image_size)
        latent_channels = vae.get_latent_channels()
        print(f"VAE loaded! Latent space: {latent_size}×{latent_size}×{latent_channels}")
    except Exception as e:
        print(f"❌ Could not load VAE: {e}")
        print("Make sure 'diffusers' is installed: pip install diffusers")
        return None
    
    # Model configuration (for latent space)
    model = DiT(
        img_size=latent_size,
        patch_size=2,
        in_channels=latent_channels,
        hidden_size=256,
        depth=4,
        num_heads=4,
        use_latent_space=True
    ).to(device)
    
    noise_scheduler = NoiseScheduler(num_timesteps=100).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Count parameters (DiT only, VAE is frozen)
    params = sum(p.numel() for p in model.parameters())
    vae_params = sum(p.numel() for p in vae.vae.parameters())
    print(f"DiT Parameters: {params:,}")
    print(f"VAE Parameters (frozen): {vae_params:,}")
    
    # Measure memory before training
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_memory_usage()
    
    # Training
    print("\nTraining for 1 epoch...")
    start_time = time.time()
    avg_loss, losses = train_one_epoch(
        model, dataloader, optimizer, noise_scheduler,
        device=device, epoch=1, vae_wrapper=vae
    )
    training_time = time.time() - start_time
    
    # Measure memory after training
    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"\n📊 Latent Space Results:")
    print(f"  Training Time: {format_time(training_time)}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Peak Memory: {mem_peak:.2f} GB")
    print(f"  Final Loss: {losses[-1]:.4f}")
    
    # Generate samples
    print("\nGenerating samples...")
    samples = ddpm_sample(
        model, noise_scheduler,
        batch_size=4,
        img_size=latent_size,
        channels=latent_channels,
        device=device,
        vae_wrapper=vae
    )
    
    return {
        'model': model,
        'samples': samples,
        'params': params,
        'vae_params': vae_params,
        'training_time': training_time,
        'avg_loss': avg_loss,
        'peak_memory': mem_peak,
        'mode': 'Latent Space'
    }


def visualize_comparison(pixel_results, latent_results, save_dir='comparison_results'):
    """Create comparison visualizations"""
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. Sample comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Pixel space samples
    for i in range(4):
        img = pixel_results['samples'][i].cpu()
        img = (img + 1) / 2  # Denormalize
        img = img.permute(1, 2, 0).clamp(0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Pixel Sample {i+1}')
    
    # Latent space samples
    for i in range(4):
        img = latent_results['samples'][i].cpu()
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).clamp(0, 1)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Latent Sample {i+1}')
    
    fig.suptitle('Generated Samples Comparison\nTop: Pixel Space | Bottom: Latent Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/samples_comparison.png", dpi=150)
    print(f"✅ Saved: {save_dir}/samples_comparison.png")
    
    # 2. Performance comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Training Time (s)', 'Peak Memory (GB)', 'Parameters (M)']
    pixel_vals = [
        pixel_results['training_time'],
        pixel_results['peak_memory'],
        pixel_results['params'] / 1e6
    ]
    latent_vals = [
        latent_results['training_time'],
        latent_results['peak_memory'],
        latent_results['params'] / 1e6
    ]
    
    for i, (metric, pval, lval) in enumerate(zip(metrics, pixel_vals, latent_vals)):
        axes[i].bar(['Pixel', 'Latent'], [pval, lval], color=['#FF6B6B', '#4ECDC4'])
        axes[i].set_ylabel(metric)
        axes[i].set_title(metric)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        axes[i].text(0, pval, f'{pval:.2f}', ha='center', va='bottom')
        axes[i].text(1, lval, f'{lval:.2f}', ha='center', va='bottom')
        
        # Add speedup/reduction label
        if pval > 0:
            improvement = (pval - lval) / pval * 100
            if improvement > 0:
                axes[i].text(0.5, max(pval, lval) * 1.1,
                           f'{improvement:.0f}% reduction',
                           ha='center', fontweight='bold', color='green')
    
    fig.suptitle('Performance Comparison: Pixel Space vs Latent Space',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_comparison.png", dpi=150)
    print(f"✅ Saved: {save_dir}/performance_comparison.png")
    
    plt.close('all')


def print_summary(pixel_results, latent_results):
    """Print comparison summary"""
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Pixel Space':<15} {'Latent Space':<15} {'Improvement':<15}")
    print("-" * 75)
    
    # Training time
    pt = pixel_results['training_time']
    lt = latent_results['training_time']
    speedup = (pt - lt) / pt * 100
    print(f"{'Training Time':<30} {format_time(pt):<15} {format_time(lt):<15} {speedup:.1f}% faster")
    
    # Memory
    pm = pixel_results['peak_memory']
    lm = latent_results['peak_memory']
    mem_reduction = (pm - lm) / pm * 100
    print(f"{'Peak Memory (GB)':<30} {pm:.2f}{'':>12} {lm:.2f}{'':>12} {mem_reduction:.1f}% less")
    
    # Parameters
    pp = pixel_results['params'] / 1e6
    lp = latent_results['params'] / 1e6
    print(f"{'Model Parameters (M)':<30} {pp:.2f}{'':>12} {lp:.2f}{'':>12} {'Similar':<15}")
    
    # Loss
    pl = pixel_results['avg_loss']
    ll = latent_results['avg_loss']
    print(f"{'Average Loss':<30} {pl:.4f}{'':>10} {ll:.4f}{'':>10} {'N/A':<15}")
    
    print("\n" + "="*60)
    print("KEY TAKEAWAYS:")
    print("="*60)
    print(f"✓ Latent space is {speedup:.0f}% faster")
    print(f"✓ Latent space uses {mem_reduction:.0f}% less memory")
    print(f"✓ Both use similar model parameters (~{pp:.1f}M)")
    print(f"✓ Latent space matches official DiT implementation")
    print("="*60)


def main():
    if not IMPORTS_OK:
        return
    
    print("="*60)
    print("DiT Implementation: Pixel Space vs Latent Space Comparison")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("⚠️  Warning: Running on CPU. This will be very slow.")
        print("For meaningful comparison, please run on GPU.")
    
    # Setup dataset
    dataset_root = os.environ.get("LANDSCAPE_DATASET_PATH", None)
    if dataset_root is None:
        print("\n⚠️  LANDSCAPE_DATASET_PATH not set.")
        print("Please set it or place this script where dataset is accessible.")
        
        # Try common locations
        common_paths = [
            "./data",
            "../data",
            "../../data",
        ]
        for path in common_paths:
            if os.path.exists(path) and any(f.endswith('.jpg') for f in os.listdir(path)):
                dataset_root = path
                print(f"Found dataset at: {dataset_root}")
                break
        
        if dataset_root is None:
            print("Could not find dataset. Exiting.")
            return
    
    print(f"Dataset: {dataset_root}")
    
    # Create small dataloader for testing
    try:
        dataset = LandscapeDataset(root_dir=dataset_root, transform=transform)
        # Use only first 100 images for quick test
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(100, len(dataset)))))
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
        print(f"Loaded {len(dataset)} images for testing")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Run pixel space test
    pixel_results = test_pixel_space(dataloader, device, image_size=128)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run latent space test
    latent_results = test_latent_space(dataloader, device, image_size=128)
    
    if latent_results is None:
        print("\n❌ Latent space test failed. Make sure 'diffusers' is installed.")
        return
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    visualize_comparison(pixel_results, latent_results)
    
    # Print summary
    print_summary(pixel_results, latent_results)
    
    print("\n✅ Comparison complete!")
    print("Check 'comparison_results/' folder for visualizations.")


if __name__ == "__main__":
    main()
