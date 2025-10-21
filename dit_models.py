"""
DiT (Diffusion Transformer) Models and Utilities Module

This module contains all the core classes and functions for DiT training and inference,
without any execution code. Use this for importing into other scripts.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid
from scipy import linalg
from torchvision.models import inception_v3

# VAE imports
try:
    from diffusers.models import AutoencoderKL
    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("Warning: diffusers not installed. VAE functionality will not be available.")
    print("Install with: pip install diffusers")


# ============================================================================
# Dataset
# ============================================================================

class LandscapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0


# Default transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# ============================================================================
# Noise Scheduler
# ============================================================================

class NoiseScheduler:
    """
    Implements a linear noise schedule for the diffusion process.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Only define the linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Square roots of cumulative products (for forward diffusion)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Additional terms for reverse diffusion (sampling)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance for DDPM sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def to(self, device):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to images according to the diffusion schedule.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * noise


# ============================================================================
# Model Components
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Number of patches along each dimension (height and width)
    
    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] positional embedding
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # Split into height and width embeddings
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Create 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Output dimension for each position
        pos: Grid of positions, shape (M,)
    
    Returns:
        emb: (M, D) positional embeddings
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding layer."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PatchEmbed(nn.Module):
    """Converts images into sequences of flattened patches."""
    def __init__(self, img_size=128, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm conditioning (adaLN)."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) model for image generation.
    
    Args:
        img_size: Size of input image (or latent representation)
        patch_size: Size of patches to divide the image into
        in_channels: Number of input channels (3 for RGB, 4 for latent space)
        hidden_size: Dimension of the transformer hidden state
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        use_latent_space: If True, expects 4-channel latent inputs (img_size should be latent_size)
    """
    def __init__(
        self,
        img_size=128,
        patch_size=4,
        in_channels=3,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        use_latent_space=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.use_latent_space = use_latent_space
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches
        
        self.time_embed = TimeEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Fixed sin-cos positional embedding (not learned, like official DiT)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
            
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize (and freeze) pos_embed with 2D sin-cos pattern
        # This matches the official DiT implementation
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)
        
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)
        
        nn.init.constant_(self.linear_final.weight, 0)
        nn.init.constant_(self.linear_final.bias, 0)
        
    def unpatchify(self, x):
        """Convert patch tokens back to images."""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        return imgs
        
    def forward(self, x, t):
        x = self.patch_embed(x) + self.pos_embed
        t_emb = self.time_embed(t)
        c = self.time_mlp(t_emb)
        
        for block in self.blocks:
            x = block(x, c)
        
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear_final(x)
        x = self.unpatchify(x)

        return x


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, noise_scheduler, device, epoch=0, vae_wrapper=None):
    """Train the DiT model for one epoch.
    
    Args:
        model: DiT model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        noise_scheduler: Noise scheduler for diffusion
        device: Device to train on
        epoch: Current epoch number
        vae_wrapper: Optional VAEWrapper for latent space training
    """
    model.train()
    total_loss = 0
    losses = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(progress_bar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Encode to latent space if using VAE
        if vae_wrapper is not None:
            with torch.no_grad():
                images = vae_wrapper.encode(images)
        
        t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
        
        noise = torch.randn_like(images)
        noisy_images, _ = noise_scheduler.add_noise(images, t, noise)
        
        predicted_noise = model(noisy_images, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, losses


def train_n_epochs(model, dataloader, optimizer, noise_scheduler, device, num_epochs=5, vae_wrapper=None):
    """Train the DiT model for multiple epochs.
    
    Args:
        model: DiT model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        noise_scheduler: Noise scheduler for diffusion
        device: Device to train on
        num_epochs: Number of epochs to train
        vae_wrapper: Optional VAEWrapper for latent space training
    """
    epoch_losses = []
    all_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        avg_loss, losses = train_one_epoch(model, dataloader, optimizer, noise_scheduler, 
                                          device, epoch + 1, vae_wrapper=vae_wrapper)
        
        epoch_losses.append(avg_loss)
        all_losses.extend(losses)
        
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    return epoch_losses, all_losses


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_loss(losses, save_path='loss_curve.png'):
    """Visualize training loss trend."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_epoch_loss(epoch_losses, save_path='epoch_loss_curve.png'):
    """Visualize average loss per epoch."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# ============================================================================
# Sampling Functions
# ============================================================================

@torch.no_grad()
def ddpm_sample(model, noise_scheduler, batch_size=1, img_size=128, 
                channels=3, device='cuda', save_intermediates=False, vae_wrapper=None):
    """DDPM sampling process (reverse diffusion).
    
    Args:
        model: DiT model
        noise_scheduler: Noise scheduler
        batch_size: Number of samples to generate
        img_size: Size of output images (or latent size if using VAE)
        channels: Number of channels (3 for RGB, 4 for latents)
        device: Device to use
        save_intermediates: Whether to save intermediate denoising steps
        vae_wrapper: Optional VAEWrapper for decoding latents to images
    """
    model.eval()
    
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    intermediates = [] if save_intermediates else None
    
    for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        predicted_noise = model(x, t_batch)
        
        alpha_t = noise_scheduler.alphas[t]
        alpha_t_cumprod = noise_scheduler.alphas_cumprod[t]
        alpha_t_cumprod_prev = noise_scheduler.alphas_cumprod_prev[t]
        
        pred_x0 = (x - torch.sqrt(1 - alpha_t_cumprod) * predicted_noise) / torch.sqrt(alpha_t_cumprod)
        
        # Only clamp if not in latent space (latents don't need clamping)
        if vae_wrapper is None:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        posterior_mean = (
            noise_scheduler.posterior_mean_coef1[t] * pred_x0 +
            noise_scheduler.posterior_mean_coef2[t] * x
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            posterior_variance = noise_scheduler.posterior_variance[t]
            x = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x = posterior_mean
        
        if save_intermediates and t % (noise_scheduler.num_timesteps // 10) == 0:
            intermediates.append(pred_x0.cpu())
    
    # Decode from latent space if using VAE
    if vae_wrapper is not None:
        x = vae_wrapper.decode(x)
        if intermediates is not None:
            intermediates = [vae_wrapper.decode(inter.to(device)).cpu() for inter in intermediates]
    
    if save_intermediates:
        return x, intermediates
    return x

# ============================================================================
# VAE Wrapper
# ============================================================================

class VAEWrapper:
    """
    Wrapper for Stable Diffusion VAE (AutoencoderKL).
    Handles encoding images to latent space and decoding back.
    """
    def __init__(self, vae_model_name="stabilityai/sd-vae-ft-ema", device='cuda'):
        if not VAE_AVAILABLE:
            raise ImportError("diffusers package is required for VAE functionality. Install with: pip install diffusers")
        
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
        self.vae.eval()
        # Scaling factor used in Stable Diffusion
        self.scale_factor = 0.18215
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to latent space.
        Args:
            images: (B, 3, H, W) tensor of images in range [-1, 1]
        Returns:
            latents: (B, 4, H//8, W//8) tensor of latents
        """
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.scale_factor
        return latents
    
    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents to images.
        Args:
            latents: (B, 4, H//8, W//8) tensor of latents
        Returns:
            images: (B, 3, H, W) tensor of images in range [-1, 1]
        """
        latents = latents / self.scale_factor
        images = self.vae.decode(latents).sample
        return images
    
    def get_latent_size(self, image_size):
        """Get the latent size for a given image size (always image_size // 8)"""
        return image_size // 8
    
    def get_latent_channels(self):
        """Get the number of latent channels (always 4 for SD VAE)"""
        return 4


# ============================================================================
# FID Computation
# ============================================================================

class InceptionFeatureExtractor:
    """Extract features from Inception v3 for FID calculation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def get_features(self, images):
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        images = self.preprocess(images)
        features = self.model(images)
        return features.cpu().numpy()


def calculate_fid(real_features, generated_features):
    """Calculate Fréchet Inception Distance."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


@torch.no_grad()
def compute_fid_score(model, noise_scheduler, real_dataloader, 
                      num_samples=1000, batch_size=32, device='cuda',
                      vae_wrapper=None, latent_size=None, latent_channels=4):
    """Compute FID score for generated samples vs real images.
    
    Args:
        model: DiT model
        noise_scheduler: Noise scheduler
        real_dataloader: DataLoader for real images
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        vae_wrapper: Optional VAEWrapper for latent space models
        latent_size: Size of latent space (if using VAE)
        latent_channels: Number of latent channels (if using VAE)
    """
    feature_extractor = InceptionFeatureExtractor(device)
    
    print("Extracting features from real images...")
    real_features = []
    for images, _ in tqdm(real_dataloader):
        if len(real_features) * images.shape[0] >= num_samples:
            break
        images = images.to(device)
        features = feature_extractor.get_features(images)
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    
    print("Generating samples and extracting features...")
    generated_features = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, num_samples - len(generated_features) * batch_size)
        
        # Determine correct parameters based on VAE usage
        if vae_wrapper is not None:
            samples = ddpm_sample(
                model, noise_scheduler, 
                batch_size=current_batch_size,
                img_size=latent_size,
                channels=latent_channels,
                device=device,
                vae_wrapper=vae_wrapper
            )
        else:
            samples = ddpm_sample(
                model, noise_scheduler, 
                batch_size=current_batch_size,
                device=device
            )
        
        features = feature_extractor.get_features(samples)
        generated_features.append(features)
    
    generated_features = np.concatenate(generated_features, axis=0)[:num_samples]
    
    fid_score = calculate_fid(real_features, generated_features)
    return fid_score
