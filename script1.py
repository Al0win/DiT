# %% [markdown]
# # Dataset Download
# Download the Landscape Pictures dataset from Kaggle using the provided code snippet.

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("arnaud58/landscape-pictures")

print("Path to dataset files:", path)

# %%
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

img = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.jpg')][:1] #limit to 1 file
img = Image.open(img[0])
plt.imshow(img)
plt.show()
# Define a custom Dataset class
class LandscapeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return image and a dummy label (0)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Path to the dataset
dataset_path = path

# Create the dataset and DataLoader
dataset = LandscapeDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # Set shuffle to True when training

# Example: Iterate through the DataLoader
for images, labels in dataloader:
    print(images.shape, labels.shape)
    plt.imshow(images[0].permute(1, 2, 0) * 0.5 + 0.5)  # Unnormalize and display the first image
    plt.show()
    break

# %% [markdown]
# # Forward Process and Input Preparation Components
# Implement functions for adding noise to images, a noise scheduler, a time embedding layer, and a patchify layer. Experiment with different patch sizes.

# %% [markdown]
# # Noise Scheduler
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class NoiseScheduler:
    """
    Implements noise scheduling for the diffusion process.
    Supports both linear and cosine schedules.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Square roots of cumulative products (for forward diffusion)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # NEW: Additional terms for reverse diffusion (sampling)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # NEW: Posterior variance for DDPM sampling
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        
        # NEW: Posterior mean coefficients
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
        
        # NEW: Move new tensors to device
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to images according to the diffusion schedule.
        
        Args:
            x_0: Original images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional noise tensor [B, C, H, W]
        
        Returns:
            x_t: Noisy images
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get the appropriate alpha values for each timestep
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Apply the forward diffusion process: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return x_t, noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        NEW: Predict x_0 from x_t and predicted noise.
        This is used during sampling.
        """
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

# %% [markdown]
# # Time Embedding Layer
# Implement a time embedding layer for DiT conditioning using sinusoidal positional embeddings.

# %%
class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding layer for conditioning the DiT on timesteps.
    Based on the Transformer positional encoding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: Timesteps [B]
        Returns:
            Time embeddings [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# %% [markdown]
# # Patchify Layer
# Implement a patchify layer to convert images into sequences of flattened patches.

# %%
class PatchEmbed(nn.Module):
    """
    Converts images into sequences of flattened patches (like ViT).
    """
    def __init__(self, img_size=128, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Use Conv2d as a more efficient patchify operation
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: Images [B, C, H, W]
        Returns:
            Patches [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# %% [markdown]
# # Patch Size Experimentation
# Experiment with various patch sizes (e.g., 2×2, 4×4, 8×8) and note their impact.

# %%
# Placeholder for patch size experimentation

# %% [markdown]
# # DiT Block Implementation

# %%
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm conditioning (adaLN).
    Based on the DiT paper architecture.
    """
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
        
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        """
        Args:
            x: Input tokens [B, N, D]
            c: Conditioning (time embedding) [B, D]
        """
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with adaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with adaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


# %% [markdown]
# # DiT Model Implementation

# %%
class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) model for image generation.
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
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches
        
        # Time embedding
        self.time_embed = TimeEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Output layers - with adaLN modulation
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_final = nn.Linear(hidden_size, patch_size * patch_size * self.out_channels, bias=True)
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
            
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers with Xavier
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        
        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers in DiT blocks (CRITICAL!)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer adaLN modulation (CRITICAL!)
        nn.init.constant_(self.adaLN_modulation_final[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_final[-1].bias, 0)
        
        # Zero-out final linear layer (CRITICAL!)
        nn.init.constant_(self.linear_final.weight, 0)
        nn.init.constant_(self.linear_final.bias, 0)
        
    def unpatchify(self, x):
        """
        Convert patch tokens back to images.
        Args:
            x: [B, N, patch_size^2 * C]
        Returns:
            imgs: [B, C, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        return imgs
        
    def forward(self, x, t):
        """
        Args:
            x: Noisy images [B, C, H, W]
            t: Timesteps [B]
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Embed patches
        x = self.patch_embed(x) + self.pos_embed  # [B, N, D]
        
        # Embed time
        t_emb = self.time_embed(t)
        c = self.time_mlp(t_emb)  # [B, D]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer with adaLN modulation
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear_final(x)  # [B, N, patch_size^2 * C]

        # Unpatchify to get back image shape
        x = self.unpatchify(x)  # [B, C, H, W]

        return x

# %% [markdown]
# # Basic Training Loop for DiT Model
# Write a one-epoch training loop for the DiT model, including dataset setup, noise prediction, loss calculation, and optimizer updates.

# %%
def train_one_epoch(model, dataloader, optimizer, noise_scheduler, device, epoch=0):
    """
    Train the DiT model for one epoch.
    """
    model.train()
    total_loss = 0
    losses = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(progress_bar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
        
        # Add noise to images
        noise = torch.randn_like(images)
        noisy_images, _ = noise_scheduler.add_noise(images, t, noise)
        
        # Predict noise
        predicted_noise = model(noisy_images, t)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, losses

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

def train_n_epochs(model, dataloader, optimizer, noise_scheduler, device, num_epochs=5):
    """
    Train the DiT model for multiple epochs.
    
    Args:
        model: DiT model
        dataloader: Training dataloader
        optimizer: Optimizer
        noise_scheduler: NoiseScheduler object
        device: Device to use
        num_epochs: Number of epochs to train
    
    Returns:
        epoch_losses: List of average losses per epoch
        all_losses: List of all batch losses across all epochs
    """
    epoch_losses = []
    all_losses = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        avg_loss, losses = train_one_epoch(model, dataloader, optimizer, noise_scheduler, device, epoch + 1)
        
        epoch_losses.append(avg_loss)
        all_losses.extend(losses)
        
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    return epoch_losses, all_losses

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

# %% [markdown]
# # DiT Model Experiments and Visualization
# Experiment with different parameters and visualize the results.

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset (use your existing dataloader code)
# dataset = LandscapeDataset(...)
# dataloader = DataLoader(...)

# Initialize model
model = DiT(
    img_size=128,
    patch_size=4,
    in_channels=3,
    hidden_size=256,
    depth=6,
    num_heads=4
).to(device)

# Initialize noise scheduler and move to device
noise_scheduler = NoiseScheduler(num_timesteps=1000, schedule_type='linear').to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# # Training (one epoch example)
# avg_loss, losses = train_one_epoch(model, dataloader, optimizer, noise_scheduler, device)
# visualize_loss(losses)

# Training for 5 epochs
epoch_losses, all_losses = train_n_epochs(model, dataloader, optimizer, noise_scheduler, device, num_epochs=5)
visualize_loss(all_losses, save_path='all_batches_loss.png')
visualize_epoch_loss(epoch_losses, save_path='epoch_loss.png')
print(f"\nFinal average loss: {epoch_losses[-1]:.4f}")

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# %% [markdown]
# # Sampling Methods

# %%
from torchvision.utils import make_grid
from scipy import linalg
from torchvision.models import inception_v3


@torch.no_grad()
def ddpm_sample(model, noise_scheduler, batch_size=1, img_size=128, 
                channels=3, device='cuda', save_intermediates=False):
    """
    DDPM sampling process (reverse diffusion).
    """
    model.eval()
    
    # Start from pure noise
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    intermediates = [] if save_intermediates else None
    
    # Reverse diffusion process
    for t in tqdm(reversed(range(noise_scheduler.num_timesteps)), desc="Sampling"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get alpha values
        alpha_t = noise_scheduler.alphas[t]
        alpha_t_cumprod = noise_scheduler.alphas_cumprod[t]
        alpha_t_cumprod_prev = noise_scheduler.alphas_cumprod_prev[t]
        
        # Predict x_0 from x_t and noise
        pred_x0 = (x - torch.sqrt(1 - alpha_t_cumprod) * predicted_noise) / torch.sqrt(alpha_t_cumprod)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clip for stability
        
        # Calculate posterior mean using the coefficients
        # q(x_{t-1} | x_t, x_0)
        posterior_mean = (
            noise_scheduler.posterior_mean_coef1[t] * pred_x0 +
            noise_scheduler.posterior_mean_coef2[t] * x
        )
        
        if t > 0:
            noise = torch.randn_like(x)
            posterior_variance = noise_scheduler.posterior_variance[t]
            x = posterior_mean + torch.sqrt(posterior_variance) * noise
        else:
            x = posterior_mean  # No noise at t=0
        
        # Save intermediate predictions
        if save_intermediates and t % (noise_scheduler.num_timesteps // 10) == 0:
            intermediates.append(pred_x0.cpu())
    
    if save_intermediates:
        return x, intermediates
    return x

@torch.no_grad()
def ddim_sample(model, noise_scheduler, batch_size=1, img_size=128, 
                channels=3, device='cuda', num_inference_steps=50, 
                eta=0.0, save_intermediates=False):
    """
    DDIM sampling process (deterministic, faster sampling).
    """
    model.eval()
    
    # Create timestep schedule
    step_size = noise_scheduler.num_timesteps // num_inference_steps
    timesteps = list(range(0, noise_scheduler.num_timesteps, step_size))
    timesteps.reverse()
    
    # Start from pure noise
    x = torch.randn(batch_size, channels, img_size, img_size, device=device)
    intermediates = [] if save_intermediates else None
    
    for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling")):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x, t_batch)
        
        # Get alpha values
        alpha_t_cumprod = noise_scheduler.alphas_cumprod[t]
        
        # Get alpha for previous timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_t_cumprod_prev = noise_scheduler.alphas_cumprod[t_prev]
        else:
            alpha_t_cumprod_prev = torch.tensor(1.0, device=device)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_t_cumprod) * predicted_noise) / torch.sqrt(alpha_t_cumprod)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # DDIM formula with eta parameter
        sigma_t = eta * torch.sqrt((1 - alpha_t_cumprod_prev) / (1 - alpha_t_cumprod)) * \
                  torch.sqrt(1 - alpha_t_cumprod / alpha_t_cumprod_prev)
        
        # Direction pointing to x_t
        direction = torch.sqrt(1 - alpha_t_cumprod_prev - sigma_t**2) * predicted_noise
        
        # Compute x_{t-1}
        x = torch.sqrt(alpha_t_cumprod_prev) * pred_x0 + direction
        
        # Add stochastic noise if eta > 0
        if eta > 0 and t > 0:
            noise = torch.randn_like(x)
            x = x + sigma_t * noise
        
        # Save intermediate predictions
        if save_intermediates:
            intermediates.append(pred_x0.cpu())
    
    if save_intermediates:
        return x, intermediates
    return x

# %% [markdown]
# # Visualising Samples and Diffusion Process

# %%
def visualize_samples(images, nrow=10, title="Generated Samples", save_path=None):
    """
    Visualize a grid of images (10x10 grid for 100 images).
    
    Args:
        images: Tensor of images [B, C, H, W] in range [-1, 1]
        nrow: Number of images per row
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
    
    # Plot
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()


def visualize_diffusion_process(intermediates, final_image, save_path=None):
    """
    Visualize the evolution of clean image predictions during sampling.
    Shows how the model's prediction improves over timesteps.
    """
    num_steps = len(intermediates)
    fig, axes = plt.subplots(1, num_steps + 1, figsize=(3 * (num_steps + 1), 3))
    
    # Show intermediate predictions
    for i, img in enumerate(intermediates):
        img = (img[0] + 1) / 2  # Denormalize first image in batch
        img = torch.clamp(img, 0, 1)
        axes[i].imshow(img.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Step {i}")
        axes[i].axis('off')
    
    # Show final image
    final = (final_image[0] + 1) / 2
    final = torch.clamp(final, 0, 1)
    axes[-1].imshow(final.permute(1, 2, 0).cpu().numpy())
    axes[-1].set_title("Final")
    axes[-1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()


# %%
# Generate and visualize samples from the trained model

# Generate a single sample with DDPM (slower but higher quality)
print("Generating sample with DDPM...")
sample_ddpm = ddpm_sample(
    model=model,
    noise_scheduler=noise_scheduler,
    batch_size=1,
    img_size=128,
    channels=3,
    device=device,
    save_intermediates=False
)

# Visualize the sample
plt.figure(figsize=(6, 6))
sample_img = (sample_ddpm[0] + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
sample_img = torch.clamp(sample_img, 0, 1)
plt.imshow(sample_img.permute(1, 2, 0).cpu().numpy())
plt.title("Generated Sample (DDPM)")
plt.axis('off')
plt.show()

# Generate multiple samples to see variety
print("\nGenerating 16 samples...")
samples = ddpm_sample(
    model=model,
    noise_scheduler=noise_scheduler,
    batch_size=16,
    img_size=128,
    channels=3,
    device=device,
    save_intermediates=False
)

# Visualize grid of samples
visualize_samples(samples, nrow=4, title="Generated Samples (16 images)")


# %%
# Visualize the diffusion process step-by-step
print("Generating sample with intermediate steps...")
sample_with_steps, intermediates = ddpm_sample(
    model=model,
    noise_scheduler=noise_scheduler,
    batch_size=1,
    img_size=128,
    channels=3,
    device=device,
    save_intermediates=True
)

# Show how the image evolves during sampling
visualize_diffusion_process(intermediates, sample_with_steps, save_path='diffusion_process.png')
print(f"Saved diffusion process visualization with {len(intermediates)} intermediate steps")


# %%
# Try faster DDIM sampling (fewer steps)
print("Generating samples with DDIM (faster)...")
samples_ddim = ddim_sample(
    model=model,
    noise_scheduler=noise_scheduler,
    batch_size=16,
    img_size=128,
    channels=3,
    device=device,
    num_inference_steps=50,  # Much faster than 1000 steps!
    eta=0.0  # 0 = deterministic, >0 adds randomness
)

visualize_samples(samples_ddim, nrow=4, title="Generated Samples (DDIM - 50 steps)")


# %% [markdown]
# # FID Computation

# %%
class InceptionFeatureExtractor:
    """Extract features from Inception v3 for FID calculation."""
    
    def __init__(self, device='cuda'):
        self.device = device
        # Load pretrained Inception v3
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Remove final classification layer
        self.model.eval()
        self.model.to(device)
        
        # Preprocessing for Inception
        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def get_features(self, images):
        """
        Extract features from images.
        
        Args:
            images: Tensor [B, C, H, W] in range [-1, 1]
        Returns:
            Features [B, 2048]
        """
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        # Preprocess
        images = self.preprocess(images)
        
        # Extract features
        features = self.model(images)
        return features.cpu().numpy()


def calculate_fid(real_features, generated_features):
    """
    Calculate Fréchet Inception Distance between real and generated images.
    
    Args:
        real_features: Features from real images [N, D]
        generated_features: Features from generated images [M, D]
    
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


@torch.no_grad()
def compute_fid_score(model, noise_scheduler, real_dataloader, 
                      num_samples=1000, batch_size=32, device='cuda'):
    """
    Compute FID score for generated samples vs real images.
    
    Args:
        model: Trained DiT model
        noise_scheduler: NoiseScheduler object
        real_dataloader: DataLoader for real images
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
    
    Returns:
        FID score
    """
    feature_extractor = InceptionFeatureExtractor(device)
    
    # Extract features from real images
    print("Extracting features from real images...")
    real_features = []
    for images, _ in tqdm(real_dataloader):
        if len(real_features) * images.shape[0] >= num_samples:
            break
        images = images.to(device)
        features = feature_extractor.get_features(images)
        real_features.append(features)
    real_features = np.concatenate(real_features, axis=0)[:num_samples]
    
    # Generate samples and extract features
    print("Generating samples and extracting features...")
    generated_features = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for _ in tqdm(range(num_batches)):
        # Generate samples
        samples = ddpm_sample(
            model, noise_scheduler, 
            batch_size=min(batch_size, num_samples - len(generated_features) * batch_size),
            device=device
        )
        
        # Extract features
        features = feature_extractor.get_features(samples)
        generated_features.append(features)
    
    generated_features = np.concatenate(generated_features, axis=0)[:num_samples]
    
    # Calculate FID
    fid_score = calculate_fid(real_features, generated_features)
    return fid_score

# %% [markdown]
# # Diffusion Timesteps Experimentation
# Experiment with different numbers of diffusion timesteps (e.g., T=100, 500, 1000).

# %%
# Placeholder for diffusion timesteps experimentation

# %% [markdown]
# # DiT Blocks Experimentation
# Experiment with different numbers of DiT blocks (e.g., 4, 6, 8) and observe the impact.

# %%
# Placeholder for DiT blocks experimentation

# %% [markdown]
# # Attention Heads Experimentation
# Experiment with different numbers of attention heads (e.g., 1, 2, 4) and observe the impact.

# %%
# Placeholder for attention heads experimentation

# %% [markdown]
# # Pre-trained DiT-based Diffusion Model Visualization
# Visualize the evolving predictions of a pre-trained DiT-based diffusion model at various timesteps.

# %%
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from PIL import Image
import numpy as np
from typing import List, Optional

@torch.no_grad()
def visualize_diffusion_evolution_pretrained(
    num_inference_steps=50,
    batch_size=100,
    prompt="a beautiful landscape with mountains and rivers",
    seed=42,
    device='cuda'
):
    """
    Visualize the evolving prediction of clean images at various timesteps
    during the reverse diffusion process using a pre-trained model.
    
    This creates a 10x10 grid (100 images) showing how the model's prediction
    of the final clean image evolves over time.
    
    Args:
        num_inference_steps: Number of denoising steps
        batch_size: Number of images to generate (should be 100 for 10x10 grid)
        prompt: Text prompt for generation
        seed: Random seed for reproducibility
        device: Device to use
    
    Returns:
        List of intermediate predictions at each timestep
    """
    print("Loading pre-trained Stable Diffusion model...")
    
    # Load Stable Diffusion with DDIM scheduler
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,  # Disable for faster loading
        requires_safety_checker=False
    )
    
    # Use DDIM scheduler as specified in the assignment
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get the model components
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    
    # Encode the prompt
    text_embeddings = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    # Create initial latent noise
    latent_shape = (1, unet.config.in_channels, 64, 64)  # SD uses 64x64 latents
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=text_embeddings.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    # Store predictions at each timestep
    all_predictions = []
    timestep_labels = []
    
    print(f"\nGenerating images with {num_inference_steps} steps...")
    
    # Reverse diffusion process
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Expand latents for classifier-free guidance (even though we're not using it here)
        latent_model_input = latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute the previous noisy sample x_t -> x_t-1
        scheduler_output = scheduler.step(noise_pred, t, latents)
        latents = scheduler_output.prev_sample
        
        # Get the predicted original sample (x_0 prediction)
        # This is what we want to visualize - the model's current estimate of the final image
        pred_original_sample = scheduler_output.pred_original_sample
        
        # Decode latent to image at key timesteps
        # To avoid too many images, we'll sample at regular intervals
        if i % max(1, len(timesteps) // 10) == 0 or i == len(timesteps) - 1:
            # Decode the predicted original sample
            with torch.no_grad():
                image = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
            
            # Convert to PIL image
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            image = Image.fromarray(image[0])
            
            all_predictions.append(image)
            timestep_labels.append(f"t={t.item()}")
            
            print(f"Captured prediction at timestep {t.item()}")
    
    print(f"\nCaptured {len(all_predictions)} intermediate predictions")
    
    # Generate 100 images at different timesteps for the 10x10 grid
    print("\nGenerating 100 images for 10x10 grid visualization...")
    grid_images = []
    
    # Sample 100 timesteps evenly across the diffusion process
    sampled_timesteps = np.linspace(0, len(timesteps)-1, 100, dtype=int)
    
    for idx in tqdm(sampled_timesteps, desc="Generating grid images"):
        # Reset to initial noise
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=text_embeddings.dtype)
        latents = latents * scheduler.init_noise_sigma
        
        # Run diffusion up to this timestep
        for i, t in enumerate(timesteps):
            if i > idx:
                break
                
            latent_model_input = scheduler.scale_model_input(latents, t)
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            scheduler_output = scheduler.step(noise_pred, t, latents)
            latents = scheduler_output.prev_sample
            
            # Get prediction at this exact timestep
            if i == idx:
                pred_original_sample = scheduler_output.pred_original_sample
                
                # Decode to image
                with torch.no_grad():
                    image = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
                
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                image = (image * 255).round().astype("uint8")
                image = Image.fromarray(image[0])
                grid_images.append(image)
                break
    
    return all_predictions, timestep_labels, grid_images

# %%
predictions, labels, grid_images = visualize_diffusion_evolution_pretrained(
    num_inference_steps=50,
    batch_size=100,
    prompt="a beautiful landscape with mountains, lakes and forests",
    seed=42,
    device=device
)

# Visualize the evolution of predictions
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, (img, label) in enumerate(zip(predictions, labels)):
    if i < 10:
        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=12)
        axes[i].axis('off')

plt.suptitle("Evolution of Model's Prediction of Clean Image During Reverse Diffusion", fontsize=16)
plt.tight_layout()
plt.savefig('pretrained_diffusion_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved evolution visualization showing {len(predictions)} timesteps")

# %%

# %%
# Create 10x10 grid of predictions at different timesteps
print("\nCreating 10x10 grid of predictions...")

# Create the grid
fig, axes = plt.subplots(10, 10, figsize=(25, 25))

for i in range(10):
    for j in range(10):
        idx = i * 10 + j
        if idx < len(grid_images):
            axes[i, j].imshow(grid_images[idx])
            # Add timestep label to first row
            if i == 0:
                timestep = int((idx / 100) * 50)  # Approximate timestep
                axes[i, j].set_title(f"Step {timestep}", fontsize=8)
        axes[i, j].axis('off')

plt.suptitle("100 Predictions at Different Timesteps (10x10 Grid)\nProgression from Noise to Final Image", 
             fontsize=20, y=0.995)
plt.tight_layout()
plt.savefig('pretrained_10x10_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved 10x10 grid visualization")

# %% [markdown]
# # Classifier-Free Guidance (CFG) Logic and Analysis
# Implement CFG logic and perform sensitivity analysis for the guidance scale.

# %% [markdown]
# # CFG Logic Implementation
# Implement logic for combining conditional and unconditional predictions.

# %%
def classifier_free_guidance(conditional_pred, unconditional_pred, guidance_scale):
    """
    Combine conditional and unconditional predictions using CFG.
    
    The CFG formula is:
        noise_pred = unconditional_pred + guidance_scale * (conditional_pred - unconditional_pred)
    
    Args:
        conditional_pred: Noise prediction with conditioning (e.g., with text prompt) [B, C, H, W]
        unconditional_pred: Noise prediction without conditioning (null/empty prompt) [B, C, H, W]
        guidance_scale: Guidance scale w (float)
            - w = 0: Use only unconditional (ignores prompt)
            - w = 1: Equal weighting
            - w > 1: Amplify conditional signal (stronger prompt adherence)
    
    Returns:
        Final noise prediction [B, C, H, W]
    """
    # CFG formula
    noise_pred = unconditional_pred + guidance_scale * (conditional_pred - unconditional_pred)
    
    return noise_pred


# %%
batch_size = 1
conditional_example = torch.randn(batch_size, 3, 4, 4)
unconditional_example = torch.randn(batch_size, 3, 4, 4)

print("\nExample predictions:")
print(f"Conditional prediction mean: {conditional_example.mean():.4f}")
print(f"Unconditional prediction mean: {unconditional_example.mean():.4f}")

print("\nCFG with different guidance scales:")
for w in [0, 1, 2, 5, 7.5, 10]:
    result = classifier_free_guidance(conditional_example, unconditional_example, w)
    print(f"  w={w:4.1f}: Result mean = {result.mean():.4f}")

print("\nObservations:")
print("- w=0: Result equals unconditional (prompt ignored)")
print("- w=1: Simple average of conditional and unconditional")
print("- w>1: Amplifies the difference, strengthening prompt adherence")

# %%
@torch.no_grad()
def generate_with_cfg_pretrained(
    prompt="a beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    device='cuda'
):
    """
    Generate images using Classifier-Free Guidance with pre-trained model.
    
    Args:
        prompt: Text prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG guidance scale (w)
        seed: Random seed
        device: Device to use
    
    Returns:
        Generated image
    """
    print(f"Generating with guidance scale w={guidance_scale}...")
    
    # Load model
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate image with specified guidance scale
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    return image



# %% [markdown]
# # Sensitivity Analysis for Guidance Scale
# Analyze the effects of varying the guidance scale (e.g., w from 0-10) on sample quality and diversity.

# %%
def cfg_sensitivity_analysis(
    prompt="a beautiful mountain landscape with lakes and forests",
    guidance_scales=[0, 1, 2, 5, 7.5, 10],
    num_inference_steps=50,
    seeds=[42, 123, 456],  # Multiple seeds for diversity analysis
    device='cuda'
):
    """
    Perform sensitivity analysis for different guidance scales.
    
    Args:
        prompt: Text prompt
        guidance_scales: List of guidance scales to test
        num_inference_steps: Number of denoising steps
        seeds: List of random seeds (for diversity analysis)
        device: Device to use
    
    Returns:
        Dictionary mapping guidance_scale -> list of images
    """
    results = {}
    
    print("="*60)
    print("CFG Sensitivity Analysis")
    print("="*60)
    print(f"Prompt: '{prompt}'")
    print(f"Testing guidance scales: {guidance_scales}")
    print(f"Number of samples per scale: {len(seeds)}")
    print("="*60)
    
    for w in guidance_scales:
        print(f"\nGenerating samples with guidance scale w={w}...")
        images = []
        
        for seed_idx, seed in enumerate(seeds):
            image = generate_with_cfg_pretrained(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=w,
                seed=seed,
                device=device
            )
            images.append(image)
            print(f"  Generated sample {seed_idx + 1}/{len(seeds)}")
        
        results[w] = images
    
    return results


# %%
cfg_results = cfg_sensitivity_analysis(
    prompt="a beautiful landscape with mountains, rivers, and sunset",
    guidance_scales=[0, 1.0, 2.0, 5.0, 7.5, 10.0],
    num_inference_steps=50,
    seeds=[42, 123, 456],
    device=device
)

print("\nCFG sensitivity analysis complete!")

# %%
guidance_scales = list(cfg_results.keys())
num_scales = len(guidance_scales)
num_samples = len(cfg_results[guidance_scales[0]])

fig, axes = plt.subplots(num_scales, num_samples, figsize=(num_samples * 4, num_scales * 4))

for i, w in enumerate(guidance_scales):
    for j in range(num_samples):
        if num_scales == 1:
            ax = axes[j]
        else:
            ax = axes[i, j]
        
        ax.imshow(cfg_results[w][j])
        if j == 0:
            ax.set_ylabel(f'w={w}', fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_title(f'Sample {j+1}', fontsize=12)
        ax.axis('off')

plt.suptitle('Classifier-Free Guidance Sensitivity Analysis\nEffect of Different Guidance Scales', 
             fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('cfg_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved CFG sensitivity visualization")

# %%
print("Generating final comparison grid...")

# Generate 4 images for each of 6 guidance scales (24 images total)
comparison_prompt = "a serene landscape with mountains and a lake at sunset"
comparison_scales = [0, 2.0, 5.0, 7.5, 10.0, 15.0]
comparison_seeds = [42, 100, 200, 300]

fig, axes = plt.subplots(len(comparison_scales), len(comparison_seeds), 
                         figsize=(len(comparison_seeds) * 3, len(comparison_scales) * 3))

for i, w in enumerate(tqdm(comparison_scales, desc="Generating comparison images")):
    for j, seed in enumerate(comparison_seeds):
        image = generate_with_cfg_pretrained(
            prompt=comparison_prompt,
            guidance_scale=w,
            seed=seed,
            device=device
        )
        
        axes[i, j].imshow(image)
        if j == 0:
            axes[i, j].set_ylabel(f'w={w}', fontsize=12, fontweight='bold')
        if i == 0:
            axes[i, j].set_title(f'Seed {seed}', fontsize=10)
        axes[i, j].axis('off')

plt.suptitle('CFG Guidance Scale Comparison\nPrompt: "' + comparison_prompt + '"', 
             fontsize=14, y=0.998)
plt.tight_layout()
plt.savefig('cfg_comparison_grid.png', dpi=150, bbox_inches='tight')
plt.show()

print("Saved comparison grid for report")




