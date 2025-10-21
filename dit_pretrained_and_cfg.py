# %% [markdown]
# # Pre-trained DiT-based Diffusion Model Visualization
# Visualize the evolving predictions of a pre-trained DiT-based diffusion model at various timesteps.
# 
# This implementation uses DiT (Diffusion Transformer) architecture instead of UNet-based models.
# Default model: facebook/DiT-XL-2-256 (DiT-XL/2 trained on ImageNet at 256x256 resolution)
# Alternative: black-forest-labs/FLUX.1-dev (more advanced DiT-based model)
#
# Key differences from UNet-based Stable Diffusion:
# - Uses Transformer blocks instead of convolutional UNet
# - DiT architecture as described in "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)

# %%
from diffusers import DiffusionPipeline, DDIMScheduler
import torch
from PIL import Image
import numpy as np
from typing import List, Optional
import os

# Set cache directory to /mnt/local/hf to avoid filling up /home/jovyan
CACHE_DIR = "/mnt/local/hf/huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Set environment variables for HuggingFace cache (MUST be set before loading models)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR

# Model selection - choose one:
# - "facebook/DiT-XL-2-256": Original DiT model (256x256, ImageNet classes) - REQUIRES class labels, not text
# - "black-forest-labs/FLUX.1-dev": Advanced DiT-based text-to-image model - REQUIRES authentication
# - "PixArt-alpha/PixArt-XL-2-1024-MS": DiT-based text-to-image (1024x1024) - NO AUTH REQUIRED
# - "PixArt-alpha/PixArt-XL-2-512x512": DiT-based text-to-image (512x512) - NO AUTH REQUIRED, faster
DEFAULT_MODEL = "PixArt-alpha/PixArt-XL-2-512x512"

@torch.no_grad()
def visualize_diffusion_evolution_pretrained(
    num_inference_steps=100,
    prompt="a beautiful landscape with mountains and rivers",
    seed=42,
    device='cuda',
    model_id="PixArt-alpha/PixArt-XL-2-512x512"
):
    """
    Visualize the evolving prediction of clean images at various timesteps
    during the reverse diffusion process using a pre-trained DiT-based model.
    
    This captures the x_0 prediction at EACH step during a SINGLE denoising run.
    Uses PixArt-alpha which has a TRANSFORMER architecture (DiT), not UNet.
    """
    print(f"Loading pre-trained DiT-based model: {model_id}...")
    
    # Load DiT-based model
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        cache_dir=CACHE_DIR
    )
    
    # IMPORTANT: Replace scheduler with DDIM for consistency
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    print(f"  Using DDIM scheduler (replaced {type(pipe.scheduler).__name__})")
    
    pipe = pipe.to(device)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get model components - NOTE: DiT uses 'transformer' not 'unet'
    transformer = pipe.transformer  # DiT transformer instead of UNet
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    # Encode prompt using T5 tokenizer
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=120,  # PixArt uses max_length=120
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    with torch.no_grad():
        # T5 encoder returns encoder outputs
        prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0]
        prompt_attention_mask = attention_mask
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Create initial latent noise ONCE
    # PixArt uses different latent dimensions than SD
    height = width = 512 if "512" in model_id else 1024
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = transformer.config.in_channels if hasattr(transformer.config, 'in_channels') else 4
    
    latent_shape = (1, latent_channels, latent_height, latent_width)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    print(f"\nGenerating single image evolution with {num_inference_steps} steps...")
    print(f"  Latent shape: {latent_shape}")
    print(f"  Using DiT transformer: {type(transformer).__name__}")
    
    # Storage for intermediate predictions
    grid_images = []
    all_predictions = []
    timestep_labels = []
    
    # Single continuous denoising loop
    from tqdm import tqdm
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # Expand latents if needed
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        # Ensure timestep is properly formatted for transformer (needs to be a tensor for batch)
        current_timestep = torch.tensor([t], device=device)
        
        # Predict noise using TRANSFORMER (not UNet)
        # PixArt transformer expects: sample, timestep, encoder_hidden_states, encoder_attention_mask
        noise_pred = transformer(
            latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False
        )[0]
        
        # PixArt transformer outputs 8 channels: [noise (4), variance (4)]
        # We only need the first 4 channels (the noise prediction)
        if noise_pred.shape[1] == 8:
            noise_pred = noise_pred[:, :4, :, :]
        
        # Compute the previous noisy sample
        scheduler_output = scheduler.step(noise_pred, t, latents, return_dict=True)
        
        # CAPTURE x_0 prediction BEFORE updating latents
        if hasattr(scheduler_output, 'pred_original_sample') and scheduler_output.pred_original_sample is not None:
            pred_original_sample = scheduler_output.pred_original_sample
        else:
            # Some schedulers don't return pred_original_sample, compute it manually
            # For DDPM/DPMSolver: x_0 = (x_t - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        
        # Decode and save this x_0 prediction
        with torch.no_grad():
            # VAE expects latents to be scaled
            scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.18215
            image = vae.decode(pred_original_sample / scaling_factor, return_dict=False)[0]
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        # Save for 10x10 grid (all 100 steps)
        grid_images.append(image)
        
        # Save for evolution visualization (10 key steps)
        if i % max(1, len(timesteps) // 10) == 0 or i == len(timesteps) - 1:
            all_predictions.append(image)
            timestep_labels.append(f"t={t.item()}")
        
        # NOW update latents for next iteration
        latents = scheduler_output.prev_sample
    
    print(f"\nCaptured {len(grid_images)} x_0 predictions from single denoising run")
    
    return all_predictions, timestep_labels, grid_images

# %%
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda:1"

predictions, labels, grid_images = visualize_diffusion_evolution_pretrained(
    num_inference_steps=100,
    prompt="a beautiful landscape with mountains, lakes and forests",
    seed=42,
    device=device,
    model_id=DEFAULT_MODEL
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
# Create 10x10 grid showing SINGLE image evolution from noise to clean
print("\nCreating 10x10 grid of single image evolution...")

fig, axes = plt.subplots(10, 10, figsize=(30, 30))

num_inference_steps = 100
prompt = "a beautiful landscape with mountains, lakes and forests"

for i in range(10):
    for j in range(10):
        idx = i * 10 + j
        if idx < len(grid_images):
            axes[i, j].imshow(grid_images[idx])
            # Calculate actual timestep number
            timestep_num = int((idx / 99) * (num_inference_steps - 1))
            axes[i, j].set_title(f"Step {timestep_num}", fontsize=10, pad=3)
            
            # Add border to final image
            if idx == len(grid_images) - 1:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        axes[i, j].axis('off')

plt.suptitle("Evolution of Single Image: Noise → Clean (100 Steps in 10x10 Grid)\n" + 
             f"Prompt: '{prompt}' | DDIM Sampler with {num_inference_steps} inference steps", 
             fontsize=24, y=0.995, fontweight='bold')

# Add annotations
fig.text(0.01, 0.5, '', fontsize=20, rotation=90, 
         verticalalignment='center', fontweight='bold', color='blue')
fig.text(0.5, 0.01, '', 
         fontsize=18, horizontalalignment='center', fontweight='bold', color='blue')
fig.text(0.5, 0.98, '', 
         fontsize=14, horizontalalignment='center', color='red')

plt.tight_layout()
plt.savefig('single_image_evolution_10x10.png', dpi=200, bbox_inches='tight')
plt.show()

print("✓ Saved 10x10 grid showing SINGLE image evolution from noise to clean")

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

@torch.no_grad()
def generate_with_cfg(
    prompt="a beautiful landscape",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    device='cuda',
    model_id="PixArt-alpha/PixArt-XL-2-512x512"
):
    """
    Generate an image with a MANUAL and CORRECT implementation of Classifier-Free Guidance.
    This function explicitly separates conditional and unconditional paths using a DiT-based model.
    Uses PixArt-alpha's TRANSFORMER architecture (DiT), not UNet.
    """
    print(f"\nGenerating with w={guidance_scale} (Manual CFG)...", end=" ")
    
    # --- 1. Setup Model and Scheduler ---
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        cache_dir=CACHE_DIR
    )
    
    # CRITICAL: Replace with DDIM scheduler as required
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    pipe = pipe.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # --- 2. Get Model Components (DiT uses TRANSFORMER not UNet) ---
    transformer = pipe.transformer  # DiT transformer instead of UNet
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    # --- 3. Prepare Text Embeddings (The CRITICAL Part) ---
    # Conditional embedding (for the prompt)
    text_inputs = tokenizer(prompt, padding="max_length", max_length=120, truncation=True, return_tensors="pt")
    cond_input_ids = text_inputs.input_ids.to(device)
    cond_attention_mask = text_inputs.attention_mask.to(device)
    cond_embeddings = text_encoder(cond_input_ids, attention_mask=cond_attention_mask)[0]
    
    # Unconditional embedding (for an empty prompt)
    uncond_inputs = tokenizer("", padding="max_length", max_length=120, return_tensors="pt")
    uncond_input_ids = uncond_inputs.input_ids.to(device)
    uncond_attention_mask = uncond_inputs.attention_mask.to(device)
    uncond_embeddings = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0]
    
    # Concatenate for batched inference
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    attention_masks = torch.cat([uncond_attention_mask, cond_attention_mask])
    
    # --- 4. Prepare Latents and Timesteps ---
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # PixArt uses different latent dimensions
    height = width = 512 if "512" in model_id else 1024
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = transformer.config.in_channels if hasattr(transformer.config, 'in_channels') else 4
    
    latent_shape = (1, latent_channels, latent_height, latent_width)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=cond_embeddings.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    # --- 5. Denoising Loop with Manual CFG ---
    from tqdm import tqdm
    for t in tqdm(timesteps, desc="Manual CFG Sampling", leave=False):
        # Expand latents for batched inference (uncond + cond)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Ensure timestep is properly formatted (needs to be expanded for batch of 2)
        current_timestep = torch.tensor([t] * latent_model_input.shape[0], device=device)
        
        # Predict noise for both unconditional and conditional using TRANSFORMER
        noise_pred_batch = transformer(
            latent_model_input,
            timestep=current_timestep,
            encoder_hidden_states=text_embeddings,
            encoder_attention_mask=attention_masks,
            return_dict=False
        )[0]
        
        # PixArt transformer outputs 8 channels: [noise (4), variance (4)]
        # We only need the first 4 channels (the noise prediction)
        if noise_pred_batch.shape[1] == 8:
            noise_pred_batch = noise_pred_batch[:, :4, :, :]
        
        # Separate the predictions
        noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)
        
        # --- THIS IS THE CFG FORMULA ---
        # When w=0, this correctly becomes `noise_pred = noise_pred_uncond`
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Compute previous latent state
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    # --- 6. Decode Final Latent to Image ---
    with torch.no_grad():
        scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.18215
        image = vae.decode(latents / scaling_factor, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    
    print("✓")
    return image
 

# %%
# Test manual CFG with different guidance scales
print("Testing Manual CFG Implementation")
print("="*60)

test_prompt = "a beautiful mountain landscape with a lake at sunset"
test_scales = [1.0, 5.0, 7.5, 10.0]

fig, axes = plt.subplots(1, len(test_scales), figsize=(20, 5))

for idx, w in enumerate(test_scales):
    image = generate_with_cfg(
        prompt=test_prompt,
        guidance_scale=w,
        num_inference_steps=30,  # Fewer steps for faster testing
        seed=42,
        device=device,
        model_id=DEFAULT_MODEL
    )
    axes[idx].imshow(image)
    axes[idx].set_title(f'CFG w={w}', fontsize=14)
    axes[idx].axis('off')

plt.suptitle(f'Manual CFG Implementation\nPrompt: "{test_prompt}"', fontsize=16)
plt.tight_layout()
plt.savefig('manual_cfg_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved manual CFG comparison")

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
            image = generate_with_cfg(
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
print("\n" + "="*80)
print("CLASSIFIER-FREE GUIDANCE (CFG) ANALYSIS")
print("="*80)

# Part 1: Show the explicit CFG formula with visual representation
print("\n1. CFG Formula Demonstration:")
print("-" * 80)
print("   noise_pred_final = noise_pred_uncond + w × (noise_pred_cond - noise_pred_uncond)")
print("   where:")
print("     - noise_pred_cond: prediction WITH prompt")
print("     - noise_pred_uncond: prediction WITHOUT prompt (empty string)")
print("     - w (guidance scale): controls strength of prompt adherence")
print("-" * 80)

# Part 2: Generate comparison across different w values
comparison_prompt = "a serene mountain landscape with a crystal clear lake at sunset"
guidance_scales = [0, 1.0, 3.0, 5.0, 7.5, 10.0]
seed = 42

print(f"\n2. Generating samples with different guidance scales...")
print(f"   Prompt: '{comparison_prompt}'")
print(f"   Seed: {seed} (same for all - to isolate effect of w)")
print(f"   Guidance scales to test: {guidance_scales}\n")

cfg_comparison_images = {}

for w in guidance_scales:
    print(f"   Generating with w={w}...", end=" ")
    image = generate_with_cfg(
        prompt=comparison_prompt,
        guidance_scale=w,
        seed=seed,
        num_inference_steps=50,
        device=device,
        model_id=DEFAULT_MODEL
    )
    cfg_comparison_images[w] = image
    print("✓")

# Visualize with detailed labels
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, w in enumerate(guidance_scales):
    axes[idx].imshow(cfg_comparison_images[w])
    
    # Create detailed title based on guidance scale
    if w == 0:
        title = f'w = {w}\n(Unconditional)\nPrompt IGNORED'
        color = 'red'
    elif w == 1.0:
        title = f'w = {w}\n(Balanced)\nEqual weighting'
        color = 'orange'
    elif w <= 5.0:
        title = f'w = {w}\n(Moderate Guidance)\nGood balance'
        color = 'green'
    elif w == 7.5:
        title = f'w = {w}\n(Strong Guidance)\nSD default, high quality'
        color = 'blue'
    else:
        title = f'w = {w}\n(Very Strong)\nMay over-saturate'
        color = 'purple'
    
    axes[idx].set_title(title, fontsize=14, fontweight='bold', color=color, pad=10)
    axes[idx].axis('off')
    
    # Add colored border
    for spine in axes[idx].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.suptitle(
    f'Classifier-Free Guidance (CFG) Sensitivity Analysis\n'
    f'Prompt: "{comparison_prompt}"\n'
    f'Formula: final = uncond + w × (cond - uncond)',
    fontsize=16, fontweight='bold', y=0.98
)

# Add explanation box
explanation = (
    "KEY OBSERVATIONS:\n"
    "• w=0: Ignores prompt completely (unconditional generation)\n"
    "• w=1: Baseline (conditional and unconditional equally weighted)\n"
    "• w=3-5: Moderate guidance (good quality-diversity trade-off)\n"
    "• w=7.5: Strong guidance (Stable Diffusion default, high prompt adherence)\n"
    "• w=10+: Very strong guidance (may reduce diversity, over-saturate colors)\n\n"
    "TRADE-OFF: Higher w → Better prompt adherence but less diversity"
)

fig.text(0.5, 0.02, explanation, fontsize=11, ha='center', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         family='monospace')

plt.tight_layout(rect=[0, 0.12, 1, 0.96])
plt.savefig('cfg_sensitivity_detailed.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n✓ Saved detailed CFG sensitivity analysis")


