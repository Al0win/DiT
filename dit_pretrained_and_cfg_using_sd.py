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
    num_inference_steps=100,
    prompt="a beautiful landscape with mountains and rivers",
    seed=42,
    device='cuda'
):
    """
    Visualize the evolving prediction of clean images at various timesteps
    during the reverse diffusion process using a pre-trained model.
    
    This captures the x_0 prediction at EACH step during a SINGLE denoising run.
    """
    print("Loading pre-trained Stable Diffusion model...")
    
    # Load Stable Diffusion with DDIM scheduler
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Get model components
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    # Encode prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    # Create initial latent noise ONCE
    latent_shape = (1, unet.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=text_embeddings.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    print(f"\nGenerating single image evolution with {num_inference_steps} steps...")
    
    # Storage for intermediate predictions
    grid_images = []
    all_predictions = []
    timestep_labels = []
    
    # Single continuous denoising loop
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        # Predict noise
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # Compute the previous noisy sample
        scheduler_output = scheduler.step(noise_pred, t, latents)
        
        # CAPTURE x_0 prediction BEFORE updating latents
        pred_original_sample = scheduler_output.pred_original_sample
        
        # Decode and save this x_0 prediction
        with torch.no_grad():
            image = vae.decode(pred_original_sample / vae.config.scaling_factor).sample
        
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
    device='cuda'
):
    """
    Generate an image with a MANUAL and CORRECT implementation of Classifier-Free Guidance.
    This function explicitly separates conditional and unconditional paths to fix the w=0 bug.
    """
    print(f"\nGenerating with w={guidance_scale} (Manual CFG)...", end=" ")
    
    # --- 1. Setup Model and Scheduler ---
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # --- 2. Get Model Components ---
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    # --- 3. Prepare Text Embeddings (The CRITICAL Part) ---
    # Conditional embedding (for the prompt)
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    cond_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
    
    # Unconditional embedding (for an empty prompt)
    uncond_inputs = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]
    
    # Concatenate for batched inference
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    
    # --- 4. Prepare Latents and Timesteps ---
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    
    latent_shape = (1, unet.config.in_channels, 64, 64)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=cond_embeddings.dtype)
    latents = latents * scheduler.init_noise_sigma
    
    # --- 5. Denoising Loop with Manual CFG ---
    for t in tqdm(timesteps, desc="Manual CFG Sampling", leave=False):
        # Expand latents for batched inference (uncond + cond)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise for both unconditional and conditional
        noise_pred_batch = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # Separate the predictions
        noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)
        
        # --- THIS IS THE CFG FORMULA ---
        # When w=0, this correctly becomes `noise_pred = noise_pred_uncond`
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Compute previous latent state
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # --- 6. Decode Final Latent to Image ---
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
    
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
        device=device
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
        device=device
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


