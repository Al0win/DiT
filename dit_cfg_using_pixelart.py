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
from typing import List, Optional, Tuple, Dict, Any
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
    num_inference_steps: int = 100,
    prompt: str = "a beautiful landscape with mountains and rivers",
    seed: int = 42,
    device: str = "cuda",
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
    use_clamp: bool = False,
    clamp_value: float = 5.0,
    capture_every: int = None,             # if None, captures ~10 evenly spaced steps
) -> Tuple[List[Image.Image], List[str], Dict[str, Any]]:
    """
    Load a pretrained DiT-based model and capture x_0 predictions at each denoising step.
    Returns: (captured_images_list, timestep_labels, diagnostics)
    diagnostics includes per-step mean absolute norms for latents and noise predictions.
    """

    print(f"Loading model '{model_id}' (device={device})...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        cache_dir=CACHE_DIR,
    )
    # Replace scheduler with DDIM for deterministic sampling if desired
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Determine transformer dtype robustly
    transformer = pipe.transformer
    try:
        transformer_dtype = next(transformer.parameters()).dtype
    except StopIteration:
        transformer_dtype = torch.float16 if "cuda" in device else torch.float32

    vae = pipe.vae
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Prepare text embeddings (T5 encoder typical)
    text_inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=120, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)[0].to(device).to(transformer_dtype)
    prompt_attention_mask = attention_mask.to(device)

    # Timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Latent shape (PixArt-specific heuristic)
    height = width = 512 if "512" in model_id else 1024
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = getattr(transformer.config, "in_channels", 4)
    latent_shape = (1, latent_channels, latent_height, latent_width)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=transformer_dtype)
    latents = latents * scheduler.init_noise_sigma

    # diagnostics
    diagnostics: Dict[str, List[Any]] = {
        "timesteps": [],
        "noise_mean_abs": [],  # mean abs of predicted noise (per-step)
        "latents_mean_abs_before": [],  # mean abs of latents before step
        "latents_mean_abs_after": [],   # mean abs of latents after step
    }

    grid_images: List[Image.Image] = []
    captured_images: List[Image.Image] = []
    timestep_labels: List[str] = []

    # choose capture_every to get ~10 captures if not specified
    if capture_every is None:
        capture_every = max(1, len(timesteps) // 10)

    print("Running denoising and capturing x_0 predictions...")
    for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
        # scale latents for model input
        latent_model_input = scheduler.scale_model_input(latents, t)

        # ensure timestep tensor with correct dtype and device for transformer
        current_timestep = torch.full((latent_model_input.shape[0],), int(t), device=device, dtype=torch.long)

        # Predict noise with transformer (encoder_hidden_states=prompt_embeds)
        # Some transformers expect timestep shaped (batch,) or (batch,1) — this matches Diffusers practice
        noise_pred = transformer(
            latent_model_input.to(transformer_dtype),
            timestep=current_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False
        )[0]

        # If transformer outputs extra channels (variance), keep only first 4
        if noise_pred.shape[1] == 8:
            noise_pred = noise_pred[:, :4, :, :]

        # Diagnostics: record mean absolute values
        diagnostics["timesteps"].append(int(t))
        diagnostics["noise_mean_abs"].append(float(noise_pred.abs().mean().cpu().item()))
        diagnostics["latents_mean_abs_before"].append(float(latents.abs().mean().cpu().item()))

        # Optional clamp for numerical stability
        if use_clamp:
            noise_pred = noise_pred.clamp(-clamp_value, clamp_value)

        # Use scheduler.step with return_dict=True for robust named fields
        scheduler_output = scheduler.step(noise_pred, t, latents, return_dict=True)
        # prev_sample is the updated latent for next iteration
        prev_sample = getattr(scheduler_output, "prev_sample", None)
        if prev_sample is None:
            # fallback if scheduler returned tuple
            prev_sample = scheduler_output[0] if isinstance(scheduler_output, (tuple, list)) else None

        # Compute pred_original_sample if provided or compute fallback
        pred_original_sample = getattr(scheduler_output, "pred_original_sample", None)
        if pred_original_sample is None:
            # fallback compute using alphas_cumprod (be careful with indexing types)
            alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
            if alphas_cumprod is None:
                # if scheduler doesn't expose, skip capturing x_0
                pred_original_sample = None
            else:
                # ensure alphas_cumprod is tensor on device
                ac = torch.as_tensor(alphas_cumprod, device=device, dtype=transformer_dtype)
                alpha_prod_t = ac[int(t)]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

        # Decode and capture x_0 prediction if available
        if pred_original_sample is not None:
            with torch.no_grad():
                scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                # decode expects dtype matching VAE weights; cast to vae dtype if necessary
                vae_dtype = next(vae.parameters()).dtype
                decoded = vae.decode((pred_original_sample / scaling_factor).to(vae_dtype), return_dict=False)[0]
                img = (decoded / 2 + 0.5).clamp(0, 1)
                img = img.cpu().permute(0, 2, 3, 1).float().numpy()
                img = (img * 255).round().astype("uint8")
                pil = Image.fromarray(img[0])
                grid_images.append(pil)
                if (i % capture_every == 0) or (i == len(timesteps) - 1):
                    captured_images.append(pil)
                    timestep_labels.append(f"t={int(t)}")

        # update diagnostics after updating latents
        if prev_sample is not None:
            latents = prev_sample
            diagnostics["latents_mean_abs_after"].append(float(latents.abs().mean().cpu().item()))
        else:
            # if we couldn't obtain prev_sample, break to avoid infinite loop
            print("Warning: scheduler did not return prev_sample; breaking.")
            break

    print(f"Captured {len(captured_images)} x_0 predictions from denoising run.")
    out_diagnostics = {"prompt": prompt, "model_id": model_id, "diagnostics": diagnostics}
    return captured_images, timestep_labels, grid_images, out_diagnostics

# %%
import transformers
from tqdm import tqdm
import matplotlib.pyplot as plt

device = "cuda:1"

predictions, labels, grid_images, a = visualize_diffusion_evolution_pretrained(
    num_inference_steps=100,
    prompt="a beautiful landscape with mountains, lakes and forests",
    seed=42,
    device=device,
    model_id=DEFAULT_MODEL
)

print(f"Diagnostic: {a}")

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
    prompt: str = "a beautiful landscape",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    device: str = "cuda",
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
    use_clamp: bool = True,
    clamp_value: float = 5.0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Manual CFG implementation for DiT-based model.
    Returns (PIL Image, diagnostics dict).
    diagnostics contains per-step norms and final metadata.
    """

    print(f"Generating with CFG w={guidance_scale} using model '{model_id}' on {device}...")

    # Load pipeline & scheduler
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        cache_dir=CACHE_DIR,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Get transformer dtype
    try:
        transformer_dtype = next(transformer.parameters()).dtype
    except StopIteration:
        transformer_dtype = torch.float16 if "cuda" in device else torch.float32

    # Prepare conditional embeddings
    text_inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=120, return_tensors="pt")
    cond_input_ids = text_inputs.input_ids.to(device)
    cond_attention_mask = text_inputs.attention_mask.to(device)
    with torch.no_grad():
        cond_embeddings = text_encoder(cond_input_ids, attention_mask=cond_attention_mask)[0].to(device).to(transformer_dtype)
    cond_attention_mask = cond_attention_mask.to(device)

    # Prepare unconditional (empty) embeddings
    uncond_inputs = tokenizer("", padding="max_length", max_length=120, return_tensors="pt")
    uncond_input_ids = uncond_inputs.input_ids.to(device)
    uncond_attention_mask = uncond_inputs.attention_mask.to(device)
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input_ids, attention_mask=uncond_attention_mask)[0].to(device).to(transformer_dtype)
    uncond_attention_mask = uncond_attention_mask.to(device)

    # Batch embeddings: [uncond, cond]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
    attention_masks = torch.cat([uncond_attention_mask, cond_attention_mask], dim=0)

    # Timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Latent shape
    height = width = 512 if "512" in model_id else 1024
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = getattr(transformer.config, "in_channels", 4)
    latent_shape = (1, latent_channels, latent_height, latent_width)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=transformer_dtype)
    latents = latents * scheduler.init_noise_sigma

    # Diagnostics
    diagnostics: Dict[str, List[Any]] = {
        "timesteps": [],
        "noise_mean_abs_uncond": [],
        "noise_mean_abs_cond": [],
        "noise_mean_abs_final": [],
        "latents_mean_abs": [],
    }

    # Denoising loop with batched (uncond+cond) inference
    print("Starting denoising loop with manual CFG...")
    for i, t in enumerate(tqdm(timesteps, desc="Manual CFG Sampling", leave=False)):
        # Expand latents for batched pred: two copies (uncond, cond)
        latent_model_input = torch.cat([latents] * 2, dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # timestep tensor for the batch
        current_timestep = torch.full((latent_model_input.shape[0],), int(t), device=device, dtype=torch.long)

        # Predict noise for both uncond+cond
        noise_pred_batch = transformer(
            latent_model_input.to(transformer_dtype),
            timestep=current_timestep,
            encoder_hidden_states=text_embeddings,
            encoder_attention_mask=attention_masks,
            return_dict=False
        )[0]

        if noise_pred_batch.shape[1] == 8:
            noise_pred_batch = noise_pred_batch[:, :4, :, :]

        # split predictions
        noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2, dim=0)

        # diagnostics before CFG
        diagnostics["timesteps"].append(int(t))
        diagnostics["noise_mean_abs_uncond"].append(float(noise_pred_uncond.abs().mean().cpu().item()))
        diagnostics["noise_mean_abs_cond"].append(float(noise_pred_cond.abs().mean().cpu().item()))
        diagnostics["latents_mean_abs"].append(float(latents.abs().mean().cpu().item()))

        # CFG combination
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Optionally clamp final noise for stability
        if use_clamp:
            noise_pred = noise_pred.clamp(-clamp_value, clamp_value)

        diagnostics["noise_mean_abs_final"].append(float(noise_pred.abs().mean().cpu().item()))

        # Step scheduler (use return_dict=True)
        scheduler_output = scheduler.step(noise_pred, t, latents, return_dict=True)
        prev_sample = getattr(scheduler_output, "prev_sample", None)
        if prev_sample is None:
            prev_sample = scheduler_output[0] if isinstance(scheduler_output, (tuple, list)) else None

        # update latents for next step
        if prev_sample is None:
            raise RuntimeError("Scheduler did not return prev_sample; cannot continue sampling.")
        latents = prev_sample

    # Decode final latents to image
    with torch.no_grad():
        scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
        vae_dtype = next(vae.parameters()).dtype
        decoded = vae.decode((latents / scaling_factor).to(vae_dtype), return_dict=False)[0]
        image = (decoded / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        pil = Image.fromarray(image[0])

    meta = {"model_id": model_id, "prompt": prompt, "guidance_scale": guidance_scale, "seed": seed}
    out = {"image": pil, "diagnostics": diagnostics, "meta": meta}
    print("Done generating.")
    return pil, out
 

# %%
# Test manual CFG with different guidance scales
# Test CFG generation at different scales
print("Testing manual CFG implementation...\n")

test_prompt = "a beautiful mountain landscape with a lake at sunset"
test_scales = [1.0, 5.0, 7.5, 10.0]
device = "cuda:0"

fig, axes = plt.subplots(1, len(test_scales), figsize=(20, 5))
for i, w in enumerate(test_scales):
    pil_img, _ = generate_with_cfg(
        prompt=test_prompt,
        guidance_scale=w,
        num_inference_steps=30,
        seed=42,
        device=device,
    )
    axes[i].imshow(pil_img)
    axes[i].set_title(f"CFG w={w}", fontsize=14)
    axes[i].axis("off")

plt.suptitle(f"Manual CFG Comparison\nPrompt: '{test_prompt}'", fontsize=16)
plt.tight_layout()
plt.savefig("manual_cfg_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Sensitivity Analysis Function
def cfg_sensitivity_analysis(
    prompt="a beautiful mountain landscape with lakes and forests",
    guidance_scales=[0, 1, 2, 5, 7.5, 10],
    num_inference_steps=30,
    seeds=[42, 123, 999],
    device="cuda:0",
):
    results = {}
    for w in guidance_scales:
        print(f"\nGenerating for w={w}")
        results[w] = []
        for s in seeds:
            pil_img, _ = generate_with_cfg(
                prompt=prompt,
                guidance_scale=w,
                num_inference_steps=num_inference_steps,
                seed=s,
                device=device,
            )
            results[w].append(pil_img)
    return results


# %%
# Run Sensitivity Analysis
prompt = "a cozy cabin by a lake surrounded by pine forests"
prompt1 = "an airplane flying in space"
prompt2 = "a futuristic city skyline at night with neon lights"

guidance_values = [0, 1, 2, 5, 7.5, 10]
results = cfg_sensitivity_analysis(prompt=prompt, guidance_scales=guidance_values, device=device)
results1 = cfg_sensitivity_analysis(prompt=prompt1, guidance_scales=guidance_values, device=device)
results2 = cfg_sensitivity_analysis(prompt=prompt2, guidance_scales=guidance_values, device=device)

# Visualize results (first seed for each w)
fig, axes = plt.subplots(1, len(guidance_values), figsize=(25, 6))
for i, w in enumerate(guidance_values):
    axes[i].imshow(results[w][0])
    axes[i].set_title(f"w={w}", fontsize=14)
    axes[i].axis("off")
plt.suptitle(f"CFG Sensitivity Analysis — Prompt: '{prompt}'", fontsize=18)
plt.tight_layout()
plt.savefig("cfg_sensitivity_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

fig1, axes1 = plt.subplots(1, len(guidance_values), figsize=(25, 6))
for i, w in enumerate(guidance_values):
    axes1[i].imshow(results1[w][0])
    axes1[i].set_title(f"w={w}", fontsize=14)
    axes1[i].axis("off")
plt.suptitle(f"CFG Sensitivity Analysis — Prompt: '{prompt1}'", fontsize=18)
plt.tight_layout()
plt.savefig("cfg_sensitivity_analysis1.png", dpi=150, bbox_inches="tight")
plt.show()

fig2, axes2 = plt.subplots(1, len(guidance_values), figsize=(25, 6))
for i, w in enumerate(guidance_values):
    axes2[i].imshow(results2[w][0])
    axes2[i].set_title(f"w={w}", fontsize=14)
    axes2[i].axis("off")
plt.suptitle(f"CFG Sensitivity Analysis — Prompt: '{prompt2}'", fontsize=18)
plt.tight_layout()
plt.savefig("cfg_sensitivity_analysis2.png", dpi=150, bbox_inches="tight")
plt.show()