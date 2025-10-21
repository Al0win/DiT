# DiT (Diffusion Transformer) ProjectDiT.ipynb is the main notebook where I wrote the codde and implemented everything.



This repository contains implementations of Diffusion Transformer (DiT) models for image generation, including both training from scratch and using pretrained models with Classifier-Free Guidance (CFG).python run_experiments.py => To run the various experiments mentioned in task 4 and get observations.



## Project StructureAfter run_experiments.py 



```python analyze_experiments.py experiments/<timestamp>/experiment_results.csv

DiT/

├── dit_models.py                    # Core DiT model classes and utilitiesto get comparison plots
├── run_experiments.py              # Train DiT models from scratch with various configurations
├── analyze_experiments.py          # Analyze training results and generate reports
├── dit_pretrained_and_cfg.py       # Pretrained DiT with CFG visualization (PRIMARY ASSIGNMENT SCRIPT)
├── visualize_trained_models.py     # Visualize evolution from trained models
├── test_dit_setup.py              # Test pretrained DiT setup
├── compare_pixel_vs_latent.py     # Compare pixel-space vs latent-space training
├── experiments/                    # Trained model checkpoints and results
│   └── 20251021_053158/           # Experiment run directory
│       ├── exp_001_p2_d8_h6_T1000/
│       ├── exp_002_p4_d8_h6_T1000/
│       └── ...
└── DiT-main/                      # Reference implementation (not used directly)
```

## Scripts Overview

### 1. `dit_models.py` - Core Module
**Purpose**: Contains all DiT model implementations and utilities without execution code.

**Key Components**:
- `DiT`: Main Diffusion Transformer model
- `NoiseScheduler`: Implements noise scheduling (linear/cosine) for diffusion process
- `VAEWrapper`: Wrapper for Stable Diffusion VAE for latent-space training
- `LandscapeDataset`: Dataset loader for landscape images
- `ddpm_sample()`: DDPM sampling (stochastic)
- `ddim_sample()`: DDIM sampling (deterministic, faster)
- Training utilities: loss functions, FID score calculation

**Usage**: Import classes/functions into other scripts.

```python
from dit_models import DiT, NoiseScheduler, ddpm_sample
```

---

### 2. `run_experiments.py` - Training Script
**Purpose**: Train DiT models from scratch with different hyperparameter configurations.

**Features**:
- Trains multiple DiT models with varying patch size, depth, num_heads, timesteps
- Supports both pixel-space and latent-space (VAE) training
- Saves checkpoints, training curves, and sample generations
- Generates comparison visualizations

**Usage**:
```bash
python run_experiments.py
```

**Configurations Tested**:
- Patch sizes: 2, 4, 8
- Depths: 4, 6, 8, 10
- Attention heads: 1, 2, 4, 6, 8
- Timesteps: 100, 250, 500, 1000

**Output**: Saves to `experiments/<timestamp>/`
- Model checkpoints: `exp_XXX_<config>_checkpoint.pth`
- Training curves: `exp_XXX_<config>_training_curves.png`
- Generated samples: `exp_XXX_<config>_samples.png`

---

### 3. `analyze_experiments.py` - Results Analysis
**Purpose**: Analyze training results across all experiments and identify best configurations.

**Features**:
- Loads all experiment results
- Generates comparative visualizations
- Creates summary tables of best configurations
- Plots loss curves, sample quality comparisons

**Usage**:
```bash
python analyze_experiments.py experiments/<timestamp>/experiment_results.csv
```

**Output**: Creates `analysis/` subdirectory with:
- Best configurations summary
- Comparative plots
- Performance metrics

---

### 4. `dit_pretrained_and_cfg.py` - **PRIMARY ASSIGNMENT SCRIPT**
**Purpose**: Uses pretrained DiT model with manual CFG implementation and DDIM sampler.

**Assignment Requirements Addressed**:
✅ Uses pretrained DiT-based model (PixArt-alpha)  
✅ DDIM sampler  
✅ Visualizes x_0 prediction evolution in 10x10 grid (100 steps)  
✅ Manual CFG implementation  
✅ Shows CFG formula explicitly  
✅ Sensitivity analysis for guidance scale w (0-10)  

**Key Features**:
- **Model**: PixArt-alpha (DiT architecture with Transformer, not UNet)
- **Scheduler**: DDIM (explicitly replaced default scheduler)
- **Manual CFG**: Implements `noise_pred = uncond + w × (cond - uncond)` explicitly
- **Evolution Visualization**: Captures x_0 at each denoising step
- **Sensitivity Analysis**: Tests w from 0 to 10

**Functions**:
1. `visualize_diffusion_evolution_pretrained()`: Generate 10x10 evolution grid
2. `generate_with_cfg()`: Manual CFG implementation
3. `classifier_free_guidance()`: Explicit CFG formula
4. `cfg_sensitivity_analysis()`: Test multiple w values with different seeds

**Usage**:
```bash
python dit_pretrained_and_cfg.py
```

**Outputs**:
- `pretrained_diffusion_evolution.png`: Evolution across 10 key timesteps
- `single_image_evolution_10x10.png`: Full 100-step evolution grid
- `manual_cfg_comparison.png`: CFG comparison (w=1.0, 5.0, 7.5, 10.0)
- `cfg_sensitivity_detailed.png`: Detailed sensitivity analysis (w=0-10)

**Important Notes**:
- **PixArt Guidance Scales**: Use w=3.0-5.0 (trained with w=4.5)
  - Unlike Stable Diffusion (w=7.5-10), PixArt uses lower w values
  - Higher w (>7) causes oversaturation and artifacts in PixArt
- **PixArt Style**: PixArt has an artistic/illustration style (not photorealistic)
- **DiT Architecture**: Uses Transformer blocks, not UNet

---

### 5. `visualize_trained_models.py` - Visualize Custom Models
**Purpose**: Generate 10x10 evolution grids from your custom-trained DiT models.

**Features**:
- Loads trained model checkpoints
- Generates evolution visualization similar to pretrained models
- Shows x_0 prediction at each denoising step
- Works with both pixel-space and latent-space models

**Usage**:
```bash
python visualize_trained_models.py
```

**Configuration** (edit in script):
```python
EXPERIMENTS_ROOT = "/mnt/local/hf/DiT/experiments/20251021_053158"
OUTPUT_DIR = "/mnt/local/hf/DiT/visualizations/trained_models"
```

**Output**: Creates `visualizations/trained_models/` with:
- `exp_XXX_<config>_evolution_10x10.png` for each experiment

---


### 6. `compare_pixel_vs_latent.py` - Training Comparison
**Purpose**: Compare pixel-space vs latent-space DiT training.

**Features**:
- Trains two identical models (one pixel, one latent)
- Compares training speed, memory usage, sample quality
- Generates comparative visualizations

**Usage**:
```bash
python compare_pixel_vs_latent.py
```

---

## Assignment Task Mapping

### Task: Pretrained DiT with Evolution Visualization

**Requirement**: *"Utilize a pre-trained DiT-based diffusion model (e.g., FLUX.1-dev or a similar DiT architecture) with a DDIM sampler. Visualize the model's evolving prediction of the clean image (100 images in a 10X10 grid) at various time steps during the reverse sampling process."*

**Implementation**: `dit_pretrained_and_cfg.py`
- ✅ Model: PixArt-alpha (DiT architecture)
- ✅ Sampler: DDIM (explicitly replaced default)
- ✅ Visualization: 10x10 grid (100 steps)
- ✅ x_0 Prediction: Captured at each step without drift

### Task: Classifier-Free Guidance (CFG)

**Requirement**: *"Implement CFG logic for combining conditional/unconditional predictions. Your code should show final noise prediction from conditional/unconditional outputs and guidance scale w."*

**Implementation**: `dit_pretrained_and_cfg.py` - `generate_with_cfg()`
- ✅ Manual CFG: Explicit formula implementation
- ✅ Shows Formula: `noise_pred = uncond + w × (cond - uncond)`
- ✅ Conditional & Unconditional: Separated explicitly
- ✅ Guidance Scale: Parameterized as `w`

**Key Code**:
```python
# Separate predictions
noise_pred_uncond, noise_pred_cond = noise_pred_batch.chunk(2)

# CFG Formula (implemented manually)
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
```

### Task: CFG Sensitivity Analysis

**Requirement**: *"Perform a sensitivity analysis for the guidance scale w (e.g., w from 0-10) and observe effects on sample quality/diversity."*

**Implementation**: `dit_pretrained_and_cfg.py` - Final section
- ✅ Tests w: 0, 1.0, 3.0, 5.0, 7.5, 10.0
- ✅ Same seed: Isolates effect of w
- ✅ Quality analysis: Higher w = stronger prompt adherence (up to a point)
- ✅ Artifacts: Shows over-saturation at w=10 (expected for PixArt)

**Difference from CFG Comparison**:
- **CFG Comparison**: Quick test with 4 w values, demonstrates CFG works
- **Sensitivity Analysis**: Comprehensive test with 6 w values, analyzes quality-diversity trade-off with detailed annotations

---

## Understanding CFG Results

### Why Higher w Produces Worse Images in PixArt

**This is EXPECTED!** PixArt-alpha was trained with w=4.5, not w=7.5-10 like Stable Diffusion.

**Reasons**:
1. **Out of Distribution**: w>7 is outside PixArt's training range
2. **Over-Amplification**: `w × (cond - uncond)` amplifies the signal too much
3. **Model Artifacts**: Causes color saturation, broken structures, visual glitches

**Correct Guidance Scales**:
- **PixArt-alpha**: w=3.0 to 5.0 (optimal: 4.5)
- **Stable Diffusion**: w=7.5 to 10.0
- **FLUX.1-dev**: w=3.5 to 5.0

### Expected Behavior by Guidance Scale

| w | Behavior | Expected Quality |
|---|----------|------------------|
| 0 | Unconditional (ignores prompt) | Low (random) |
| 1.0 | Balanced | Moderate |
| 3.0-5.0 | Optimal for PixArt | High |
| 7.5 | Too strong for PixArt | Artifacts |
| 10.0 | Very strong | Over-saturated, broken |

### PixArt Style

PixArt-alpha naturally produces **artistic/illustration-style** images, not photorealistic ones. This is by design, not an error. CFG works correctly with this style.

---

## Key Technical Details

### 1. DiT vs UNet Architecture

**UNet** (Stable Diffusion):
- Convolutional encoder-decoder
- Downsampling → Processing → Upsampling
- Spatial inductive bias

**DiT** (PixArt, FLUX, SD3):
- Transformer blocks (like Vision Transformer)
- Patch-based processing
- Self-attention across patches
- More scalable than UNet

**Code Difference**:
```python
# UNet-based (Stable Diffusion)
noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeds)

# DiT-based (PixArt)
noise_pred = pipe.transformer(latents, t, encoder_hidden_states=text_embeds)
```

### 2. DDIM vs DDPM Sampling

**DDPM** (Denoising Diffusion Probabilistic Models):
- Stochastic (adds noise at each step)
- Requires many steps (e.g., 1000)
- High quality but slow

**DDIM** (Denoising Diffusion Implicit Models):
- Deterministic (or optionally stochastic)
- Fewer steps (e.g., 50-100)
- Faster, same quality

**Assignment uses DDIM** as specified.

### 3. Manual CFG Implementation

**Why manual?** Assignment requires showing the CFG formula explicitly.

**Pipeline's built-in CFG** (hidden):
```python
pipe(prompt, guidance_scale=7.5)  # CFG happens internally
```

**Manual CFG** (explicit):
```python
# Get unconditional prediction
uncond_noise = transformer(latents, t, uncond_embeds)

# Get conditional prediction  
cond_noise = transformer(latents, t, cond_embeds)

# Apply CFG formula (THIS IS WHAT WE'RE DEMONSTRATING)
noise_pred = uncond_noise + w * (cond_noise - uncond_noise)
```

---

## Environment Setup

### Required Packages
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install matplotlib pillow numpy tqdm scipy
```

### HuggingFace Cache Configuration
**IMPORTANT**: Set these before running scripts to avoid filling up `/home/jovyan`:

```bash
export HF_HOME=/mnt/local/hf/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/local/hf/huggingface_cache
export HF_DATASETS_CACHE=/mnt/local/hf/huggingface_cache
export HF_METRICS_CACHE=/mnt/local/hf/huggingface_cache
export XDG_CACHE_HOME=/mnt/local/hf/huggingface_cache
```

---

## Running the Assignment Scripts

### 1. Test Setup (Recommended First)
```bash
cd /mnt/local/hf/DiT
export HF_HOME=/mnt/local/hf/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/local/hf/huggingface_cache
python test_dit_setup.py
```

### 2. Generate All Assignment Outputs
```bash
python dit_pretrained_and_cfg.py
```

**Expected Outputs**:
- `pretrained_diffusion_evolution.png`
- `single_image_evolution_10x10.png`
- `manual_cfg_comparison.png`
- `cfg_sensitivity_detailed.png`

### 3. Visualize Trained Models (Optional)
```bash
python visualize_trained_models.py
```

---

## Troubleshooting

### Out of Disk Space
**Problem**: `/home/jovyan` fills up  
**Solution**: Set HF cache environment variables (see above)

### ImportError: diffusers
**Problem**: Package not installed  
**Solution**: `pip install diffusers transformers`

### CUDA Out of Memory
**Problem**: GPU memory exhausted  
**Solution**: 
- Reduce `num_inference_steps`
- Use smaller model variant (512x512 instead of 1024)
- Clear cache: `torch.cuda.empty_cache()`

### PixArt Produces "Weird" Images
**Problem**: Images look artistic/surreal  
**Solution**: This is expected! PixArt has an illustration style. Try:
- Adjusting guidance scale (w=3.0-5.0)
- Different prompts
- For photorealistic images, use different model (SD, FLUX)

---

## References

- **DiT Paper**: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
- **PixArt-α**: https://huggingface.co/PixArt-alpha
- **DDIM Paper**: "Denoising Diffusion Implicit Models" (Song et al., 2020)
- **CFG Paper**: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)

---

## Summary

This project demonstrates:
1. ✅ **DiT Architecture**: Transformer-based diffusion (not UNet)
2. ✅ **DDIM Sampling**: Fast deterministic sampling
3. ✅ **Manual CFG**: Explicit formula implementation
4. ✅ **Evolution Visualization**: x_0 prediction at each step
5. ✅ **Sensitivity Analysis**: Effect of guidance scale w
6. ✅ **Training from Scratch**: Custom DiT models

**Primary Assignment Script**: `dit_pretrained_and_cfg.py`
