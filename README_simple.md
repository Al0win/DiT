# DiT (Diffusion Transformer) Project — Quick Guide

This repository contains scripts for training, analyzing, and visualizing Diffusion Transformer (DiT) models for image generation.

## Script Overview

- **dit_models.py**: Core DiT model classes and utilities. Import this in other scripts; do not run directly.

- **run_experiments.py**: Trains DiT models from scratch with different hyperparameters. Saves results and checkpoints.
  - Run: `python run_experiments.py`

- **analyze_experiments.py**: Analyzes experiment results and generates plots from the CSV output of training.
  - Run: `python analyze_experiments.py experiments/<timestamp>/experiment_results.csv`

- **dit_pretrained_using_pixelart.py**: Uses a pretrained PixArt-XL-2-512x512 model with DDIM sampling and classifier-free guidance (CFG). Visualizes the denoising process and CFG effects.
  - Run: `python dit_pretrained_using_pixelart.py`

- **visualize_trained_models.py**: Loads your trained DiT models and creates 10x10 grids showing the denoising evolution.
  - Run: `python visualize_trained_models.py`

- **dit_pretrained_and_cfg_using_sd.py**: Variant using Stable Diffusion as the backend for pretrained DiT-based visualization and CFG analysis.
  - Run: `python dit_pretrained_and_cfg_using_sd.py`

## Notes
- For experiment outputs, see the `experiments/`, `visualizations/`, `pixelart_results` and `sd1.4_results` folders.
