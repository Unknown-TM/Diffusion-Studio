# download_sd_v1_5.py

import os
from diffusers import StableDiffusionPipeline
import torch

# ---- Configuration ----
model_name = "runwayml/stable-diffusion-v1-5"  # model identifier
save_dir = "./model/stable-diffusion-v1-5"          # local folder to save the model
use_auth_token = True                          # set to True if private repo (requires HF token)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

print(f"Downloading {model_name} to {save_dir}...")

# Load the model pipeline (this will download all model files)
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    use_auth_token=use_auth_token
)

# Save model locally
pipe.save_pretrained(save_dir)

print(f"Model saved to {save_dir}!")
