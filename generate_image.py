# generate_image_gpu_ram_optimized.py

import os
import shutil
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime

# ---- Configuration ----
model_path = "./model/stable-diffusion-v1-5"  # path to your downloaded model
num_inference_steps = 25   # fewer steps = faster generation
guidance_scale = 7.5       # creativity vs prompt adherence
base_output_dir = "./generated_images"

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create session folder
session_name = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(base_output_dir, session_name)
os.makedirs(output_dir, exist_ok=True)

# Load pipeline with optimizations
print(f"Loading model from {model_path} on {device}...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # Optional: disables NSFW filter to save VRAM
)

# Enable attention slicing for memory efficiency
pipe.enable_attention_slicing()

# Move to device and apply optimizations
if device == "cuda":
    try:
        pipe.enable_model_cpu_offload()  # Offload unused layers to RAM
        print("✅ GPU detected: using mixed precision + CPU offload + attention slicing")
    except Exception as e:
        print(f"⚠️ Could not enable CPU offload: {e}")
        pipe = pipe.to("cuda")
else:
    pipe = pipe.to("cpu")
    print("⚠️ Running on CPU only (slow).")

print(f"\n--- Interactive Stable Diffusion ---")
print(f"All images for this session will be saved in: {output_dir}")
print("Type your prompt and press Enter to generate an image.")
print("Type 'exit' to quit the program.\n")

image_counter = 1
images_created = 0

def cleanup_empty_folders():
    """Remove empty session folders inside generated_images."""
    if not os.path.exists(base_output_dir):
        return
    for folder in os.listdir(base_output_dir):
        folder_path = os.path.join(base_output_dir, folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"🗑️ Removed empty session folder: {folder_path}")

try:
    while True:
        prompt = input("Enter prompt: ")
        if prompt.strip().lower() == "exit":
            print("Exiting...")
            break
        if not prompt.strip():
            print("Prompt cannot be empty!")
            continue

        print(f"✨ Generating image for prompt: '{prompt}'")
        if device == "cuda":
            with torch.autocast("cuda"):
                image = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
        else:
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        output_path = os.path.join(output_dir, f"image_{image_counter}.png")
        image.save(output_path)
        print(f"✅ Saved image to {output_path}\n")
        image_counter += 1
        images_created += 1

finally:
    # If current session created no images, remove it
    if images_created == 0 and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"🗑️ Removed empty session folder: {output_dir}")

    # Cleanup all past empty session folders
    cleanup_empty_folders()
