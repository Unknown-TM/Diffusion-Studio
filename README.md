# Stable Diffusion Image Generator

A Python-based image generation tool using Stable Diffusion v1.5 that allows you to create images from text prompts with GPU optimization and memory management.

## Features

- 🎨 **Text-to-Image Generation**: Create images from natural language prompts
- 🚀 **GPU Optimized**: Automatic CUDA detection with memory optimizations
- 💾 **Memory Efficient**: Attention slicing and CPU offloading for lower VRAM usage
- 📁 **Session Management**: Organized output with timestamped sessions
- 🧹 **Auto Cleanup**: Removes empty session folders automatically
- ⚡ **Interactive Mode**: Continuous prompt input for batch generation

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for GPU)
- 4GB+ VRAM for GPU usage

## Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model

Run the model download script to get Stable Diffusion v1.5:

```bash
python download_sd_v1_5.py
```

This will download the model files to `./model/stable-diffusion-v1-5/` (approximately 4GB).

## Usage

### Generate Images

Start the interactive image generation:

```bash
python generate_image.py
```

The script will:
1. Load the Stable Diffusion model
2. Create a new session folder with timestamp
3. Prompt you for text descriptions
4. Generate and save images to the session folder

### Example Prompts

- `"a beautiful sunset over mountains, digital art"`
- `"a cute cat wearing a space helmet, cartoon style"`
- `"futuristic city skyline at night, neon lights, cyberpunk"`
- `"vintage car in a garage, photorealistic"`

### Exit the Program

Type `exit` and press Enter to quit the program.

## Configuration

You can modify these settings in `generate_image.py`:

```python
# Generation parameters
num_inference_steps = 25   # Quality vs speed (20-50 recommended)
guidance_scale = 7.5       # Prompt adherence (7-15 recommended)

# Paths
model_path = "./model/stable-diffusion-v1-5"
base_output_dir = "./generated_images"
```

## Output Structure

Generated images are organized by session:

```
generated_images/
├── session_2025-01-15_14-30-25/
│   ├── image_1.png
│   ├── image_2.png
│   └── image_3.png
└── session_2025-01-15_15-45-10/
    └── image_1.png
```

## Performance Tips

### GPU Optimization
- The script automatically detects CUDA and applies optimizations
- Uses mixed precision (float16) for faster generation
- Enables attention slicing to reduce VRAM usage
- CPU offloading moves unused model parts to RAM

### CPU Usage
- Works on CPU but will be significantly slower
- Consider reducing `num_inference_steps` for faster generation
- Increase `guidance_scale` for better prompt adherence

### Memory Management
- If you encounter out-of-memory errors, try:
  - Reducing `num_inference_steps`
  - Using a smaller model variant
  - Closing other GPU-intensive applications

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce `num_inference_steps` to 15-20
- Close other GPU applications
- Restart the script to clear GPU memory

**"Model not found"**
- Ensure you've run `download_sd_v1_5.py` first
- Check that the model files are in `./model/stable-diffusion-v1-5/`

**Slow generation on CPU**
- This is normal - GPU is 10-50x faster
- Consider using Google Colab or other cloud GPU services

### Dependencies Issues

If you encounter dependency conflicts:

```bash
# Recreate environment
deactivate
rm -rf venv
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## File Structure

```
diffusion/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── download_sd_v1_5.py         # Model download script
├── generate_image.py           # Main image generation script
├── model/                      # Downloaded model files
│   └── stable-diffusion-v1-5/
└── generated_images/           # Output images by session
    └── session_YYYY-MM-DD_HH-MM-SS/
```

## License

This project uses the Stable Diffusion v1.5 model, which is subject to the [CreativeML Open RAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for the excellent library
- [Stability AI](https://stability.ai/) for the Stable Diffusion model
- [RunwayML](https://runwayml.com/) for hosting the model
