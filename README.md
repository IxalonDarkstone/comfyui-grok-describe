# ComfyUI Grok Image Describer

**Generate high-quality image descriptions with Grok (xAI) optimized for image-to-video models** (Wan, Hunyuan, LTX, etc.).

Features:
- Freeform instructions for tailored prompts
- Visual Grok API Key node (like other popular API nodes)
- Supports direct input, connected key node, or `GROK_API_KEY` env var
- High-detail vision + temperature / max_tokens control
- Image passthrough for easy workflow chaining

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for **"Grok Image Describer"** or **"comfyui-grok-describe"**
3. Install and restart ComfyUI

### Manual Git Install
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/comfyui-grok-describe.git
cd comfyui-grok-describe
pip install -r requirements.txt