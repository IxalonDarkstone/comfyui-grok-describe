**Updated README.md** for your current Grok Prompts custom node:

```markdown
# ComfyUI Grok Prompts

Custom nodes for ComfyUI that use xAI's Grok API to generate and enhance prompts from images or text.

## Nodes

### 🔥 Grok Image to Prompt (I2T)
Converts an image into a detailed text prompt.
- **Inputs**: Image, base prompt, prompt guidance instructions, model, API key file
- **Output**: Enhanced prompt string
- Best for creating high-quality prompts from reference images

### 🔥 Grok Image Describer (I2V)
Generates motion-focused prompts for image-to-video workflows.
- **Inputs**: Image, image description instructions, action/motion instructions, model, API key file
- **Output**: Image + video-ready prompt
- Optimized for Wan and other I2V models

### 🔥 Grok Prompt Enhancer (T2I)
Expands a short base prompt into a rich, detailed prompt.
- **Inputs**: Base prompt, enhancement instructions, model, API key file
- **Output**: Enhanced prompt string
- Ideal for text-to-image workflows

## Features

- **Secure API key handling** — Load key from `grok_api_key.txt` (or custom path). Never stored in workflows.
- **Caching** — Skips API calls when inputs haven't changed.
- **Models**: `grok-4.5` (default) and `grok-4.3`
- **Controls**: temperature, max_tokens, clear cache

## Installation

1. Clone into your custom nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/YOUR_USERNAME/comfyui-grok-describe.git
   ```

2. Install dependencies:
   ```bash
   pip install openai Pillow numpy
   ```

3. Create `grok_api_key.txt` in your ComfyUI root folder (or custom path) containing only your xAI API key:
   ```
   xai-your-key-here
   ```

4. Restart ComfyUI. Nodes appear under **Grok/xAI**.

## Usage

- Set `api_key_file` on each node (default: `grok_api_key.txt`).
- For vision nodes (I2T / I2V), connect an image.
- Use the guidance/instruction fields to control output style (Danbooru tags, natural language, NSFW, cinematic, etc.).
- Enable caching for repeated runs with the same inputs.

## Requirements

- ComfyUI
- xAI API key (get one at [console.x.ai](https://console.x.ai))
- `openai` Python package

## Notes

- Vision requests cost more tokens than text-only.
- Clear cache if you change instructions or want a fresh generation.
- Works with any diffusion model that accepts text prompts (Flux, SDXL, Wan, etc.).
```
2. Push the changes.

Let me know the exact GitHub URL if you want me to adjust anything, or if you need a version that includes screenshots/workflow examples.
