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
Install
```

Restart ComfyUI.

### Getting API Key

1.  Go to https://console.x.ai/
2.  Create / copy your API key (starts with xai-)

### How to Use

1.  Add 🔑 Grok API Key node → paste your key (recommended)
2.  Add 🔥 Grok Image Describer node
3.  Connect:
	•  Image input
	•  Optional: grok_api_key from the key node
4.  Write your instructions (default is optimized for I2V)
5.  (Recommended) Add a Prompt Verify or Pause Workflow node after it so you can edit the prompt before sending to Wan / I2V.

### Nodes

•  🔑 Grok API Key – Store key once, reuse across workflows
•  🔥 Grok Image Describer – Main vision + prompt generation node