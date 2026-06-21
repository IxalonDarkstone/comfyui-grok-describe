# __init__.py
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import hashlib
from openai import OpenAI


# ====================== GROK API KEY FROM FILE ======================
class GrokAPIKey:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key_file": ("STRING", {
                    "default": "grok_api_key.txt",
                    "multiline": False,
                    "placeholder": "grok_api_key.txt or full path",
                }),
            },
            "optional": {
                "manual_key": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Manual override (optional)",
                    "password": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"
    CATEGORY = "Grok/xAI"

    def get_key(self, api_key_file, manual_key=""):
        if manual_key.strip():
            print("✅ Grok: Using manual API key")
            return (manual_key.strip(),)

        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if key.startswith("xai-"):
                    print(f"✅ Grok API key loaded from: {api_key_file}")
                    return (key,)
                else:
                    raise ValueError("File does not contain a valid xAI key")
            except Exception as e:
                raise ValueError(f"Error reading API key file: {e}")
        else:
            raise ValueError(f"API key file not found: {api_key_file}\nPlease create grok_api_key.txt")


# ====================== GROK IMAGE DESCRIBER (I2V) ======================
class GrokImageDescriber:
    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_description_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image with extreme cinematic detail..."
                }),
                "action_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Describe desired motion, camera movement, and actions..."
                }),
                "model": (["grok-4.3", "grok-2-vision-latest"], {"default": "grok-4.3"}),
                "use_cache": ("BOOLEAN", {"default": True, "label": "Use Cache"}),
                "clear_cache": ("BOOLEAN", {"default": False, "label": "Clear Cache"}),
            },
            "optional": {
                "grok_api_key": ("STRING", {"forceInput": True}),
                "max_tokens": ("INT", {"default": 1200, "min": 256, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "describe_image"
    CATEGORY = "Grok/xAI"

    def _get_input_hash(self, image, desc, action, model, max_tokens):
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image
        small = torch.nn.functional.avg_pool2d(img_tensor.permute(2,0,1).unsqueeze(0), kernel_size=8).squeeze(0)
        img_bytes = small.numpy().tobytes()
        combined = img_bytes + str(desc).encode() + str(action).encode() + str(model).encode() + str(max_tokens).encode()
        return hashlib.md5(combined).hexdigest()

    def describe_image(self, image, image_description_instructions, action_instructions,
                       model="grok-4.3", use_cache=True, clear_cache=False,
                       grok_api_key=None, max_tokens=1200, temperature=0.75):

        if clear_cache:
            self._cache.clear()
            print("🗑️ Grok I2V cache cleared")

        final_key = grok_api_key or os.getenv("GROK_API_KEY", "").strip()
        if not final_key:
            raise ValueError("❌ Grok API key required! Use GrokAPIKey node.")

        max_tokens = max(256, min(4096, int(max_tokens)))
        input_hash = self._get_input_hash(image, image_description_instructions, action_instructions, model, max_tokens)

        if use_cache and input_hash in self._cache:
            print("✅ Grok I2V: Cache hit")
            return (image, self._cache[input_hash])

        print("🔄 Grok Image Describer: Calling API...")

        # Image → Base64
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        np_img = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img).convert("RGB")

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=92)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        full_prompt = f"""You are an expert prompt engineer for image-to-video.

IMAGE DESCRIPTION:
{image_description_instructions}

DESIRED MOTION/ACTION:
{action_instructions}

Combine into one rich, detailed prompt."""

        client = OpenAI(api_key=final_key, base_url="https://api.x.ai/v1")

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}", "detail": "high"}}
                ]
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_prompt = response.choices[0].message.content.strip()

        if use_cache:
            self._cache[input_hash] = generated_prompt

        print(f"✅ Grok Image Describer: Generated {len(generated_prompt)} chars")
        return (image, generated_prompt)


# Register Nodes
NODE_CLASS_MAPPINGS = {
    "GrokAPIKey": GrokAPIKey,
    "GrokImageDescriber": GrokImageDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokAPIKey": "🔑 Grok API Key (File)",
    "GrokImageDescriber": "🔥 Grok Image Describer",
}

print("🔥 Grok Prompts: Loaded successfully (Latest Version)")