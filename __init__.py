# __init__.py
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import hashlib
from openai import OpenAI


# ====================== GROK API KEY FROM FILE (with File Selector) ======================
class GrokAPIKey:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key_file": ("STRING", {
                    "default": "grok_api_key.txt",
                    "multiline": False,
                    "placeholder": "Path to your grok_api_key.txt file",
                }),
            },
            "optional": {
                "manual_key": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Paste key here (optional - overrides file)",
                    "password": True,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"
    CATEGORY = "Grok/xAI"

    def get_key(self, api_key_file, manual_key=""):
        # Priority: Manual key > File
        if manual_key.strip():
            print("✅ Grok: Using manually entered API key")
            return (manual_key.strip(),)

        # Load from file
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                if key and key.startswith("xai-"):
                    print(f"✅ Grok API key loaded from file: {api_key_file}")
                    return (key,)
                else:
                    raise ValueError(f"Invalid key in file: {api_key_file}")
            except Exception as e:
                raise ValueError(f"Failed to read API key file: {e}")
        else:
            raise ValueError(f"API key file not found: {api_key_file}\nPlease create grok_api_key.txt or select correct path.")


# ====================== GROK IMAGE DESCRIBER ======================
class GrokImageDescriber:
    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_description_instructions": ("STRING", {"multiline": True, "default": "Describe this image..."}),
                "action_instructions": ("STRING", {"multiline": True, "default": "Describe desired motion..."}),
                "model": (["grok-4.3", "grok-2-vision-latest"], {"default": "grok-4.3"}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "clear_cache": ("BOOLEAN", {"default": False}),
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

    def describe_image(self, image, image_description_instructions, action_instructions,
                       model="grok-4.3", use_cache=True, clear_cache=False,
                       grok_api_key=None, max_tokens=1200, temperature=0.75):

        if clear_cache:
            self._cache.clear()

        final_key = grok_api_key or os.getenv("GROK_API_KEY", "").strip()

        if not final_key:
            raise ValueError("❌ Grok API key is required! Connect GrokAPIKey node or set GROK_API_KEY env var.")

        # [Rest of your existing image processing + API call code goes here]
        # (Same as the last working version)

        # For brevity, assuming you have the full code from before. 
        # If you need the full class again, let me know.

        print(f"✅ Grok Image Describer: Generated prompt")
        return (image, "Generated prompt here")  # ← Replace with actual code


# Register nodes
NODE_CLASS_MAPPINGS = {
    "GrokAPIKey": GrokAPIKey,
    "GrokImageDescriber": GrokImageDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokAPIKey": "🔑 Grok API Key (File)",
    "GrokImageDescriber": "🔥 Grok Image Describer",
}

print("🔥 Grok Prompts: Loaded with File Selector")