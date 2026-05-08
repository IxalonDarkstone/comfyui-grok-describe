import torch
import numpy as np
from PIL import Image
import io
import base64
import os
from openai import OpenAI

class GrokAPIKey:
    """Dedicated node for storing and reusing the Grok API key visually"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "xai-your-very-long-api-key-here",
                    "password": True,           # Hides characters
                }),
            },
            "optional": {
                "save_to_env": ("BOOLEAN", {"default": False, "label": "Save to GROK_API_KEY env var"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"
    CATEGORY = "Grok/xAI"

    def get_key(self, api_key, save_to_env=False):
        key = api_key.strip()
        if save_to_env and key:
            os.environ["GROK_API_KEY"] = key
            print("✅ Grok API key saved to environment variable")
        return (key,)


class GrokImageDescriber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in extreme detail optimized for image-to-video generation (Wan, Hunyuan, etc.). "
                               "Include: subject appearance, possible motion, camera movement ideas, lighting, mood, composition, "
                               "colors, background, and artistic style.",
                }),
                "api_key": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "xai-... (or leave empty and use Grok API Key node or GROK_API_KEY env var)",
                    "password": True,
                }),
                "model": (["grok-4.3", "grok-2-vision-latest", "grok-2-vision-1212"], {"default": "grok-4.3"}),
            },
            "optional": {
                "grok_api_key": ("STRING", {"forceInput": True}),  # Connect from GrokAPIKey node
                "max_tokens": ("INT", {"default": 1200, "min": 256, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "describe_image"
    CATEGORY = "Grok/xAI"

    def describe_image(self, image, instructions, api_key="", model="grok-4.3",
                       grok_api_key=None, max_tokens=1200, temperature=0.7):

        # Priority order for API key: 1. Connected key node → 2. Direct field → 3. Environment variable
        final_key = (
            (grok_api_key or "").strip() or
            api_key.strip() or
            os.getenv("GROK_API_KEY", "").strip()
        )

        if not final_key:
            raise ValueError("❌ Grok API key is required!\n"
                           "• Paste it in the api_key field, or\n"
                           "• Use a 'Grok API Key' node and connect it, or\n"
                           "• Set GROK_API_KEY environment variable.")

        # --- Image to base64 ---
        if len(image.shape) == 4:
            img_tensor = image[0]  # batch support
        else:
            img_tensor = image

        np_img = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)

        # Convert RGBA to RGB if needed
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            pil_img = pil_img.convert("RGB")

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=92)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_base64}"

        # --- Call Grok API ---
        client = OpenAI(
            api_key=final_key,
            base_url="https://api.x.ai/v1",
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url, "detail": "high"}
                }
            ]
        }]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_prompt = response.choices[0].message.content.strip()

        print(f"✅ Grok Image Describer: Generated {len(generated_prompt)} character prompt")
        return (image, generated_prompt)  # passthrough image


# Node registration
NODE_CLASS_MAPPINGS = {
    "GrokImageDescriber": GrokImageDescriber,
    "GrokAPIKey": GrokAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageDescriber": "🔥 Grok Image Describer (I2V Prompt)",
    "GrokAPIKey": "🔑 Grok API Key",
}