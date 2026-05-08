# __init__.py
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import hashlib
from openai import OpenAI


class GrokAPIKey:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "xai-your-very-long-api-key-here",
                    "password": True,
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
    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                
                "image_description_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Describe the image with extreme cinematic detail. "
                               "Focus on subject appearance, clothing, facial expression, lighting, colors, "
                               "composition, background elements, and overall mood.",
                }),
                
                "action_instructions": ("STRING", {
                    "multiline": True,
                    "default": "Tell the model exactly what should happen in this image for video generation.\n\n"
                               "Focus on main subject movement, camera motion (pan, zoom, orbit, dolly), "
                               "timing, speed, secondary motion (hair, clothes, particles), and mood changes.",
                }),
                
                "model": (["grok-4.3", "grok-2-vision-latest", "grok-2-vision-1212"], {"default": "grok-4.3"}),
                "use_cache": ("BOOLEAN", {"default": True, "label": "✅ Use Cache (skip if unchanged)"}),
                "clear_cache": ("BOOLEAN", {"default": False, "label": "🗑️ Clear Cache"}),
            },
            "optional": {
                "grok_api_key": ("STRING", {"forceInput": True}),
                "max_tokens": ("INT", {
                    "default": 1200,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "label": "Max Tokens (256–4096)"
                }),
                "temperature": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "describe_image"
    CATEGORY = "Grok/xAI"

    def _get_input_hash(self, image, img_desc, action, model, max_tokens):
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        small = torch.nn.functional.avg_pool2d(img_tensor.permute(2,0,1).unsqueeze(0), kernel_size=8).squeeze(0)
        img_bytes = small.numpy().tobytes()

        combined = (img_bytes + 
                   str(img_desc).encode() + 
                   str(action).encode() + 
                   str(model).encode() + 
                   str(max_tokens).encode())
        return hashlib.md5(combined).hexdigest()

    def describe_image(self, image, 
                       image_description_instructions,
                       action_instructions,
                       model="grok-4.3",
                       use_cache=True,
                       clear_cache=False,
                       grok_api_key=None, 
                       max_tokens=1200, 
                       temperature=0.75):

        if clear_cache:
            self._cache.clear()
            print("🗑️ Grok cache cleared")

        final_key = (
            (grok_api_key or "").strip() or
            os.getenv("GROK_API_KEY", "").strip()
        )

        if not final_key:
            raise ValueError("❌ Grok API key is required!")

        # Safety clamp
        max_tokens = max(256, min(4096, int(max_tokens)))

        input_hash = self._get_input_hash(image, image_description_instructions, 
                                        action_instructions, model, max_tokens)

        # Cache hit
        if use_cache and input_hash in self._cache:
            cached = self._cache[input_hash]
            print(f"✅ Cache hit! Reusing prompt ({len(cached)} chars)")
            return (image, cached)

        print("🔄 Calling Grok API...")

        # === Image to base64 ===
        if len(image.shape) == 4:
            img_tensor = image[0]
        else:
            img_tensor = image

        np_img = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)

        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            pil_img = pil_img.convert("RGB")

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=92)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{img_base64}"

        full_prompt = f"""You are an expert prompt engineer for image-to-video models.

IMAGE DESCRIPTION GUIDELINES:
{image_description_instructions}

DESIRED MOTION / ACTION:
{action_instructions}

Combine both into one highly detailed, cohesive, and cinematic prompt."""

        client = OpenAI(api_key=final_key, base_url="https://api.x.ai/v1")

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}}
                ]
            }],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_prompt = response.choices[0].message.content.strip()

        if use_cache:
            self._cache[input_hash] = generated_prompt

        print(f"✅ Grok Image Describer: Generated {len(generated_prompt)} character prompt")
        return (image, generated_prompt)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "GrokImageDescriber": GrokImageDescriber,
    "GrokAPIKey": GrokAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageDescriber": "🔥 Grok Image Describer (Dual + Cache)",
    "GrokAPIKey": "🔑 Grok API Key",
}

print("🔥 Grok Prompts: Loaded successfully (Dual Instructions + Caching)")