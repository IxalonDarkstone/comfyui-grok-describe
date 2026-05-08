import torch
import numpy as np
from PIL import Image
import io
import base64
from openai import OpenAI

class GrokImageDescriber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "instructions": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in extreme detail optimized for image-to-video generation (e.g. Wan or similar i2v models). Include: main subject and its appearance, potential motion/actions, camera angle/movement suggestions, lighting, colors, atmosphere/mood, composition, background elements, style/aesthetic."
                }),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["grok-4.3", "grok-2-vision-latest", "grok-2-vision-1212"], {"default": "grok-4.3"}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1200, "min": 256, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "prompt")
    FUNCTION = "describe_image"
    CATEGORY = "Grok/xAI"

    import os
    from openai import OpenAI

    def describe_image(self, image, instructions, api_key, model, max_tokens=1200, temperature=0.7):
    # Priority: Node input > Environment variable
        final_key = api_key.strip() if api_key else os.getenv("GROK_API_KEY", "").strip()
    
        if not final_key:
            raise ValueError("❌ Grok API key is required! set GROK_API_KEY environment variable.")

    # ... rest of your code unchanged ...

        # Convert ComfyUI IMAGE tensor → base64 JPEG
        if len(image.shape) == 4:
            img_tensor = image[0]  # take first image in batch
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

        # Call Grok API (OpenAI-compatible)
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instructions},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        "detail": "high"
                    }
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

        print(f"✅ Grok Image Describer: Generated {len(generated_prompt)} char prompt")
        return (image, generated_prompt)  # passthrough image + new prompt


# Register the node
NODE_CLASS_MAPPINGS = {"GrokImageDescriber": GrokImageDescriber}
NODE_DISPLAY_NAME_MAPPINGS = {"GrokImageDescriber": "🔥 Grok Image Describer (I2V Prompt)"}