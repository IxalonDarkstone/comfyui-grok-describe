from .grok_describe import GrokImageDescriber  # adjust import if you change the filename/class

NODE_CLASS_MAPPINGS = {
    "GrokImageDescriber": GrokImageDescriber
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageDescriber": "🔥 Grok Image Describer (I2V Prompt)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]