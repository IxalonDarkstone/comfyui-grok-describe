from .grok_describe import (
    GrokImageDescriber,
    GrokAPIKey
)

NODE_CLASS_MAPPINGS = {
    "GrokImageDescriber": GrokImageDescriber,
    "GrokAPIKey": GrokAPIKey,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokImageDescriber": "🔥 Grok Image Describer (I2V Prompt)",
    "GrokAPIKey": "🔑 Grok API Key",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]