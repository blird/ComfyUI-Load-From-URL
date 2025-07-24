from .load_lora_url_node import LoadLoraFromURL
from .load_video_url_node import LoadVideoFromURL

NODE_CLASS_MAPPINGS = {
    "Load LoRA From URL": LoadLoraFromURL,
    "Load Video From URL": LoadVideoFromURL
}

__all__ = ['NODE_CLASS_MAPPINGS']