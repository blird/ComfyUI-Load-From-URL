from .load_lora_url_node import LoadLoraFromURL
from .load_video_url_node import LoadVideoFromURL
from .load_lora_json_node import LoadLoraFromJSON

NODE_CLASS_MAPPINGS = {
    "Load LoRA From URL": LoadLoraFromURL,
    "Load Video From URL": LoadVideoFromURL,
    "Load LoRAs from JSON": LoadLoraFromJSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load LoRA From URL": "Load LoRA from URL",
    "Load Video From URL": "Load Video from URL",
    "Load LoRAs from JSON": "Load LoRAs from JSON",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']