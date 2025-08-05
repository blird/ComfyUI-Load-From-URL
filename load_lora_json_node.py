import os
import json
import folder_paths
import comfy.utils, comfy.sd
from .utils import get_cache_path, ensure_cache_dir, download_if_needed


class LoadLoraFromJSON:
    """
    Load multiple LoRAs from URLs specified in a JSON array.
    
    Expected JSON format:
    [
      {
        "url": "https://example.com/lora.safetensors",
        "strength": 1.0
      },
      {
        "url": "https://example.com/another_lora.safetensors", 
        "strength": 0.8
      }
    ]
    """

    CATEGORY = "loaders"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_loras_from_json"

    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_loras")
        ensure_cache_dir(self.cache_dir)

    # ----- ComfyUI interface -----
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras_json": ("STRING", {
                    "default": '[\n  {\n    "url": "https://example.com/lora.safetensors",\n    "strength": 1.0\n  }\n]',
                    "multiline": True,
                    "placeholder": "JSON array of LoRA objects with url and strength"
                }),
            },
            "optional": {
                "downloader": (["auto", "aria2", "python"], {"default": "auto"}),
            }
        }



    # ----- public entry point -----
    def load_loras_from_json(self, model, loras_json, downloader="auto", **kwargs):
        """Load multiple LoRAs from URLs specified in a JSON array."""
        try:
            current_model = model
            loaded_count = 0
            
            # Parse JSON input
            try:
                loras_data = json.loads(loras_json.strip())
                if not isinstance(loras_data, list):
                    print(f"[LoadLoraFromJSON] Error: JSON must be an array/list, got {type(loras_data)}")
                    return (model,)
            except json.JSONDecodeError as e:
                print(f"[LoadLoraFromJSON] Error parsing JSON: {e}")
                return (model,)
            
            # Process each LoRA in the JSON array
            for i, lora_data in enumerate(loras_data):
                if not isinstance(lora_data, dict):
                    print(f"[LoadLoraFromJSON] Skipping item {i}: not a dictionary")
                    continue
                
                # Extract LoRA properties with defaults
                url = lora_data.get("url", "").strip()
                strength = lora_data.get("strength", 1.0)
                
                # Skip if no URL provided
                if not url:
                    print(f"[LoadLoraFromJSON] Skipping LoRA {i+1}: no URL provided")
                    continue
                    
                # Skip if strength is 0
                if strength == 0.0:
                    print(f"[LoadLoraFromJSON] Skipping LoRA {i+1}: strength is 0")
                    continue
                
                try:
                    dest = get_cache_path(url, self.cache_dir)
                    seconds, tool = download_if_needed(url, dest, downloader)
                    print(f"[LoadLoraFromJSON] LoRA {i+1} - Downloader: {tool}, Time: {seconds:.2f}s")
                    
                    lora_obj = comfy.utils.load_torch_file(dest)
                    
                    # Apply lora with strength (clip strength = 0 to match original behavior)
                    current_model, _ = comfy.sd.load_lora_for_models(
                        current_model, None, lora_obj, strength, 0
                    )
                    loaded_count += 1
                    print(f"[LoadLoraFromJSON] Successfully loaded LoRA {i+1} (strength: {strength})")
                    
                except Exception as e:
                    print(f"[LoadLoraFromJSON] Error loading LoRA {i+1} from {url}: {e}")
                    continue
            
            print(f"[LoadLoraFromJSON] Successfully loaded {loaded_count} LoRAs")
            return (current_model,)
            
        except Exception as e:
            print(f"[LoadLoraFromJSON] Fatal error: {e}")
            return (model,) 