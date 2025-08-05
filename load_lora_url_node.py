import os, hashlib, shutil, subprocess, time, requests, re
from typing import Literal

from tqdm import tqdm
import folder_paths, comfy.utils, comfy.sd
from .utils import get_cache_path, ensure_cache_dir, download_if_needed


# ------------------------------------------------------------
# Quick CDN‑cache check (1 HEAD request, 200 ms)
# ------------------------------------------------------------
_CACHE_HIT_RE = re.compile(r"\bHIT\b", re.I)

def _edge_is_hot(url: str, timeout: float = 2.0) -> bool:
    """Return True if Front Door says the object is in edge cache."""
    try:
        r = requests.head(url, timeout=timeout)
        # Age header present and > 0 => cached
        if int(r.headers.get("Age", "0")) > 0:
            return True
        # X-Cache or X-FD-Cache-Type may say "HIT"
        x_cache = r.headers.get("X-Cache") or r.headers.get("X-FD-Cache-Type", "")
        return bool(_CACHE_HIT_RE.search(x_cache))
    except Exception:
        # If HEAD fails, fall back to assuming it's cold
        return False


# ------------------------------------------------------------
# Helper: choose downloader
# ------------------------------------------------------------
def _detect_downloader(url: str, preference: str = "auto") -> str:
    """
    'auto' = python if edge‑hot, else aria2; otherwise honor explicit choice.
    """
    if preference != "auto":
        return preference

    # Decide based on cache state
    is_hot = _edge_is_hot(url)
    if is_hot:
        return "python"  # single stream is fastest on warm cache
    if shutil.which("aria2c"):
        return "aria2"
    return "python"


# ------------------------------------------------------------
# Main node
# ------------------------------------------------------------
class LoadLoraFromURL:
    CATEGORY     = "loaders"
    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "load_lora"

    def __init__(self):
        self.cache_dir = os.path.join(folder_paths.get_input_directory(), "url_loras")
        ensure_cache_dir(self.cache_dir)

    # ----- ComfyUI interface -----
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url":      ("STRING",),
                "model":    ("MODEL",),
                "strength": ("FLOAT", { "default": 1.0, "min": 0.0,
                                        "max": 10.0, "step": 0.01 }),
            },
            "optional": {
                "downloader": (["auto", "aria2", "python"],),
            }
        }

    # ----- public entry -----
    def load_lora(self, url, model, strength, downloader="auto"):
        try:
            dest = get_cache_path(url, self.cache_dir)
            seconds, tool = download_if_needed(url, dest, downloader)
            print(f"[LoadLoraFromURL] Downloader: {tool}, Time: {seconds:.2f}s")
            lora_obj  = comfy.utils.load_torch_file(dest)
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora_obj, strength, 0
            )
            return (model_lora,)
        except Exception as e:
            print(f"[LoadLoraFromURL] Error: {e}")
            return (model,)
