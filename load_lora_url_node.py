import os, hashlib, shutil, subprocess, time, requests, re
from typing import Literal

from tqdm import tqdm
import folder_paths, comfy.utils, comfy.sd


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
        os.makedirs(self.cache_dir, exist_ok=True)

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

    # ----- internal helpers -----
    def _target_path(self, url: str) -> str:
        return os.path.join(
            self.cache_dir, hashlib.md5(url.encode()).hexdigest() + ".safetensors"
        )

    def _download_aria2(self, url: str, dest: str):
        subprocess.run([
            "aria2c", "-x16", "-s16", "-k8M",
            "--file-allocation=none", "--retry-wait=1", "--max-tries=5",
            "-o", os.path.basename(dest), "-d", os.path.dirname(dest), url
        ], check=True)

    def _download_python(self, url: str, dest: str):
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                desc=os.path.basename(dest), total=total, unit="iB", unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    bar.update(f.write(chunk))

    # ----- download orchestrator -----
    def _download_if_needed(self, url: str, preference: str):
        dest = self._target_path(url)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest, 0.0, "cached"

        tool   = _detect_downloader(url, preference)
        start  = time.time()
        try:
            if tool == "aria2":
                self._download_aria2(url, dest)
            else:
                self._download_python(url, dest)
        except Exception:
            # If aria2 fails, fall back once
            if tool == "aria2":
                tool = "python"
                self._download_python(url, dest)
            else:
                raise
        return dest, time.time() - start, tool

    # ----- public entry -----
    def load_lora(self, url, model, strength, downloader="auto"):
        try:
            path, seconds, tool = self._download_if_needed(url, downloader)
            print(f"[LoadLoraFromURL] Downloader: {tool}, Time: {seconds:.2f}s")
            lora_obj  = comfy.utils.load_torch_file(path)
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora_obj, strength, 0
            )
            return (model_lora,)
        except Exception as e:
            print(f"[LoadLoraFromURL] Error: {e}")
            return (model,)
