import os, hashlib, shutil, subprocess, time, requests, re
from typing import Literal

from tqdm import tqdm
import folder_paths


# ------------------------------------------------------------
# Quick CDN‑cache check (1 HEAD request, 200 ms)
# ------------------------------------------------------------
_CACHE_HIT_RE = re.compile(r"\bHIT\b", re.I)


def edge_is_hot(url: str, timeout: float = 2.0) -> bool:
    """Return True if Front Door says the object is in edge cache."""
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
def detect_downloader(url: str, preference: str = "auto") -> str:
    """
    'auto' = python if edge‑hot, else aria2; otherwise honor explicit choice.
    """
    if preference != "auto":
        return preference

    # Decide based on cache state
    is_hot = edge_is_hot(url)
    if is_hot:
        return "python"  # single stream is fastest on warm cache
    if shutil.which("aria2c"):
        return "aria2"
    return "python"


# ------------------------------------------------------------
# Download utilities
# ------------------------------------------------------------
def download_aria2(url: str, dest: str):
    """Download using aria2c for high-speed parallel downloads."""
    subprocess.run([
        "aria2c", "-x16", "-s16", "-k8M",
        "--file-allocation=none", "--retry-wait=1", "--max-tries=5",
        "-o", os.path.basename(dest), "-d", os.path.dirname(dest), url
    ], check=True)


def download_python(url: str, dest: str):
    """Download using Python requests with progress bar."""
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=os.path.basename(dest), total=total, unit="iB", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                bar.update(f.write(chunk))


def download_if_needed(url: str, dest: str, preference: str = "auto"):
    """Download file if not already cached, return timing info."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0.0, "cached"

    tool = detect_downloader(url, preference)
    start = time.time()
    try:
        if tool == "aria2":
            download_aria2(url, dest)
        else:
            download_python(url, dest)
    except Exception:
        # If aria2 fails, fall back once
        if tool == "aria2":
            tool = "python"
            download_python(url, dest)
        else:
            raise
    return time.time() - start, tool


# ------------------------------------------------------------
# Cache path utilities
# ------------------------------------------------------------
def get_cache_path(url: str, cache_dir: str, extension: str = ".safetensors") -> str:
    """Return deterministic cache path for the URL."""
    return os.path.join(
        cache_dir, hashlib.md5(url.encode()).hexdigest() + extension
    )


def ensure_cache_dir(cache_dir: str):
    """Ensure cache directory exists."""
    os.makedirs(cache_dir, exist_ok=True) 