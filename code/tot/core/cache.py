"""
Disk-backed cache for LLM calls.

ToT calls the same evaluation prompts repeatedly during development. Caching
makes iteration affordable. Keyed on (model, prompt, temperature, n).
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class LLMCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, prompt: str, temperature: float, n: int) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        h.update(b"\x00")
        h.update(f"{temperature:.4f}".encode())
        h.update(b"\x00")
        h.update(str(n).encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        # Shard into subdirs to keep filesystem happy
        return self.cache_dir / key[:2] / f"{key}.json"

    def get(self, model: str, prompt: str, temperature: float, n: int) -> Optional[dict]:
        path = self._path(self._key(model, prompt, temperature, n))
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, model: str, prompt: str, temperature: float, n: int, value: dict) -> None:
        path = self._path(self._key(model, prompt, temperature, n))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f)

    def clear(self) -> int:
        """Remove all cached entries. Returns count of files removed."""
        count = 0
        for p in self.cache_dir.rglob("*.json"):
            p.unlink()
            count += 1
        return count