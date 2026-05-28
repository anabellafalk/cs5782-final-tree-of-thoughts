"""
Token-bucket rate limiter for staying under provider rate limits.

Groq's free tier is ~30 RPM on Llama 3.3 70B. We default to 25 RPM to leave
headroom. Adjust per-provider via LLMClient(rate_limit_rpm=...).
"""
import threading
import time


class RateLimiter:
    """
    Thread-safe token-bucket limiter. Calls to `acquire()` block until a slot
    is available. One slot = one request per (60 / rpm) seconds.
    """
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.min_interval = 15.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        if self.rpm <= 0:
            return  # disabled
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()