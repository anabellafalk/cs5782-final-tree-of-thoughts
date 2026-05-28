"""
Unified LLM client supporting multiple providers (OpenAI, Anthropic, Together).

All tasks call through this module so model swaps are a one-line config change.
Handles retries, structured logging, and integrates with the cache + cost tracker.
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import backoff
from dotenv import load_dotenv

from .cache import LLMCache
from .cost_tracker import CostTracker
from .rate_limiter import RateLimiter

load_dotenv()
logger = logging.getLogger(__name__)


# Pricing per 1M tokens as of early 2026. Update as needed.
# Used by CostTracker for budget estimation.
MODEL_PRICING = {
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Groq examples
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    # Together AI examples
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Llama-3.1-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.88, "output": 0.88},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"input": 0.18, "output": 0.18},
    "deepseek-ai/DeepSeek-V3": {"input": 1.25, "output": 1.25},
    # Groq free tier (logged at $0 — token counts still tracked)
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
    # Ollama (local, free)
    "ollama/llama3.1:70b": {"input": 0.0, "output": 0.0},
    "ollama/llama3.1:8b": {"input": 0.0, "output": 0.0},
    "ollama/llama3.3:70b": {"input": 0.0, "output": 0.0},
}


@dataclass
class LLMResponse:
    """Structured response from any LLM provider."""
    completions: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cached: bool = False
    raw: dict = field(default_factory=dict)


class LLMClient:
    """
    Provider-agnostic LLM client.

    Usage:
        client = LLMClient(model="gpt-4o-mini")
        response = client.generate(prompt="...", temperature=0.7, n=5)
        for completion in response.completions:
            ...
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        cache_dir: Optional[str] = "./.llm_cache",
        cost_log_path: Optional[str] = "./results/cost_log.jsonl",
        provider: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
    ):
        self.model = model
        self.provider = provider or self._infer_provider(model)
        self.cache = LLMCache(cache_dir) if cache_dir else None
        self.cost_tracker = CostTracker(cost_log_path) if cost_log_path else None
        # Sensible per-provider rate limit defaults. Pass rate_limit_rpm=0 to disable.
        # NOTE: Groq's free tier binds on tokens-per-minute (~6000 TPM) much more
        # tightly than requests-per-minute. With ~1500-token few-shot prompts, we
        # can only afford ~4 calls/minute. Set to 4 here; override via config if
        # using shorter prompts or a paid tier.
        if rate_limit_rpm is None:
            rate_limit_rpm = {
                "groq": 4,         # Free tier 6K TPM + ~1.5K token prompts -> ~4 RPM
                "ollama": 0,       # Local — no need to throttle
                "openai": 0,       # Tier 1+ usually has ample headroom
                "anthropic": 0,
                "together": 0,
            }.get(self.provider, 0)
        self.rate_limiter = RateLimiter(rate_limit_rpm)
        self._client = self._init_client()

    def _infer_provider(self, model: str) -> str:
        if model.startswith(("gpt-", "o1-", "o3-")):
            return "openai"
        if model.startswith("claude-"):
            return "anthropic"
        if model.startswith("ollama/"):
            return "ollama"
        # Groq's Llama models use a flat naming scheme with no slash
        if model.startswith(("llama-3", "mixtral-", "gemma-")) and "/" not in model:
            return "groq"
        if "/" in model:  # Together AI uses org/model format
            return "together"
        raise ValueError(f"Cannot infer provider for model {model!r}; pass provider= explicitly.")

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if self.provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if self.provider == "together":
            from openai import OpenAI  # Together is OpenAI-compatible
            return OpenAI(
                api_key=os.environ["TOGETHER_API_KEY"],
                base_url=os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1"),
            )
        if self.provider == "groq":
            from openai import OpenAI  # Groq is OpenAI-compatible
            return OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
        if self.provider == "ollama":
            from openai import OpenAI  # Ollama exposes an OpenAI-compatible endpoint
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OpenAI(api_key="ollama", base_url=base_url)  # api key is unused by ollama
        raise ValueError(f"Unsupported provider: {self.provider}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        max_time=120,
        giveup=lambda e: "invalid_api_key" in str(e).lower(),
    )
    def _call_openai_compatible(self, prompt: str, temperature: float, n: int, max_tokens: int, stop: Optional[list[str]]) -> LLMResponse:
        """Used by OpenAI, Together, Groq, and Ollama (all OpenAI-compatible)."""
        self.rate_limiter.acquire()
        # Ollama doesn't reliably support n>1 in a single request, so loop.
        # Groq supports n but it's safer to loop here for consistency across providers.
        if self.provider in ("ollama", "groq") and n > 1:
            completions, total_in, total_out = [], 0, 0
            for i in range(n):
                if i > 0:  # First iteration's slot already acquired above
                    self.rate_limiter.acquire()
                resp = self._client.chat.completions.create(
                    model=self.model.replace("ollama/", "") if self.provider == "ollama" else self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    n=1,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                completions.append(resp.choices[0].message.content)
                # Ollama may omit usage; default to 0
                if resp.usage is not None:
                    total_in += resp.usage.prompt_tokens
                    total_out += resp.usage.completion_tokens
            cost = self._compute_cost(total_in, total_out)
            return LLMResponse(
                completions=completions,
                model=self.model,
                prompt_tokens=total_in,
                completion_tokens=total_out,
                cost_usd=cost,
            )

        # Standard path (OpenAI, Together) — single call returns n completions
        resp = self._client.chat.completions.create(
            model=self.model.replace("ollama/", "") if self.provider == "ollama" else self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
        )
        completions = [c.message.content for c in resp.choices]
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0
        cost = self._compute_cost(prompt_tokens, completion_tokens)
        return LLMResponse(
            completions=completions,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw={"id": resp.id} if hasattr(resp, "id") else {},
        )

    @backoff.on_exception(backoff.expo, Exception, max_tries=5, max_time=120)
    def _call_anthropic(self, prompt: str, temperature: float, n: int, max_tokens: int, stop: Optional[list[str]]) -> LLMResponse:
        """Anthropic API doesn't support n>1 natively, so we loop."""
        completions = []
        total_in, total_out = 0, 0
        for i in range(n):
            self.rate_limiter.acquire()
            resp = self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=stop,
            )
            completions.append(resp.content[0].text)
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
        cost = self._compute_cost(total_in, total_out)
        return LLMResponse(
            completions=completions,
            model=self.model,
            prompt_tokens=total_in,
            completion_tokens=total_out,
            cost_usd=cost,
        )

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            return 0.0  # Unknown model — log zero rather than crash
        return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        n: int = 1,
        max_tokens: int = 1000,
        stop: Optional[list[str]] = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """
        Generate `n` completions for `prompt`.

        Caching key includes model + prompt + temperature + n. If temperature=0
        and the call is cached, we return the cached result. For temperature>0,
        cache hits are rarer because each call may want fresh samples — but if
        you've already paid for n samples at this prompt, we reuse them.
        """
        # Check cache
        if use_cache and self.cache is not None:
            cached = self.cache.get(self.model, prompt, temperature, n)
            if cached is not None:
                logger.debug(f"Cache hit for prompt[:60]={prompt[:60]!r}")
                return LLMResponse(
                    completions=cached["completions"],
                    model=self.model,
                    prompt_tokens=cached["prompt_tokens"],
                    completion_tokens=cached["completion_tokens"],
                    cost_usd=0.0,  # Cached — no new cost
                    cached=True,
                )

        # Make the actual call
        if self.provider in ("openai", "together", "groq", "ollama"):
            # Ollama does not return token usage by default — handle it specially
            response = self._call_openai_compatible(prompt, temperature, n, max_tokens, stop)
        elif self.provider == "anthropic":
            response = self._call_anthropic(prompt, temperature, n, max_tokens, stop)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Persist to cache + cost log
        if use_cache and self.cache is not None:
            self.cache.set(
                self.model, prompt, temperature, n,
                {
                    "completions": response.completions,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                },
            )
        if self.cost_tracker is not None:
            self.cost_tracker.log(
                model=self.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
            )

        return response
