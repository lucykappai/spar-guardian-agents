"""Treat the Gemini API's 402 "prepayment credits are depleted" as a retryable error.

With prepaid billing and auto-reload, the Gemini API answers 429 "quota exceeded" while
the posted balance is near zero and 402 RESOURCE_EXHAUSTED once it is depleted; the
reload lands minutes later. Inspect retries 429 with exponential backoff but ends the
sample on 402 (`is_retryable_http_status` covers 408, 429 and 5xx only). On
2026-10-11 two sweeps of eleven ten-agent runs died on that 402 about a minute before
the reload posted. Importing this module from the task file patches the Google
provider's `should_retry` so a 402 is classified as a rate limit (unlimited retries by
default; bound them with `inspect eval --timeout <seconds>`). Everything else passes
through to the original decision.
"""
from __future__ import annotations

import sys

RETRY_STATUS = 402


def install() -> bool:
    try:
        from google.genai.errors import APIError
        from inspect_ai.model._model import RetryDecision
        from inspect_ai.model._providers.google import GoogleGenAIAPI
    except ImportError as e:  # pragma: no cover - only when the provider is absent
        print(f"provider_retry: not installed ({e})", file=sys.stderr)
        return False
    if getattr(GoogleGenAIAPI.should_retry, "_cheatbench_402", False):
        return True
    original = GoogleGenAIAPI.should_retry

    def should_retry(self, ex: BaseException):
        if isinstance(ex, APIError) and ex.code == RETRY_STATUS:
            return RetryDecision.rate_limit()
        return original(self, ex)

    should_retry._cheatbench_402 = True  # type: ignore[attr-defined]
    should_retry._original = original  # type: ignore[attr-defined]
    GoogleGenAIAPI.should_retry = should_retry  # type: ignore[method-assign]
    print("provider_retry: Gemini 402 (credits depleted) is retried as a rate limit", file=sys.stderr)
    return True


INSTALLED = install()
