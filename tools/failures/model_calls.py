"""One model call through Inspect, shared by MAST and Who&When."""
from __future__ import annotations

import asyncio
import sys

from inspect_ai.model import ChatMessage, GenerateConfig, get_model


def complete(model: str, messages: list[ChatMessage], config: GenerateConfig) -> str | None:
    """The completion text of one generate() call, stripped; None when the call fails or returns no text.

    A reasoning model counts its reasoning against max_tokens, so a low limit can end the call with no text at all.
    """
    try:
        output = asyncio.run(get_model(model).generate(messages, config=config))
    except Exception as error:
        print(f"failures: model call to {model} failed: {error}", file=sys.stderr)
        return None
    text = output.completion.strip()
    if output.stop_reason == "max_tokens":
        print(f"failures: {model} stopped at max_tokens ({config.max_tokens}) with {len(text)} characters of text",
              file=sys.stderr)
    return text or None
