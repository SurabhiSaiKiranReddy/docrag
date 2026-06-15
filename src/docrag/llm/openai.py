"""OpenAI LLM provider (``gpt-4o-mini`` by default) with streaming support."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

from .base import LLM

if TYPE_CHECKING:
    from openai import Stream
    from openai.types.chat import ChatCompletionChunk


class OpenAILLM(LLM):
    """Chat-completion client backed by the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OpenAI LLM selected but no API key provided. "
                "Set OPENAI_API_KEY in your environment or .env file."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key)

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        stream = cast(
            "Stream[ChatCompletionChunk]",
            self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            ),
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token

    def complete(self, prompt: str, system: str | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
