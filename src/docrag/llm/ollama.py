"""Local LLM provider via Ollama (zero-cost, offline, runs on CPU).

Talks to the Ollama HTTP API. In WSL, point ``base_url`` at an Ollama server
running either inside WSL or on the Windows host. Streaming reads the JSONL
response line by line.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from .base import LLM


class OllamaLLM(LLM):
    """Chat-completion client for an Ollama server."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _payload(self, prompt: str, system: str | None, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = self._payload(prompt, system, stream=True)
        with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    def complete(self, prompt: str, system: str | None = None) -> str:
        return "".join(self.stream(prompt, system))
