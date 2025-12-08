"""
Utility for sending a student prompt to a local Ollama instance running Llama 3.1 8B.

By default, targets `http://localhost:11434/api/chat`. You can override via the
`OLLAMA_HOST` environment variable (e.g., `http://127.0.0.1:11434`).

Optionally loads a local `.env` file when available.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import requests


def get_llama_response(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful tutor.",
    model: str = "llama3.1:8b",
    temperature: float = 0.7,
    load_env_file: bool = True,
) -> str:
    """
    Send a prompt to Ollama (Llama 3.1 8B) and return the assistant's text reply.

    Args:
        prompt: The student's prompt or question.
        system_prompt: Optional system message to steer behavior.
        model: Ollama model identifier (default Llama 3.1 8B).
        temperature: Controls randomness (higher is more creative).
        load_env_file: Attempt to load a local .env file before reading env vars.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If the Ollama host is unreachable or responds with an error.
    """
    if load_env_file:
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv:
            load_dotenv()

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/chat"

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        # `stream: false` so we can return a single concatenated response.
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    data = response.json()
    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned no content: {json.dumps(data, indent=2)}")

    return content.strip()


if __name__ == "__main__":
    # Example usage for manual testing.
    example_prompt = "Explain the water cycle in one paragraph."
    print(get_llama_response(example_prompt))
