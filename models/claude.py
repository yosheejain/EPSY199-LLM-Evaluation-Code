"""
Utility for sending a student prompt to Claude and returning the reply.

Requires the `CLAUDE_API_KEY` environment variable to be set (Anthropic key).
Optionally loads a local `.env` file when available.
"""

from __future__ import annotations

import os
from typing import Optional


def get_claude_response(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful tutor.",
    model: str = "claude-3-5-sonnet-20240620",
    max_tokens: int = 512,
    temperature: float = 0.7,
    load_env_file: bool = True,
) -> str:
    """
    Send a prompt to the Claude API and return the assistant's text reply.

    Args:
        prompt: The student's prompt or question.
        system_prompt: Optional system message to steer behavior.
        model: Claude model identifier.
        max_tokens: Maximum tokens to generate in the reply.
        temperature: Controls randomness (higher is more creative).
        load_env_file: Attempt to load a local .env file before reading env vars.

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If the CLAUDE_API_KEY environment variable is missing.
    """
    # Local import to avoid hard dependency when Claude is not used.
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package is not installed. Install it with `pip install anthropic` "
            "or omit Claude from the --tutor-models/--judge-model list."
        ) from exc

    if load_env_file:
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv:
            load_dotenv()

    api_key: Optional[str] = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("CLAUDE_API_KEY is not set. Export it before calling.")

    client = Anthropic(api_key=api_key)

    # Anthropomorphic chat format uses a single "messages" list.
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )

    # Return concatenated text from the first message output.
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


if __name__ == "__main__":
    # Example usage for manual testing.
    example_prompt = "Explain the water cycle in one paragraph."
    print(get_claude_response(example_prompt))
