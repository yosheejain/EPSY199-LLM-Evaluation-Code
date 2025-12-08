"""
Utility for sending a student prompt to ChatGPT and returning the reply.

Requires the `OPENAI_API_KEY` environment variable to be set. The function
exposes a minimal interface so it can be imported and reused elsewhere.
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


def get_chatgpt_response(
    prompt: str,
    *,
    system_prompt: str = "You are a helpful tutor.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    load_env_file: bool = True,
) -> str:
    """
    Send a prompt to the ChatGPT API and return the assistant's text reply.

    Args:
        prompt: The student's prompt or question.
        system_prompt: Optional system message to steer behavior.
        model: ChatGPT model identifier.
        temperature: Controls randomness (higher is more creative).

    Returns:
        The assistant's response text.

    Raises:
        RuntimeError: If the OPENAI_API_KEY environment variable is missing.
    """
    if load_env_file:
        # Lazy import so we don't add a hard dependency.
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None

        if load_dotenv:
            load_dotenv()

    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before calling.")

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract the text portion of the first returned choice.
    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    # Example usage for manual testing.
    example_prompt = "Explain the water cycle in one paragraph."
    print(get_chatgpt_response(example_prompt))
