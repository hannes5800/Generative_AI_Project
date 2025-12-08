from __future__ import annotations

from typing import Optional

try:
    from openai import APIError, APITimeoutError, OpenAI
except ImportError:  # OpenAI SDK not installed
    APIError = APITimeoutError = None  # type: ignore
    OpenAI = None  # type: ignore


def generate_tokenomics_image(prompt: str) -> Optional[str]:
    """
    Generate a tokenomics diagram image URL using the OpenAI DALL-E 3 API.

    Args:
        prompt: Natural language description of the desired tokenomics diagram.

    Returns:
        The image URL if generation succeeds; otherwise None.
    """
    if not prompt or not prompt.strip():
        print("[tokenomics-image] Prompt is required.")
        return None

    if OpenAI is None:
        print("[tokenomics-image] OpenAI SDK is not installed. Please install `openai`.")
        return None

    client = OpenAI()

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        if not getattr(response, "data", None):
            print("[tokenomics-image] No image data returned by OpenAI.")
            return None
        return response.data[0].url
    except (APITimeoutError, APIError) as exc:
        print(f"[tokenomics-image] OpenAI API error: {exc}")
    except Exception as exc:
        print(f"[tokenomics-image] Unexpected error: {exc}")
    return None
