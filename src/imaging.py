from __future__ import annotations
from typing import Optional
from openai import APIError, APITimeoutError, OpenAI
from IPython.display import Image, display


def load_api_key_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()

def generate_tokenomics_image(prompt: str) -> Optional[str]:
    """
    Generate a tokenomics diagram image URL using the OpenAI DALL-E 3 API.

    Args:
        prompt: Natural language description of the desired tokenomics diagram.

    Returns:
        The image URL if generation succeeds; otherwise None.
    """
    if OpenAI is None:
        print("[tokenomics-image] OpenAI SDK is not installed. Please install `openai`.")
        return None
    
    api_key = load_api_key_from_file("OpenAI_API.txt")
    client = OpenAI(api_key=api_key)

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        if not getattr(response, "data", None):
            print("No image data returned by OpenAI.")
            return None
        return response.data[0].url
    except (APITimeoutError, APIError) as exc:
        print(f"OpenAI API error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
    return None
