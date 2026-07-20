"""
Thin wrapper around the GitHub Models API (OpenAI-compatible endpoint)
used by agents to turn structured findings into human-readable rule
proposals / narratives.

If GITHUB_TOKEN is not set, falls back to deterministic templated text
so the whole pipeline still runs end-to-end without network access — useful
for local demos and offline grading.
"""
from src.config import GITHUB_TOKEN, GITHUB_MODEL, GITHUB_MODELS_URL

_client = None
if GITHUB_TOKEN:
    try:
        from openai import OpenAI
        _client = OpenAI(base_url=GITHUB_MODELS_URL, api_key=GITHUB_TOKEN)
    except Exception:
        _client = None


def llm_complete(system: str, prompt: str, max_tokens: int = 600, fallback: str = "") -> str:
    """Call GitHub Models API if configured; otherwise return the provided fallback text."""
    if _client is None:
        return fallback
    try:
        resp = _client.chat.completions.create(
            model=GITHUB_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip() or fallback
    except Exception:
        return fallback


def is_llm_enabled() -> bool:
    return _client is not None
