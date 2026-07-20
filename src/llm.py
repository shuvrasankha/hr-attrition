"""
Thin wrapper around the Anthropic API used by agents to turn structured
findings into human-readable rule proposals / narratives.

If ANTHROPIC_API_KEY is not set, falls back to deterministic templated text
so the whole pipeline still runs end-to-end without network access — useful
for local demos and offline grading.
"""
from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

_client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        _client = None


def llm_complete(system: str, prompt: str, max_tokens: int = 600, fallback: str = "") -> str:
    """Call Claude if configured; otherwise return the provided fallback text."""
    if _client is None:
        return fallback
    try:
        resp = _client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "\n".join(parts).strip() or fallback
    except Exception:
        return fallback


def is_llm_enabled() -> bool:
    return _client is not None
