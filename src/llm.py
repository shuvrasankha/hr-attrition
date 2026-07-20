"""
Thin wrapper around the HuggingFace Inference API used by agents to turn
structured findings into human-readable rule proposals / narratives.

If HUGGINGFACE_TOKEN is not set, falls back to deterministic templated text
so the whole pipeline still runs end-to-end without network access — useful
for local demos and offline grading.
"""
from src.config import HF_TOKEN, HF_MODEL

_client = None
if HF_TOKEN:
    try:
        from huggingface_hub import InferenceClient
        _client = InferenceClient(token=HF_TOKEN)
    except Exception:
        _client = None


def llm_complete(system: str, prompt: str, max_tokens: int = 600, fallback: str = "") -> str:
    """Call HuggingFace model if configured; otherwise return the provided fallback text."""
    if _client is None:
        return fallback
    try:
        full_prompt = f"<|system|>\n{system}\n<|end|>\n<|user|>\n{prompt}\n<|end|>\n<|assistant|>"
        resp = _client.text_generation(
            full_prompt,
            model=HF_MODEL,
            max_new_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.strip() or fallback
    except Exception:
        return fallback


def is_llm_enabled() -> bool:
    return _client is not None
