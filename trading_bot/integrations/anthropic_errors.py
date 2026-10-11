"""Plain-language explanations for Anthropic API failures shown in Telegram."""

from __future__ import annotations


def explain_api_failure(exc: BaseException) -> str | None:
    """Return a short, user-facing reason for account-level API failures, else None.

    Only account or key problems the owner has to fix are recognised (billing, bad key,
    rate limit, overload). Anything else returns None so callers keep their own message.
    """
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if "credit balance" in text:
        return (
            "The Anthropic API key has no credit left. Add credits under Plans & Billing in the "
            "Anthropic console, then try again. The / commands still work (/help)."
        )
    if status in (401, 403) or "invalid x-api-key" in text or "authentication" in text:
        return (
            "Anthropic rejected the API key. Check ANTHROPIC_API_KEY on Railway. "
            "The / commands still work (/help)."
        )
    if status == 429:
        return "Anthropic is rate limiting this key right now. Wait a minute and try again."
    if status in (500, 502, 503, 529) or "overloaded" in text:
        return "Anthropic is overloaded right now. Try again in a minute."
    return None
