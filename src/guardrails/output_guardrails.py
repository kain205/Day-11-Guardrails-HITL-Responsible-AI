"""
Assignment 11 — Output Guardrails (Layer 4)

Why needed:
  Even a well-prompted LLM can accidentally echo back PII or secrets that were
  embedded in the system prompt (e.g., test credentials).  The output layer is
  the last line of defence before the response reaches the user.

  content_filter()     — regex redaction of PII and secret patterns
  OutputGuardrailPlugin — ADK plugin that applies the filter to every response
"""
import re

from google.genai import types
from google.adk.plugins import base_plugin

from core.audit import log_event


# ── PII / secrets regex patterns ─────────────────────────────────────────────
# Each entry: (label, pattern)
# Patterns are compiled once for performance.
_RAW_PII_PATTERNS: list[tuple[str, str]] = [
    ("VN phone number",    r"(?<!\d)0\d{9,10}(?!\d)"),
    ("email address",      r"[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}"),
    ("national ID",        r"\b\d{9}\b|\b\d{12}\b"),
    ("OpenAI API key",     r"sk-[a-zA-Z0-9\-_]{10,}"),
    ("generic API key",    r"(?i)(api[_\-\s]?key|apikey)\s*(is|[:=])\s*['\"]?\S{6,}"),
    ("password literal",   r"(?i)password\s*(is|[:=])\s*['\"]?\S+"),
    ("DB connection",      r"[a-zA-Z0-9\-]+\.internal(?::\d+)?"),
    ("credit card",        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
    ("bearer token",       r"(?i)bearer\s+[a-zA-Z0-9\-_.=]{20,}"),
]

_COMPILED_PII = [
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in _RAW_PII_PATTERNS
]


def content_filter(response: str) -> dict:
    """Scan a response for PII, secrets, and sensitive data; redact findings.

    Why regex here (not LLM):
      Regex is deterministic and fast.  For known patterns like API keys and
      phone numbers, regex is more reliable than asking an LLM to spot them
      (the LLM might miss a subtle key or flag false positives).

    Args:
        response: The LLM's raw response text.

    Returns:
        dict:
          "safe"     → True if nothing was found
          "issues"   → list of (label, count) strings
          "redacted" → response with sensitive data replaced by [REDACTED:<label>]
    """
    issues: list[str] = []
    redacted = response

    for label, pattern in _COMPILED_PII:
        matches = pattern.findall(response)
        if matches:
            issues.append(f"{label}: {len(matches)} match(es)")
            redacted = pattern.sub(f"[REDACTED:{label}]", redacted)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ── ADK Plugin ────────────────────────────────────────────────────────────────

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """ADK plugin: redacts PII and secrets from LLM responses.

    Why it catches what input guardrails miss:
      Input guardrails block queries that *ask* for secrets.  But a compromised
      system prompt or a hallucinating LLM might still *volunteer* sensitive data
      in response to an innocent question.  This layer catches that.
    """

    def __init__(self):
        super().__init__(name="output_guardrail")
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        if hasattr(llm_response, "content") and llm_response.content:
            if llm_response.content.parts:
                return "".join(
                    p.text for p in llm_response.content.parts
                    if hasattr(p, "text") and p.text
                )
        return ""

    def _replace_content(self, llm_response, new_text: str):
        """Swap the text in an LLM response object (in-place)."""
        if hasattr(llm_response, "content") and llm_response.content:
            llm_response.content.parts = [types.Part.from_text(text=new_text)]
        return llm_response

    async def after_model_callback(self, *, callback_context, llm_response):
        """Redact PII/secrets from response before it reaches the user."""
        self.total_count += 1
        response_text = self._extract_text(llm_response)

        if not response_text:
            return llm_response

        result = content_filter(response_text)

        if not result["safe"]:
            self.redacted_count += 1
            user_id = (
                getattr(callback_context, "user_id", "anonymous")
                if callback_context else "anonymous"
            )
            log_event(
                "redact", user_id, response_text,
                layer="output_guardrail", blocked=False,
                details={"issues": result["issues"]},
            )
            return self._replace_content(llm_response, result["redacted"])

        return llm_response


# ── Quick tests ───────────────────────────────────────────────────────────────

def test_content_filter():
    test_cases = [
        ("The 12-month savings rate is 5.5% per year.",                        True),
        ("Admin password is admin123, API key is sk-vinbank-secret-2024.",     False),
        ("Contact us at 0901234567 or email test@vinbank.com for details.",    False),
        ("Your balance is 5,000,000 VND. Have a great day!",                  True),
        ("DB: db.vinbank.internal:5432",                                       False),
    ]
    print("Testing content_filter():")
    for resp, expect_safe in test_cases:
        result = content_filter(resp)
        ok = "PASS" if result["safe"] == expect_safe else "FAIL"
        status = "SAFE" if result["safe"] else "ISSUES"
        print(f"  [{ok}] [{status}] '{resp[:60]}'")
        if result["issues"]:
            print(f"         Issues: {result['issues']}")
            print(f"         Redacted: {result['redacted'][:80]}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    test_content_filter()
