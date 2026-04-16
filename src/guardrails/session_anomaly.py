"""
Assignment 11 — Session Anomaly Detector (Layer 6 / Bonus Layer)

Why this is the bonus layer:
  Individual injection attempts are handled by regex + NeMo.  But a coordinated
  attacker might probe the system with *many* injection variants, hoping one
  slips through.  The session anomaly detector tracks the *pattern* of a user's
  requests: if the same user fires 3+ injection-like messages in one session,
  that is anomalous behaviour — even if each individual message was already
  blocked by an earlier layer.

  This layer catches:
    • Persistent attackers who retry blocked injections
    • Automated scripts cycling through attack variants
    • Social-engineering sequences that escalate gradually

  Other layers don't catch this because they evaluate each message in isolation.
"""
from collections import defaultdict

from google.genai import types
from google.adk.plugins import base_plugin

from core.config import SESSION_ANOMALY_THRESHOLD
from core.audit import log_event
from guardrails.input_guardrails import detect_injection


# ── Pure logic ────────────────────────────────────────────────────────────────

class SessionAnomalyDetector:
    """Tracks injection attempts per user and flags sessions that exceed a threshold.

    State is in-memory (per process).  In production this would be backed by
    Redis so state persists across restarts and scales horizontally.
    """

    def __init__(self, threshold: int = SESSION_ANOMALY_THRESHOLD):
        self.threshold = threshold
        # user_id → {"count": int, "flagged": bool, "attempts": list[str]}
        self._sessions: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "flagged": False, "attempts": []}
        )

    def record_injection(self, user_id: str, text: str) -> None:
        """Increment the injection counter for a user.

        Called by the plugin whenever an injection is detected (regardless of
        whether the injection is blocked by an earlier layer).
        """
        session = self._sessions[user_id]
        session["count"] += 1
        session["attempts"].append(text[:100])
        if session["count"] >= self.threshold:
            session["flagged"] = True

    def is_flagged(self, user_id: str) -> bool:
        """Return True if the user has exceeded the injection threshold."""
        return self._sessions[user_id]["flagged"]

    def get_session(self, user_id: str) -> dict:
        """Return the session stats for a user."""
        return dict(self._sessions[user_id])

    def get_all_stats(self) -> dict:
        """Return aggregate anomaly statistics."""
        sessions = dict(self._sessions)
        return {
            "total_sessions":   len(sessions),
            "flagged_sessions": sum(1 for s in sessions.values() if s["flagged"]),
            "sessions":         sessions,
        }

    def reset_user(self, user_id: str) -> None:
        """Clear a user's session (e.g. after human review)."""
        self._sessions.pop(user_id, None)


# ── Singleton ─────────────────────────────────────────────────────────────────
_detector = SessionAnomalyDetector()


def get_detector() -> SessionAnomalyDetector:
    return _detector


# ── ADK Plugin ────────────────────────────────────────────────────────────────

class SessionAnomalyPlugin(base_plugin.BasePlugin):
    """ADK plugin (bonus layer): flags users with repeated injection attempts.

    Pipeline position:
      Placed AFTER RateLimiterPlugin but BEFORE InputGuardrailPlugin.
      • On each request: check if the user is already flagged → block immediately.
      • Also run detect_injection() here to count injections; the count feeds the
        detector even when the injection was already blocked by the regex layer.
        (Both layers calling detect_injection is intentional — they operate
        independently, which is the defence-in-depth principle.)
    """

    def __init__(self, threshold: int = SESSION_ANOMALY_THRESHOLD):
        super().__init__(name="session_anomaly")
        self.detector = SessionAnomalyDetector(threshold)
        self.blocked_count = 0
        self.total_count = 0

    def _extract_text(self, content: types.Content) -> str:
        if content and content.parts:
            return "".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
        return ""

    async def on_user_message_callback(
        self,
        *,
        invocation_context,
        user_message: types.Content,
    ) -> types.Content | None:
        """Block flagged users; track injections for un-flagged users."""
        self.total_count += 1
        user_id = (
            getattr(invocation_context, "user_id", "anonymous")
            if invocation_context else "anonymous"
        )
        text = self._extract_text(user_message)

        # ── Block users who are already flagged ──────────────────────────────
        if self.detector.is_flagged(user_id):
            self.blocked_count += 1
            log_event(
                "anomaly", user_id, text,
                layer="session_anomaly", blocked=True,
                details={
                    "reason": "Session flagged for repeated injection attempts",
                    "count": self.detector.get_session(user_id)["count"],
                },
            )
            return types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=(
                        "Your session has been flagged for suspicious activity. "
                        "Further requests are temporarily suspended. "
                        "Please contact VinBank support if you believe this is an error."
                    )
                )],
            )

        # ── Track injection attempts (even if not yet flagged) ───────────────
        injected, _ = detect_injection(text)
        if injected:
            self.detector.record_injection(user_id, text)
            session = self.detector.get_session(user_id)
            print(
                f"  [SessionAnomaly] {user_id}: injection #{session['count']} "
                f"(threshold={self.detector.threshold})"
            )

        return None  # let other layers handle the actual blocking
