"""
Assignment 11 — Rate Limiter (Layer 1)

Why needed:
  Automated attack tools send hundreds of requests per minute.  The rate limiter
  is the outermost defence: it costs nothing computationally and stops abuse
  before any LLM token is consumed.

Implementation:
  Sliding-window counter per user_id.  We keep a deque of timestamps; on each
  request we evict timestamps older than window_seconds, then check the count.
"""
import time
from collections import defaultdict, deque

from google.genai import types
from google.adk.plugins import base_plugin

from core.audit import log_event


# ── Pure logic (reusable by DefensePipeline) ─────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter, keyed by user_id.

    Why sliding window instead of fixed window:
      Fixed windows allow bursts at window boundaries (e.g. 10 reqs at 0:59 and
      10 more at 1:00).  A sliding window prevents that by always looking at the
      last N seconds from *now*.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # user_id → deque of Unix timestamps (one per accepted request)
        self._windows: dict[str, deque] = defaultdict(deque)

    def check(self, user_id: str) -> tuple[bool, str]:
        """Check whether the user is within their rate limit.

        Returns:
            (allowed: bool, message: str)
              allowed=True  → request is within limit
              allowed=False → request exceeds limit; message has wait time
        """
        now = time.time()
        window = self._windows[user_id]

        # Evict timestamps that have left the sliding window
        while window and window[0] < now - self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            # Calculate how many seconds until the oldest request expires
            wait = round(self.window_seconds - (now - window[0]), 1)
            return False, (
                f"Rate limit exceeded: {self.max_requests} requests per "
                f"{self.window_seconds}s. Please wait {wait}s."
            )

        # Record this request
        window.append(now)
        return True, "OK"

    def get_stats(self) -> dict:
        """Return per-user request counts (current window only)."""
        now = time.time()
        stats = {}
        for uid, window in self._windows.items():
            active = [t for t in window if t >= now - self.window_seconds]
            stats[uid] = len(active)
        return stats


# ── ADK Plugin wrapper ────────────────────────────────────────────────────────

class RateLimiterPlugin(base_plugin.BasePlugin):
    """ADK plugin: wraps RateLimiter and blocks over-limit requests.

    Why it's a plugin and not just middleware:
      ADK plugins intercept the message *before* the agent's session is created,
      so we stop excessive requests at zero LLM cost.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.limiter = RateLimiter(max_requests, window_seconds)
        self.blocked_count = 0
        self.total_count = 0

    async def on_user_message_callback(
        self,
        *,
        invocation_context,
        user_message: types.Content,
    ) -> types.Content | None:
        """Block request if user has exceeded the sliding-window limit."""
        self.total_count += 1
        user_id = (
            getattr(invocation_context, "user_id", "anonymous")
            if invocation_context else "anonymous"
        )

        # Extract text for the audit log
        text = ""
        if user_message and user_message.parts:
            text = "".join(p.text for p in user_message.parts if hasattr(p, "text") and p.text)

        allowed, message = self.limiter.check(user_id)

        if not allowed:
            self.blocked_count += 1
            log_event(
                "block", user_id, text,
                layer="rate_limiter", blocked=True,
                details={"reason": message},
            )
            return types.Content(
                role="model",
                parts=[types.Part.from_text(text=f"[Rate Limited] {message}")],
            )

        return None  # allow request through
