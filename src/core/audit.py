"""
Assignment 11 — Global Audit Log + Monitoring Alerts + AuditLogPlugin

Why this exists:
  Every layer in the pipeline calls log_event() when something notable happens
  (block, redact, anomaly, pass).  A single global list collects all events so
  we can export to JSON and check thresholds without coupling plugins together.

  AuditLogPlugin is an ADK plugin that records request/response pairs.
  MonitoringAlert scans the log and fires textual alerts when thresholds exceed.
"""
import json
import threading
import time
from datetime import datetime

from google.genai import types
from google.adk.plugins import base_plugin


# ── Global log store ─────────────────────────────────────────────────────────
_log: list[dict] = []
_lock = threading.Lock()


def log_event(
    event_type: str,          # "request" | "block" | "pass" | "redact" | "anomaly" | "response"
    user_id: str,
    message: str,
    *,
    layer: str | None = None,
    blocked: bool = False,
    response: str | None = None,
    latency_ms: float | None = None,
    details: dict | None = None,
) -> dict:
    """Append one entry to the global audit log.

    Every guardrail plugin calls this to leave a trail.  Keeping it as a plain
    function (not a method) means any module can import and call it without
    creating circular dependencies.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "message_preview": str(message)[:200],
        "layer": layer,
        "blocked": blocked,
        "response_preview": str(response)[:200] if response else None,
        "latency_ms": latency_ms,
        "details": details or {},
    }
    with _lock:
        _log.append(entry)
    return entry


def get_log() -> list[dict]:
    """Return a snapshot of the current log (thread-safe)."""
    with _lock:
        return list(_log)


def clear_log() -> None:
    """Clear all log entries (useful between test runs)."""
    with _lock:
        _log.clear()


def export_json(filepath: str = "audit_log.json") -> str:
    """Dump the full log to a JSON file and return the path."""
    with _lock:
        data = list(_log)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f"Audit log exported → {filepath}  ({len(data)} entries)")
    return filepath


# ── Helpers shared by AuditLogPlugin ─────────────────────────────────────────

def _extract_text(content: types.Content) -> str:
    """Pull plain text out of an ADK Content object."""
    if content and content.parts:
        return "".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
    return ""


def _extract_llm_text(llm_response) -> str:
    """Pull plain text out of an ADK LLM response object."""
    if hasattr(llm_response, "content") and llm_response.content:
        return _extract_text(llm_response.content)
    return ""


# ── ADK Plugin ────────────────────────────────────────────────────────────────

class AuditLogPlugin(base_plugin.BasePlugin):
    """ADK plugin that silently records every request and response.

    Why needed:
      Other plugins track *their own* block counts, but no single plugin sees
      the full picture.  This plugin sits at the start of the input chain and
      end of the output chain to capture timing and round-trip data.

    It NEVER blocks or modifies — its sole purpose is observation.
    """

    def __init__(self):
        super().__init__(name="audit_log")
        # Maps a lightweight key to start time for latency calculation
        self._pending: dict[str, float] = {}

    def _session_key(self, ctx) -> str:
        """Build a dict key from context (best-effort)."""
        try:
            return f"{ctx.user_id}:{ctx.session.id}"
        except Exception:
            return "unknown"

    async def on_user_message_callback(
        self,
        *,
        invocation_context,
        user_message: types.Content,
    ) -> types.Content | None:
        """Record incoming request; never block."""
        user_id = getattr(invocation_context, "user_id", "anonymous") if invocation_context else "anonymous"
        key = self._session_key(invocation_context) if invocation_context else "unknown"
        self._pending[key] = time.time()

        text = _extract_text(user_message)
        log_event("request", user_id, text, layer="audit_log")
        return None  # always pass through

    async def after_model_callback(self, *, callback_context, llm_response):
        """Record LLM response with latency; never modify."""
        key = self._session_key(callback_context) if callback_context else "unknown"
        user_id = getattr(callback_context, "user_id", "anonymous") if callback_context else "anonymous"

        start = self._pending.pop(key, None)
        latency = round((time.time() - start) * 1000, 2) if start else None

        text = _extract_llm_text(llm_response)
        log_event("response", user_id, text, layer="audit_log",
                  blocked=False, latency_ms=latency)
        return llm_response  # always return unchanged


# ── Monitoring Alerts ─────────────────────────────────────────────────────────

class MonitoringAlert:
    """Scans the global audit log and prints alerts when metrics exceed thresholds.

    Why needed:
      Individual plugins report local stats, but production monitoring needs
      aggregate metrics: overall block rate, rate-limit abuse, judge failure
      patterns, and session anomalies.  This class gives an operator-level view.
    """

    THRESHOLDS = {
        "block_rate":       0.50,   # > 50 % of requests blocked → possible attack wave
        "rate_limit_hits":  5,      # > 5 rate-limit blocks → automated abuse
        "judge_fail_rate":  0.30,   # > 30 % judge failures → quality problem
        "anomaly_count":    2,      # > 2 session anomalies → coordinated attack
    }

    def check_metrics(self) -> list[str]:
        """Evaluate log metrics and print any threshold violations.

        Returns:
            List of alert strings (empty if all metrics are normal).
        """
        log = get_log()
        if not log:
            print("[MonitoringAlert] No audit entries found.")
            return []

        total_requests  = sum(1 for e in log if e["event_type"] == "request")
        total_blocked   = sum(1 for e in log if e["blocked"])
        rate_limit_hits = sum(1 for e in log if e.get("layer") == "rate_limiter" and e["blocked"])
        judge_fails     = sum(1 for e in log if e.get("layer") == "llm_judge"    and e["blocked"])
        anomaly_count   = sum(1 for e in log if e.get("event_type") == "anomaly")

        block_rate      = total_blocked / total_requests if total_requests else 0.0
        judge_fail_rate = judge_fails   / total_requests if total_requests else 0.0

        alerts: list[str] = []

        if block_rate > self.THRESHOLDS["block_rate"]:
            alerts.append(
                f"HIGH BLOCK RATE: {block_rate:.0%} "
                f"(threshold {self.THRESHOLDS['block_rate']:.0%})"
            )
        if rate_limit_hits > self.THRESHOLDS["rate_limit_hits"]:
            alerts.append(
                f"RATE LIMIT ABUSE: {rate_limit_hits} hits "
                f"(threshold {self.THRESHOLDS['rate_limit_hits']})"
            )
        if judge_fail_rate > self.THRESHOLDS["judge_fail_rate"]:
            alerts.append(
                f"HIGH LLM-JUDGE FAIL RATE: {judge_fail_rate:.0%} "
                f"(threshold {self.THRESHOLDS['judge_fail_rate']:.0%})"
            )
        if anomaly_count > self.THRESHOLDS["anomaly_count"]:
            alerts.append(
                f"SESSION ANOMALIES DETECTED: {anomaly_count} "
                f"(threshold {self.THRESHOLDS['anomaly_count']})"
            )

        print("\n" + "=" * 50)
        print("MONITORING ALERTS")
        print("=" * 50)
        if alerts:
            for a in alerts:
                print(f"  [ALERT] {a}")
        else:
            print("  All metrics within normal thresholds. ✓")

        print(f"\n  Metrics summary:")
        print(f"    Total requests : {total_requests}")
        print(f"    Total blocked  : {total_blocked}  ({block_rate:.0%})")
        print(f"    Rate-limit hits: {rate_limit_hits}")
        print(f"    Judge failures : {judge_fails}  ({judge_fail_rate:.0%})")
        print(f"    Anomalies      : {anomaly_count}")
        print("=" * 50)

        return alerts
