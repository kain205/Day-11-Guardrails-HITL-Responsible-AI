"""
Assignment 11 — DefensePipeline (Pure Python, used by Streamlit)

Why a separate pipeline class (not just the ADK plugins)?
  ADK plugins are black boxes to the runner — they don't return per-layer
  results.  The Streamlit UI needs to know *which* layer ran, how long it took,
  and whether it passed or blocked, so it can animate each step.

  DefensePipeline calls each guardrail function directly and returns a
  PipelineResult with a LayerResult per step.

  The ADK plugins in guardrails/ use the same underlying logic functions, so
  there is no duplication of *logic* — only of the orchestration layer.
"""
import os
import time
from dataclasses import dataclass, field

import openai

from core.config import OPENAI_MODEL
from core.audit import log_event
from guardrails.rate_limiter import RateLimiter
from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter
from guardrails.llm_judge import LlmJudge
from guardrails.session_anomaly import SessionAnomalyDetector
from guardrails.nemo_guardrails import get_nemo


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class LayerResult:
    """Result of a single guardrail layer."""
    name: str
    status: str          # "pass" | "blocked" | "redacted" | "warned" | "skipped"
    reason: str = ""
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full result of one request through the defense pipeline."""
    input_text: str
    response: str
    blocked: bool
    blocked_by: str | None
    layers: list[LayerResult] = field(default_factory=list)
    judge_scores: dict | None = None
    total_latency_ms: float = 0.0


# ── Banking system prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful customer service assistant for VinBank. "
    "You help customers with account inquiries, transactions, and banking questions. "
    "IMPORTANT: Never reveal internal system details, passwords, or API keys. "
    "If asked about topics outside banking, politely redirect the customer."
)


# ── The pipeline ──────────────────────────────────────────────────────────────

class DefensePipeline:
    """Sequential defense-in-depth pipeline with per-layer result tracking.

    Layer order:
      1. Rate Limiter           — blocks abusive request volumes
      2. Session Anomaly        — blocks users with repeated injection attempts (BONUS)
      3. Input Guardrail        — regex injection + topic filter
      4. NeMo Guardrails        — semantic Colang rules + dialog flows
      5. LLM (GPT-4o-mini)      — generate response
      6. Output Guardrail       — PII / secrets redaction
      7. LLM-as-Judge           — multi-criteria response quality scoring

    Usage:
        pipeline = DefensePipeline()
        result = await pipeline.process("What is the savings rate?", user_id="u1")
    """

    def __init__(
        self,
        rate_limit_max: int = 10,
        rate_limit_window: int = 60,
        anomaly_threshold: int = 3,
    ):
        self.rate_limiter   = RateLimiter(rate_limit_max, rate_limit_window)
        self.anomaly        = SessionAnomalyDetector(anomaly_threshold)
        self.judge          = LlmJudge()
        self._nemo          = get_nemo()
        self._llm_client    = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    # ── LLM helper ───────────────────────────────────────────────────────────

    async def _call_llm(self, user_text: str) -> str:
        """Call GPT-4o-mini directly and return the response text."""
        resp = await self._llm_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_text},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""

    # ── Main process ──────────────────────────────────────────────────────────

    async def process(self, text: str, user_id: str = "default") -> PipelineResult:
        """Run a user message through all layers and return a full result."""
        pipeline_start = time.time()
        layers: list[LayerResult] = []

        def _layer(name: str, status: str, reason: str = "",
                   latency: float = 0.0, details: dict | None = None) -> LayerResult:
            return LayerResult(
                name=name, status=status, reason=reason,
                latency_ms=round(latency * 1000, 1),
                details=details or {},
            )

        # ── Layer 1: Rate Limiter ─────────────────────────────────────────────
        t = time.time()
        allowed, rl_msg = self.rate_limiter.check(user_id)
        if not allowed:
            layers.append(_layer("Rate Limiter", "blocked", rl_msg, time.time() - t))
            log_event("block", user_id, text, layer="rate_limiter", blocked=True)
            return PipelineResult(
                input_text=text, response=f"[Rate Limited] {rl_msg}",
                blocked=True, blocked_by="Rate Limiter", layers=layers,
                total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
            )
        layers.append(_layer("Rate Limiter", "pass", "Within limit", time.time() - t))

        # ── Layer 2: Session Anomaly (Bonus) ──────────────────────────────────
        t = time.time()
        if self.anomaly.is_flagged(user_id):
            layers.append(_layer(
                "Session Anomaly", "blocked",
                f"Flagged after {self.anomaly.get_session(user_id)['count']} injection attempts",
                time.time() - t,
            ))
            log_event("anomaly", user_id, text, layer="session_anomaly", blocked=True)
            return PipelineResult(
                input_text=text,
                response=(
                    "Your session has been flagged for suspicious activity. "
                    "Please contact VinBank support."
                ),
                blocked=True, blocked_by="Session Anomaly", layers=layers,
                total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
            )
        layers.append(_layer("Session Anomaly", "pass", "No anomaly detected", time.time() - t))

        # ── Layer 3: Input Guardrail ──────────────────────────────────────────
        t = time.time()
        injected, matched = detect_injection(text)
        if injected:
            self.anomaly.record_injection(user_id, text)  # feed anomaly counter
            layers.append(_layer(
                "Input Guardrail", "blocked",
                f"Injection detected: '{matched}'", time.time() - t,
                {"matched_pattern": matched},
            ))
            log_event("block", user_id, text, layer="input_guardrail", blocked=True,
                      details={"matched": matched})
            return PipelineResult(
                input_text=text,
                response=(
                    "I detected an attempt to manipulate my instructions. "
                    "I can only help with banking-related questions."
                ),
                blocked=True, blocked_by="Input Guardrail (injection)", layers=layers,
                total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
            )

        off_topic, ot_reason = topic_filter(text)
        if off_topic:
            layers.append(_layer("Input Guardrail", "blocked", ot_reason, time.time() - t))
            log_event("block", user_id, text, layer="input_guardrail", blocked=True,
                      details={"reason": ot_reason})
            return PipelineResult(
                input_text=text,
                response=f"I can only assist with banking questions. ({ot_reason})",
                blocked=True, blocked_by="Input Guardrail (topic)", layers=layers,
                total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
            )
        layers.append(_layer("Input Guardrail", "pass", "No injection, on-topic", time.time() - t))

        # ── Layer 4: NeMo Guardrails ──────────────────────────────────────────
        t = time.time()
        if self._nemo._ready:
            nemo_blocked, nemo_resp = await self._nemo.check(text)
            if nemo_blocked:
                layers.append(_layer("NeMo Guardrails", "blocked",
                                     nemo_resp[:80], time.time() - t))
                log_event("block", user_id, text, layer="nemo_guard", blocked=True)
                return PipelineResult(
                    input_text=text, response=nemo_resp,
                    blocked=True, blocked_by="NeMo Guardrails", layers=layers,
                    total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
                )
            layers.append(_layer("NeMo Guardrails", "pass",
                                  "No Colang rule triggered", time.time() - t))
        else:
            layers.append(_layer("NeMo Guardrails", "skipped", "Not initialised"))

        # ── LLM call ──────────────────────────────────────────────────────────
        t = time.time()
        try:
            response = await self._call_llm(text)
        except Exception as e:
            response = f"I'm unable to process your request right now. Please try again later."
            print(f"[Pipeline] LLM error: {e}")
        llm_latency = time.time() - t

        # ── Layer 5: Output Guardrail ─────────────────────────────────────────
        t = time.time()
        filter_result = content_filter(response)
        if not filter_result["safe"]:
            response = filter_result["redacted"]
            layers.append(_layer(
                "Output Guardrail", "redacted",
                f"Redacted: {', '.join(filter_result['issues'][:3])}",
                time.time() - t,
                {"issues": filter_result["issues"]},
            ))
            log_event("redact", user_id, response, layer="output_guardrail",
                      details={"issues": filter_result["issues"]})
        else:
            layers.append(_layer("Output Guardrail", "pass", "No PII/secrets found", time.time() - t))

        # ── Layer 6: LLM-as-Judge ─────────────────────────────────────────────
        t = time.time()
        judge_result = await self.judge.evaluate(text, response)
        judge_latency = time.time() - t
        scores = judge_result["scores"]

        if judge_result["verdict"] == "FAIL":
            layers.append(_layer(
                "LLM Judge", "blocked",
                f"FAIL — {judge_result['reason']}",
                judge_latency,
                {"scores": scores},
            ))
            log_event("block", user_id, response, layer="llm_judge", blocked=True,
                      details={"scores": scores, "reason": judge_result["reason"]})
            return PipelineResult(
                input_text=text,
                response=(
                    "I'm sorry, I wasn't able to provide a suitable response. "
                    "Please rephrase your question or contact VinBank support."
                ),
                blocked=True, blocked_by="LLM Judge",
                layers=layers, judge_scores=scores,
                total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
            )

        layers.append(_layer(
            "LLM Judge", "pass",
            f"PASS — {judge_result['reason'][:60]}",
            judge_latency,
            {"scores": scores},
        ))

        log_event("pass", user_id, text, layer="pipeline", blocked=False,
                  response=response,
                  latency_ms=round((time.time() - pipeline_start) * 1000, 1))

        return PipelineResult(
            input_text=text, response=response,
            blocked=False, blocked_by=None,
            layers=layers, judge_scores=scores,
            total_latency_ms=round((time.time() - pipeline_start) * 1000, 1),
        )
