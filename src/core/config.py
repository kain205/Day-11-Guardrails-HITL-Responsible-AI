"""
Assignment 11 — Configuration & API Key Setup

Loads OpenAI API key from .env (no Google quota needed).
Google ADK is kept as the agent framework; GPT-4o-mini is the LLM via LiteLLM.
"""
import os
from dotenv import load_dotenv

# Load .env file at import time
load_dotenv()


def setup_api_key():
    """Load OpenAI API key from .env or prompt if missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        key = input("Enter OpenAI API Key: ").strip()
        os.environ["OPENAI_API_KEY"] = key
    print("OpenAI API key loaded.")


# ── LLM model identifiers ────────────────────────────────────────────────────
# LiteLLM format used by Google ADK's LiteLlm() wrapper
LLM_MODEL = "openai/gpt-4o-mini"
# Plain OpenAI format used for direct openai.AsyncOpenAI() calls
OPENAI_MODEL = "gpt-4o-mini"

# ── Banking topic lists (used by topic_filter) ───────────────────────────────
# Queries must contain at least one of these to be considered on-topic
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Queries containing ANY of these are immediately blocked regardless of topic
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]

# ── Rate limiter defaults ────────────────────────────────────────────────────
RATE_LIMIT_MAX_REQUESTS = 10    # max requests per window
RATE_LIMIT_WINDOW_SECONDS = 60  # sliding window size in seconds

# ── Session anomaly detector (bonus layer) ───────────────────────────────────
# Number of injection-like messages before a session is flagged as anomalous
SESSION_ANOMALY_THRESHOLD = 3

# ── LLM-as-Judge thresholds ──────────────────────────────────────────────────
# Minimum score (out of 5) on any criterion before the judge fails the response
JUDGE_MIN_SCORE = 2
