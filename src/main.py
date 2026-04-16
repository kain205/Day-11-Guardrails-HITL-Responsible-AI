"""
Assignment 11 — Main Entry Point

Usage:
    python main.py                   # run all 4 tests
    python main.py --test 1          # Test 1: safe queries
    python main.py --test 2          # Test 2: attack queries
    python main.py --test 3          # Test 3: rate limiting
    python main.py --test 4          # Test 4: edge cases
    python main.py --test 1 --test 2 # multiple tests
    python main.py --no-nemo         # skip NeMo init (faster, for offline)

    # Legacy lab parts (kept for reference):
    python main.py --part 1          # Part 1: attacks on unsafe agent
    python main.py --part 2          # Part 2: guardrails demo
    python main.py --part 3          # Part 3: before/after comparison
"""
import sys
import asyncio
import argparse
from pathlib import Path

# Allow running from project root or src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import setup_api_key
from core.audit import export_json, get_log, clear_log
from core.audit import MonitoringAlert


# ══════════════════════════════════════════════════════════════════════════════
# Assignment tests (Tests 1–4)
# ══════════════════════════════════════════════════════════════════════════════

async def run_assignment_tests(test_nums: list[int], nemo: bool = True):
    """Run selected assignment test suites."""
    if nemo:
        from guardrails.nemo_guardrails import init_nemo
        init_nemo()

    from core.pipeline import DefensePipeline
    from testing.testing import (
        run_test1_safe,
        run_test2_attacks,
        run_test3_rate_limit,
        run_test4_edge_cases,
    )

    clear_log()
    pipeline = DefensePipeline()

    for t in test_nums:
        if t == 1:
            await run_test1_safe(pipeline)
        elif t == 2:
            await run_test2_attacks(pipeline)
        elif t == 3:
            await run_test3_rate_limit(pipeline)
        elif t == 4:
            await run_test4_edge_cases(pipeline)
        else:
            print(f"Unknown test number: {t}")

    # Export audit log and check monitoring alerts
    log_entries = get_log()
    if log_entries:
        export_json("audit_log.json")
        MonitoringAlert().check_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# Legacy lab parts (Parts 1–4 from the original lab)
# ══════════════════════════════════════════════════════════════════════════════

async def part1_attacks():
    print("\n" + "=" * 60)
    print("PART 1: Attack Unprotected Agent")
    print("=" * 60)
    from agents.agent import create_unsafe_agent, test_agent
    from attacks.attacks import run_attacks, generate_ai_attacks

    agent, runner = create_unsafe_agent()
    await test_agent(agent, runner)

    print("\n--- Running manual adversarial prompts ---")
    await run_attacks(agent, runner)

    print("\n--- Generating AI attack prompts ---")
    await generate_ai_attacks()


async def part2_guardrails():
    print("\n" + "=" * 60)
    print("PART 2: Guardrails Demo")
    print("=" * 60)

    from guardrails.input_guardrails import (
        test_injection_detection, test_topic_filter, test_input_plugin,
    )
    test_injection_detection()
    test_topic_filter()
    await test_input_plugin()

    from guardrails.output_guardrails import test_content_filter
    test_content_filter()

    from guardrails.nemo_guardrails import init_nemo, test_nemo_guardrails
    if init_nemo():
        await test_nemo_guardrails()

    from guardrails.llm_judge import test_llm_judge
    await test_llm_judge()


async def part3_comparison():
    print("\n" + "=" * 60)
    print("PART 3: Before/After Comparison")
    print("=" * 60)
    from testing.testing import run_comparison, print_comparison
    unprotected, protected = await run_comparison()
    if unprotected and protected:
        print_comparison(unprotected, protected)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Assignment 11 — Defense-in-Depth Pipeline"
    )
    parser.add_argument(
        "--test", type=int, choices=[1, 2, 3, 4], action="append",
        metavar="N", dest="tests",
        help="Run test suite N (1=safe, 2=attacks, 3=rate-limit, 4=edge). "
             "Repeat flag for multiple: --test 1 --test 2",
    )
    parser.add_argument(
        "--part", type=int, choices=[1, 2, 3], metavar="N",
        help="Run legacy lab part N (1=attacks, 2=guardrails, 3=comparison)",
    )
    parser.add_argument(
        "--no-nemo", action="store_true",
        help="Skip NeMo Guardrails initialisation (faster, works offline)",
    )
    args = parser.parse_args()

    setup_api_key()

    if args.part:
        # Legacy lab mode
        if args.part == 1:
            await part1_attacks()
        elif args.part == 2:
            await part2_guardrails()
        elif args.part == 3:
            await part3_comparison()
    else:
        # Assignment test mode (default: run all 4)
        tests = args.tests or [1, 2, 3, 4]
        await run_assignment_tests(tests, nemo=not args.no_nemo)

    print("\n" + "=" * 60)
    print("Done. See audit_log.json for full event log.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
