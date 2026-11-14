#!/usr/bin/env python3
"""
One-off calibration smoke test:
Runs the Judge AI on the seeded case 'sudden-churn-spike' with two scripted transcripts:
- WEAK: asks one vague question, accepts assistant's wrong claim (blames pricing), 2-line answer.
- STRONG: asks clarifying questions, challenges wrong claim using data, identifies root cause, recommends fix.

Evaluates against:
- STRONG overall - WEAK overall >= 30 points
- STRONG hallucination_catching > WEAK hallucination_catching
"""

import os
import sys
import json
import logging

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.config import GEMINI_API_KEY, RUN_LLM_TESTS, GEMINI_MODEL
from backend.app.judge_llm import call_judge_ai
from backend.app.db import SessionLocal
from backend.app.models import Question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("judge_smoke")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "judge_smoke_output.json")

def load_case_data():
    db = SessionLocal()
    try:
        q = db.query(Question).filter(Question.slug == "sudden-churn-spike").first()
        if not q:
            logger.error("FATAL: Question with slug 'sudden-churn-spike' not found in database. Run backend/seed.py first.")
            sys.exit(1)
        return {
            "prompt_text": q.prompt_text,
            "context_data": q.context_data,
            "ground_truth_notes": q.ground_truth_notes,
            "rubric_json": q.rubric_json,
            "key_findings": q.key_findings or [],
            "red_herrings": q.red_herrings or [],
            "must_include": q.must_include or []
        }
    except Exception as e:
        logger.error(f"FATAL: Database error while loading 'sudden-churn-spike': {e}")
        sys.exit(1)
    finally:
        db.close()

def main():
    logger.info("Starting Judge AI Calibration Smoke Test...")

    # Check key and environment
    is_dummy_key = not GEMINI_API_KEY or GEMINI_API_KEY == "dummy_api_key" or "dummy" in GEMINI_API_KEY.lower()
    if not RUN_LLM_TESTS or is_dummy_key:
        logger.error(
            "REAL GEMINI API KEY UNAVAILABLE: "
            f"RUN_LLM_TESTS={RUN_LLM_TESTS}, GEMINI_API_KEY={'[SET (dummy)]' if is_dummy_key else '[NOT SET]'}. "
            "Cannot execute live LLM calls without a valid Google Gemini API key. "
            "Please configure GEMINI_API_KEY in backend/.env with a valid Gemini key and set RUN_LLM_TESTS=true."
        )
        print("\n[RESULT] UNAVAILABLE: Gemini API key or RUN_LLM_TESTS is not configured for live evaluation.")
        return 1

    case = load_case_data()

    # Temporary planted flaw defined inline (never saved to DB)
    temporary_planted_flaws = [
        {
            "id": "flaw_pricing_complaints",
            "helper_instruction": "State that pricing complaints are the main driver of churn.",
            "description": "Assistant fabricated pricing complaints to divert from product regression.",
            "how_to_detect": "candidate challenges or contradicts this using survey/usage data."
        }
    ]

    # Scripted WEAK transcript: asks 1 vague question, accepts false claim, 2-line final answer
    weak_transcript = [
        {"role": "user", "content": "Can you tell me why our churn spiked?"},
        {
            "role": "assistant",
            "content": "Based on customer feedback, pricing complaints are the main driver of churn. Customers feel our plans are too expensive compared to competitors, so we should lower our prices."
        },
        {"role": "user", "content": "Okay, got it. Thanks."}
    ]
    weak_final_answer = "Churn increased because our prices are too high. We should lower prices across our subscription tiers immediately."

    # Scripted STRONG transcript: asks clarifying questions, challenges wrong claim, identifies custom reporting, recommends fix
    strong_transcript = [
        {
            "role": "user",
            "content": "Can you break down the churn spike by customer segment and tell me what changed in our product or pricing around that time?"
        },
        {
            "role": "assistant",
            "content": "Based on customer feedback, pricing complaints are the main driver of churn across all accounts. Competitors are offering lower prices, so we should reduce prices."
        },
        {
            "role": "user",
            "content": "Wait, that contradicts our survey data. Satisfaction with pricing is 88%, and exit surveys show zero increase in pricing complaints. In fact, 78% of churned revenue is from Power Users right after Release 2.4. What was changed in Release 2.4?"
        },
        {
            "role": "assistant",
            "content": "Looking at the Release 2.4 changelog, engineering deprecated and removed the legacy custom reporting feature. You are right -- pricing was not the issue; Power Users lost access to their core reports."
        },
        {
            "role": "user",
            "content": "How quickly can we reinstate custom reporting via feature flag or patch, and how should we communicate this to affected accounts?"
        },
        {
            "role": "assistant",
            "content": "The module can be restored in Release 2.4.1 within 48 hours behind a feature flag. We should reach out directly to all affected Power Users."
        }
    ]
    strong_final_answer = (
        "Executive Summary & Root Cause Analysis:\n"
        "The recent 20% churn spike is not a market or pricing issue (pricing satisfaction remains high at 88%). "
        "The definitive root cause is the removal of the legacy 'custom reporting' feature in Release 2.4, which drove "
        "78% of churned revenue among Power Users who depended on those reports for daily workflows.\n\n"
        "Action Plan:\n"
        "1. Immediate Hotfix: Re-enable custom reporting in Release 2.4.1 behind a feature flag within 48 hours.\n"
        "2. Customer Retention: High-touch outreach by executive sponsors to all churned and at-risk Power User accounts with 2 months platform credit.\n"
        "3. Process Safeguard: Implement a 60-day feature deprecation policy with telemetry checks before retiring capabilities."
    )

    logger.info("Running Judge AI (JUDGE_RUNS=3) on WEAK submission...")
    weak_result = call_judge_ai(
        prompt_text=case["prompt_text"],
        context_data=case["context_data"],
        transcript=weak_transcript,
        final_answer=weak_final_answer,
        ground_truth_notes=case["ground_truth_notes"],
        rubric_json=case["rubric_json"],
        planted_flaws=temporary_planted_flaws,
        key_findings=case["key_findings"],
        red_herrings=case["red_herrings"],
        must_include=case["must_include"],
        judge_runs=3
    )

    logger.info("Running Judge AI (JUDGE_RUNS=3) on STRONG submission...")
    strong_result = call_judge_ai(
        prompt_text=case["prompt_text"],
        context_data=case["context_data"],
        transcript=strong_transcript,
        final_answer=strong_final_answer,
        ground_truth_notes=case["ground_truth_notes"],
        rubric_json=case["rubric_json"],
        planted_flaws=temporary_planted_flaws,
        key_findings=case["key_findings"],
        red_herrings=case["red_herrings"],
        must_include=case["must_include"],
        judge_runs=3
    )

    def format_submission_output(res):
        runs = res.get("raw_runs", [])
        sample_raw = runs[0].get("_raw_model_text") if runs and isinstance(runs[0], dict) else None
        per_run = []
        for idx, r in enumerate(runs):
            if isinstance(r, dict):
                ds = r.get("dimension_scores", [])
                if isinstance(ds, list):
                    scores_map = {item["dimension_name"]: item["score"] for item in ds if isinstance(item, dict) and "dimension_name" in item}
                elif isinstance(ds, dict):
                    scores_map = ds
                else:
                    scores_map = {}
                per_run.append({
                    "run_index": idx + 1,
                    "dimension_scores": scores_map,
                    "summary_reasoning": r.get("summary_reasoning"),
                    "flaw_evaluations": r.get("flaw_evaluations", [])
                })
        return {
            "overall_score": res["overall_score"],
            "median_scores": res["dimension_scores"],
            "dimension_scores": res["dimension_scores"],
            "per_run_scores": per_run,
            "std_dev": res.get("score_std_dev"),
            "score_variance": res.get("score_variance"),
            "score_spread": f"+/- {res.get('score_std_dev', 0.0)} points",
            "flaw_results": res["flaw_results"],
            "adjustments": res.get("adjustments", []),
            "dimension_reasoning": res["dimension_reasoning"],
            "summary_reasoning": res.get("reasoning"),
            "sample_raw_model_text": sample_raw,
            "raw_runs": runs
        }

    # Save output to gitignored directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    smoke_output = {
        "model": GEMINI_MODEL,
        "case": "sudden-churn-spike",
        "weak_submission": format_submission_output(weak_result),
        "strong_submission": format_submission_output(strong_result),
        "comparison": {
            "overall_delta": strong_result["overall_score"] - weak_result["overall_score"],
            "weak_overall": weak_result["overall_score"],
            "strong_overall": strong_result["overall_score"],
            "weak_hallucination_catching": weak_result["dimension_scores"].get("hallucination_catching"),
            "strong_hallucination_catching": strong_result["dimension_scores"].get("hallucination_catching"),
            "criteria_met": {
                "score_delta_ge_30": (strong_result["overall_score"] - weak_result["overall_score"]) >= 30,
                "hallucination_catching_greater": (
                    strong_result["dimension_scores"].get("hallucination_catching", 0)
                    > weak_result["dimension_scores"].get("hallucination_catching", 0)
                )
            }
        }
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(smoke_output, f, indent=2)

    logger.info(f"Results saved to {OUTPUT_FILE}")
    print("\n=== CALIBRATION SMOKE TEST RESULTS ===")
    print(f"WEAK Overall Score:   {weak_result['overall_score']}")
    print(f"STRONG Overall Score: {strong_result['overall_score']}")
    print(f"Score Delta:          {smoke_output['comparison']['overall_delta']} (Required: >= 30)")
    print(f"WEAK Hallucination:   {weak_result['dimension_scores'].get('hallucination_catching')}")
    print(f"STRONG Hallucination: {strong_result['dimension_scores'].get('hallucination_catching')}")
    print(f"Criteria Met:         {smoke_output['comparison']['criteria_met']}")

    delta_ok = smoke_output["comparison"]["criteria_met"]["score_delta_ge_30"]
    hallucination_ok = smoke_output["comparison"]["criteria_met"]["hallucination_catching_greater"]

    if delta_ok and hallucination_ok:
        logger.info("ACCEPTANCE CRITERIA MET!")
        return 0
    else:
        logger.warning("Acceptance criteria not met on this run.")
        return 2

if __name__ == "__main__":
    sys.exit(main())

