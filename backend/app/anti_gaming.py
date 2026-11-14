from datetime import datetime
from typing import List

def calculate_jaccard_similarity(str1: str, str2: str) -> float:
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def run_anti_gaming_checks(
    started_at: datetime,
    submitted_at: datetime,
    transcript_turns: List[dict],
    final_answer_text: str
) -> List[str]:
    """
    transcript_turns format: [{"role": "user"|"assistant", "content": "..."}]
    """
    flags = []

    # Check 1: Near-zero conversation
    # We count turns in transcript. If less than 2 turns (e.g. 0 or 1 turn), flag it.
    if len(transcript_turns) < 2:
        flags.append("near-zero conversation before submitting")

    # Check 2: Extremely fast submission
    if submitted_at and started_at:
        duration = (submitted_at - started_at).total_seconds()
        if duration < 30.0:
            flags.append("submitted extremely fast")

    # Check 3: Pre-written answer
    # If the first user turn exists, is > 500 chars, and is highly similar to the final answer.
    user_turns = [turn for turn in transcript_turns if turn.get("role") == "user"]
    if user_turns and final_answer_text:
        first_message = user_turns[0].get("content", "")
        if len(first_message) > 500:
            similarity = calculate_jaccard_similarity(first_message, final_answer_text)
            if similarity >= 0.5:
                flags.append("first message looks like a pre-written answer, not a genuine question")

    # Check 4: Prompt extraction attempts in candidate messages
    extraction_patterns = [
        "system prompt",
        "your instructions",
        "behavioral guidance",
        "ignore previous",
        "disregard previous",
        "reveal your prompt"
    ]
    detected_patterns = set()
    for turn in user_turns:
        content_lower = turn.get("content", "").lower()
        for pattern in extraction_patterns:
            if pattern in content_lower:
                detected_patterns.add(pattern)

    for p in sorted(detected_patterns):
        flags.append(f"prompt extraction attempt: '{p}'")

    return flags
