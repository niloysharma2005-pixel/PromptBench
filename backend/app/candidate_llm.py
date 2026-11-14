import logging
from backend.app.config import GEMINI_API_KEY, RUN_LLM_TESTS
from backend.app.llm_client import get_llm_client

logger = logging.getLogger("candidate_llm")

CANDIDATE_SYSTEM_PROMPT = """You are a helpful AI assistant available to the candidate
for this case study. You have access only to the case task and context provided below.
You do NOT have access to any rubric, scoring criteria, or ground-truth answer.
Respond naturally and helpfully to the candidate's questions, as a real-world AI
assistant would in a business setting.

Case Task:
{prompt_text}

Case Context:
{context_data}
"""

def call_helper_ai(prompt_text: str, context_data: str, history: list) -> str:
    """
    history format: [{"role": "user"|"assistant", "content": "..."}]
    """
    # If in test/mock mode or no API key, return a generic mock response utilizing prompt_text
    if not RUN_LLM_TESTS or not GEMINI_API_KEY:
        last_query = history[-1]["content"] if history else "your initial analysis"
        prompt_snippet = prompt_text[:80].strip() if prompt_text else "the case objectives"
        return (
            f"Mocked Helper AI response for case '{prompt_snippet}': "
            f"Regarding {last_query[:60]}, let's break down the available data and evaluate the key constraints."
        )

    system_prompt = CANDIDATE_SYSTEM_PROMPT.format(
        prompt_text=prompt_text,
        context_data=context_data
    )
    
    try:
        client = get_llm_client()
        return client.generate(
            system=system_prompt,
            messages=history,
            temperature=0.7,
            max_tokens=1000
        )
    except Exception as e:
        logger.error(f"Helper AI call failed: {e}")
        raise RuntimeError(f"Helper AI API call failed: {str(e)}")
