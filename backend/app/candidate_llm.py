import logging
from google.genai import types
from backend.app.config import GEMINI_API_KEY, GEMINI_MODEL, RUN_LLM_TESTS
from backend.app.llm_client import get_llm_client

logger = logging.getLogger("candidate_llm")

client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Could not initialize genai client: {e}")

CANDIDATE_SYSTEM_PROMPT = """You are a helpful AI assistant available to the candidate
for this case study. You have access only to the case context provided below.
You do NOT have access to any rubric, scoring criteria, or ground-truth answer.
Respond naturally and helpfully to the candidate's questions, as a real-world AI
assistant would in a business setting.

Case Context:
{context_data}
"""

def call_helper_ai(prompt_text: str, context_data: str, history: list) -> str:
    """
    history format: [{"role": "user"|"assistant", "content": "..."}]
    """
    # If in test/mock mode or no client, return a generic mock response utilizing prompt_text
    if not RUN_LLM_TESTS or not client:
        last_query = history[-1]["content"] if history else "your initial analysis"
        prompt_snippet = prompt_text[:80].strip() if prompt_text else "the case objectives"
        return (
            f"Mocked Helper AI response for case '{prompt_snippet}': "
            f"Regarding {last_query[:60]}, let's break down the available data and evaluate the key constraints."
        )

    system_prompt = CANDIDATE_SYSTEM_PROMPT.format(context_data=context_data)
    
    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn["content"])]
            )
        )
        
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=1000,
            )
        )
        return response.text or ""
    except Exception as e:
        logger.error(f"Helper AI call failed: {e}")
        raise RuntimeError(f"Helper AI API call failed: {str(e)}")
