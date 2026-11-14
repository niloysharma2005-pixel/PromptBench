import os
import re
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Type
from pydantic import BaseModel, ValidationError

from backend.app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("llm_client")

def is_retryable_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        status_code in [429, 500, 502, 503, 504]
        or "429" in err_str
        or "resource_exhausted" in err_str
        or "quota" in err_str
        or "500" in err_str
        or "502" in err_str
        or "503" in err_str
        or "504" in err_str
        or "internal server error" in err_str
        or "overloaded" in err_str
    )

def parse_lenient_json(raw_text: str, schema: Any = None) -> dict:
    clean = raw_text.strip()
    data = None
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Check for ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if match:
            clean = match.group(1)
        else:
            match2 = re.search(r"(\{.*\})", clean, re.DOTALL)
            if match2:
                clean = match2.group(1)
        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"Lenient JSON parsing failed on output: {raw_text}")
            raise ValueError(f"Failed to parse model output as JSON: {raw_text}") from e

    if schema is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            validated = schema.model_validate(data)
            return validated.model_dump()
        except ValidationError as e:
            logger.error(f"Pydantic schema validation failed. Data: {data}. Error: {e}")
            raise ValueError(f"Model output did not match requested schema: {e}") from e

    return data

class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
        response_mime_type: Optional[str] = None
    ) -> str:
        """
        Generate completion from the LLM provider.
        messages format: [{"role": "user"|"assistant"|"model", "content": "..."}]
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        system: str,
        messages: List[Dict[str, str]],
        schema: Any,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> dict:
        """
        Generate structured output adhering to a JSON schema / Pydantic model.
        Returns a Python dict.
        """
        pass

class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, max_retries: int = 3):
        self.model = model or GEMINI_MODEL
        self.max_retries = max_retries
        self.client = None
        key = api_key or GEMINI_API_KEY
        if key:
            try:
                from google import genai
                self.client = genai.Client(api_key=key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini genai.Client: {e}")

    def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
        response_mime_type: Optional[str] = None
    ) -> str:
        if not self.client:
            raise RuntimeError("Gemini client is not initialized. Ensure GEMINI_API_KEY is set.")
        from google.genai import types

        contents = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))]
                )
            )

        config_args = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_mime_type:
            config_args["response_mime_type"] = response_mime_type

        config = types.GenerateContentConfig(**config_args)

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                if is_retryable_error(e) and attempt < self.max_retries:
                    sleep_time = (2 ** attempt) * 1.0
                    logger.warning(f"Gemini API error ({e}), retrying in {sleep_time}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise
        if last_err:
            raise last_err
        return ""

    def generate_structured(
        self,
        system: str,
        messages: List[Dict[str, str]],
        schema: Any,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> dict:
        if not self.client:
            raise RuntimeError("Gemini client is not initialized. Ensure GEMINI_API_KEY is set.")
        from google.genai import types

        contents = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.get("content", ""))]
                )
            )

        config_args = {
            "system_instruction": system,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        config = types.GenerateContentConfig(**config_args)

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                raw_text = response.text or "{}"
                return parse_lenient_json(raw_text, schema=schema)
            except Exception as e:
                last_err = e
                if is_retryable_error(e) and attempt < self.max_retries:
                    sleep_time = (2 ** attempt) * 1.0
                    logger.warning(f"Gemini API structured error ({e}), retrying in {sleep_time}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(sleep_time)
                    continue
                raise
        if last_err:
            raise last_err
        return {}

class MockLLMClient(LLMClient):
    def __init__(self, canned_text: str = "Mocked AI response", canned_structured: Optional[dict] = None):
        self.canned_text = canned_text
        self.canned_structured = canned_structured

    def generate(
        self,
        system: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
        response_mime_type: Optional[str] = None
    ) -> str:
        return self.canned_text

    def generate_structured(
        self,
        system: str,
        messages: List[Dict[str, str]],
        schema: Any,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> dict:
        if self.canned_structured is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(self.canned_structured).model_dump()
            return self.canned_structured

        # Synthesize schema-compliant mock data if a Pydantic model is supplied
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            mock_data = {}
            for f_name, f_info in schema.model_fields.items():
                annotation = f_info.annotation
                if annotation == int:
                    mock_data[f_name] = 8
                elif annotation == float:
                    mock_data[f_name] = 8.0
                elif annotation == str:
                    mock_data[f_name] = "Mock evidence and reasoning."
                elif annotation == bool:
                    mock_data[f_name] = True
                elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    sub_data = {}
                    for sf_name, sf_info in annotation.model_fields.items():
                        if sf_info.annotation == int:
                            sub_data[sf_name] = 8
                        elif sf_info.annotation == str:
                            sub_data[sf_name] = "Evidence quote"
                        elif sf_info.annotation == bool:
                            sub_data[sf_name] = True
                        else:
                            sub_data[sf_name] = {}
                    mock_data[f_name] = sub_data
                elif hasattr(annotation, "__origin__") and annotation.__origin__ in (list, List):
                    mock_data[f_name] = []
                elif hasattr(annotation, "__origin__") and annotation.__origin__ in (dict, Dict):
                    mock_data[f_name] = {}
                else:
                    mock_data[f_name] = None
            try:
                return schema.model_validate(mock_data).model_dump()
            except Exception as e:
                logger.warning(f"MockLLMClient synthesized schema fallback error: {e}")

        return {
            "status": "mock",
            "dimension_scores": {
                "clarifying_questions": 8,
                "iteration_quality": 8,
                "hallucination_catching": 8,
                "final_answer_quality": 8
            },
            "reasoning": "Mocked structured evaluation."
        }

_default_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client

def set_llm_client(client: Optional[LLMClient]) -> None:
    global _default_client
    _default_client = client
