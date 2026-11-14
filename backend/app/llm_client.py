import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from backend.app.config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("llm_client")

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

class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.model = model or GEMINI_MODEL
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

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""

_default_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client

def set_llm_client(client: Optional[LLMClient]) -> None:
    global _default_client
    _default_client = client
