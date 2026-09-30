import time
from typing import Tuple, Optional
from app.core.config import settings
from app.core.logging import logger
from app.rag.prompts import SYSTEM_RAG_PROMPT, NO_CONTEXT_FALLBACK


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.api_key = settings.LLM_API_KEY
        self.model_name = settings.LLM_MODEL
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key in ("mock_key", "your_api_key_here"):
            logger.info("Using Mock LLM Mode (no valid LLM_API_KEY provided)")
            return

        try:
            if self.provider == "google":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._client = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                )
                logger.info(f"Initialized Google Gemini LLM ({self.model_name})")
            elif self.provider == "openai":
                from langchain_openai import ChatOpenAI
                self._client = ChatOpenAI(
                    model=self.model_name,
                    api_key=self.api_key,
                    temperature=0.0,
                )
                logger.info(f"Initialized OpenAI LLM ({self.model_name})")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM provider '{self.provider}': {e}. Falling back to Mock LLM.")
            self._client = None

    def generate_answer(self, question: str, formatted_context: str) -> Tuple[str, float]:
        start_time = time.perf_counter()

        if not formatted_context.strip():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return NO_CONTEXT_FALLBACK, round(elapsed_ms, 2)

        prompt = SYSTEM_RAG_PROMPT.format(context=formatted_context, question=question)

        if self._client is not None:
            try:
                response = self._client.invoke(prompt)
                answer = response.content.strip()
            except Exception as e:
                logger.error(f"LLM API call failed: {e}. Generating fallback response.")
                answer = f"Error generating answer from LLM: {str(e)}"
        else:
            # Simulated / Mock LLM response grounded in context for offline testing
            answer = f"Based on the provided documents: {formatted_context.split('Content:')[-1].strip()}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return answer, round(elapsed_ms, 2)
