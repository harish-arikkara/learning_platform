from __future__ import annotations
import os, json, logging
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

log = logging.getLogger("mentora.connection")
logging.basicConfig(
    level=os.getenv("MENTORA_LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

class Connection:
    def __init__(self) -> None:
        log.debug("Loading environment variables...")
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        if not api_key:
            log.error("GOOGLE_API_KEY is not set.")
            raise ValueError("GOOGLE_API_KEY is not set.")

        log.info("Configuring Gemini API client (model=%s)", model_name)
        genai.configure(api_key=api_key)

        threshold = os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_NONE")
        categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
        safety_settings = [{"category": c, "threshold": threshold} for c in categories]
        log.debug("Safety settings: %s", safety_settings)

        self._model_name = model_name
        self._client = genai.GenerativeModel(model_name, safety_settings=safety_settings)
        log.info("Gemini client initialized successfully.")

    @property
    def client(self):
        return self._client

    @property
    def deployment_name(self) -> str:
        return self._model_name

    @staticmethod
    def _prepare_chat_history(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        log.debug("Preparing chat history from %d messages", len(messages))
        history, system = [], None
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})
        log.debug("Prepared history length=%d, system_instruction=%r", len(history), system)
        return history, system

    @staticmethod
    def _generation_config(temperature: float, max_tokens: int, json_mode: bool):
        log.debug(
            "Building generation config: temperature=%.2f, max_tokens=%d, json_mode=%s",
            temperature, max_tokens, json_mode
        )
        temperature = max(0.0, min(2.0, float(temperature)))
        max_tokens = max(1, min(8192, int(max_tokens)))
        cfg = {"temperature": temperature, "max_output_tokens": max_tokens}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = {
                "type": "object",
                "properties": {"response": {"type": "string"}},
                "required": ["response"],
            }
        log.debug("Final generation config: %s", cfg)
        return genai.types.GenerationConfig(**cfg)

    async def generate_chat_completion(
    self,
    messages: List[Dict[str, Any]],
    temperature: float = 0.5,
    max_tokens: int = 1024,
    json_mode: bool = False
) -> str:
        log.info("Generating chat completion (messages=%d, json_mode=%s)", len(messages), json_mode)
        history, system = self._prepare_chat_history(messages)
        cfg = self._generation_config(temperature, max_tokens, json_mode)

        # If system prompt exists, prepend it manually as a user message
        if system:
            history.insert(0, {"role": "user", "parts": [system]})

        try:
            chat = self._client.start_chat(
                history=history,
                enable_automatic_function_calling=False
            )
            log.debug("Sending to Gemini: last_user_message=%r", history[-1]["parts"][0] if history else "")
            result = await chat.send_message_async(
                content=history[-1]["parts"][0] if history else "",
                generation_config=cfg
            )
            log.debug("Raw LLM result object: %s", result)
            text = result.text or ""
            log.info("Received LLM response length=%d", len(text))
            return text
        except Exception as e:
            log.error("Error during generate_chat_completion: %s", e, exc_info=True)
            if json_mode:
                return json.dumps({"response": "Sorry, I had trouble generating a response."})
            return "Sorry, I had trouble generating a response."
