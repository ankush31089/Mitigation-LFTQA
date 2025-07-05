"""
OpenAI structured output call utility for G-Eval detection.
"""

import logging
import time
from openai import OpenAI
from typing import Type, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from g_eval.helpers.schemas import AnswerRewrite

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY not set in environment")
    raise ValueError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=api_key)

def call_openai_structured(
    prompt: str,
    schema: Type[BaseModel],
    field: str,
    model: str = "gpt-4o",
    max_retries: int = 5,
    temperature: float = 0.0
) -> int:
    """
    Return a 1–5 score using OpenAI structured output mode.
    """
    messages = [
        {"role": "system", "content": "You are a helpful evaluator."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
                temperature=temperature
            )
            score = getattr(response.choices[0].message.parsed, field)
            print(f"Response: {score} for prompt: {prompt}")
            return score
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(
                f"Attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait}s…"
            )
            time.sleep(wait)
    raise RuntimeError(f"OpenAI structured call failed after {max_retries} retries.")

def call_openai_mitigation(prompt: str, model: str = "gpt-4o", temperature: float = 0.0, max_retries: int = 20) -> Optional[str]:
    """
    Calls OpenAI for mitigation and returns the revised answer string, or None if failed.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            parsed = AnswerRewrite.model_validate_json(content)
            return parsed.answer.strip()
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"[Retry {attempt}/{max_retries}] OpenAI error: {e} — waiting {wait}s")
            time.sleep(wait)
    return None 