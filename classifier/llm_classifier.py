from __future__ import annotations
import hashlib
import json
import logging
import os
import httpx
from .models import ClassificationResult, TaskType

logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """You are a request complexity classifier.
Analyse the user request and return ONLY a JSON object — no prose, no markdown.

JSON schema:
{
  "complexity_score": <float 0.0-1.0>,
  "task_type": <"factual"|"code"|"analysis"|"creative"|"math"|"summarize"|"chat">,
  "estimated_output_tokens": <int>,
  "reasoning": "<one sentence>"
}

Scoring guide:
  0.0-0.35  Simple lookups, greetings, yes/no questions
  0.36-0.70 Moderate: code, comparisons, structured summaries
  0.71-1.0  Complex: system design, multi-step reasoning, proofs
"""


class LLMClassifier:

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._cache: dict[str, ClassificationResult] = {}

    async def classify(
        self,
        messages: list[dict],
        system_prompt: str = "",
        heuristic_result: ClassificationResult | None = None,
    ) -> ClassificationResult:

        cache_key = self._hash(messages, system_prompt)

        if cache_key in self._cache:
            logger.debug("LLM classifier cache hit")
            return self._cache[cache_key]

        try:
            result = await self._call(messages, system_prompt)
        except Exception as exc:
            logger.warning("LLM classifier failed (%s), using heuristic", exc)
            if heuristic_result:
                return heuristic_result
            return ClassificationResult(
                complexity_score=0.5,
                task_type=TaskType.CHAT,
                estimated_output_tokens=300,
                requires_long_context=False,
                confidence=0.4,
                signals=["llm_classifier_error_fallback"],
                classifier_used="heuristic_fallback",
            )

        self._cache[cache_key] = result
        return result

    async def _call(
        self, messages: list[dict], system_prompt: str
    ) -> ClassificationResult:

        user_content = self._build_input(messages, system_prompt)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 256,
                    "system": CLASSIFIER_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_content}],
                },
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"]

        data = json.loads(raw)

        return ClassificationResult(
            complexity_score=float(data["complexity_score"]),
            task_type=TaskType(data.get("task_type", "chat")),
            estimated_output_tokens=int(data.get("estimated_output_tokens", 300)),
            requires_long_context=False,
            confidence=0.90,
            signals=[f"llm: {data.get('reasoning', '')}"],
            classifier_used="llm",
        )

    def _build_input(self, messages: list[dict], system_prompt: str) -> str:
        parts = []
        if system_prompt:
            parts.append(f"[SYSTEM]: {system_prompt[:500]}")
        for msg in messages[-3:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            parts.append(f"[{role.upper()}]: {content[:800]}")
        return "\n\n".join(parts)[:2000]

    @staticmethod
    def _hash(messages: list[dict], system_prompt: str) -> str:
        payload = json.dumps(
            {"system": system_prompt, "messages": messages},
            sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()