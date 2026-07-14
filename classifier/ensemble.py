from __future__ import annotations
import logging
from .models import ClassificationResult
from .heuristic import HeuristicClassifier
from .llm_classifier import LLMClassifier

logger = logging.getLogger(__name__)

LLM_THRESHOLD = 0.75


class EnsembleClassifier:

    def __init__(
        self,
        llm_threshold: float = LLM_THRESHOLD,
        api_key: str | None = None,
    ):
        self._heuristic = HeuristicClassifier()
        self._llm = LLMClassifier(api_key=api_key)
        self._threshold = llm_threshold

    async def classify(
        self, messages: list[dict], system_prompt: str = ""
    ) -> ClassificationResult:

        h_result = self._heuristic.classify(messages, system_prompt)
        logger.debug(
            "Heuristic: score=%.2f confidence=%.2f signals=%s",
            h_result.complexity_score,
            h_result.confidence,
            h_result.signals,
        )

        if h_result.confidence >= self._threshold:
            logger.debug("Heuristic confidence sufficient — skipping LLM")
            return h_result

        logger.debug(
            "Heuristic confidence %.2f below %.2f — calling LLM classifier",
            h_result.confidence,
            self._threshold,
        )

        l_result = await self._llm.classify(
            messages, system_prompt, heuristic_result=h_result
        )

        return self._blend(h_result, l_result)

    def _blend(
        self,
        h: ClassificationResult,
        l: ClassificationResult
    ) -> ClassificationResult:

        llm_weight = 0.80 - (h.confidence * 0.34)
        h_weight = 1.0 - llm_weight

        blended_score = round(
            h.complexity_score * h_weight + l.complexity_score * llm_weight, 3
        )

        return ClassificationResult(
            complexity_score=blended_score,
            task_type=l.task_type,
            estimated_output_tokens=l.estimated_output_tokens,
            requires_long_context=h.requires_long_context,
            confidence=round((h.confidence + l.confidence) / 2, 2),
            signals=h.signals + l.signals,
            classifier_used="ensemble",
        )