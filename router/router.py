from __future__ import annotations
import logging
import yaml
from classifier.models import ClassificationResult, ModelTier, RoutingDecision

logger = logging.getLogger(__name__)

TIER_THRESHOLDS = {
    ModelTier.CHEAP:    (0.0,  0.35),
    ModelTier.BALANCED: (0.36, 0.70),
    ModelTier.CAPABLE:  (0.71, 1.0),
}


class Router:

    def __init__(
        self,
        config_path: str = "config/pricing.yaml",
        caller_policies: dict[str, ModelTier] | None = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._tier_models = cfg["tiers"]
        self._pricing     = cfg["models"]
        self._default     = cfg["default"]
        self._policies    = caller_policies or {}

    def route(
        self,
        classification: ClassificationResult,
        requested_model: str,
        caller_id: str,
        tier_header: str | None = None,
    ) -> RoutingDecision:

        tier, override_reason = self._pick_tier(
            classification, caller_id, tier_header
        )

        routed_model = self._tier_models[tier.value].get(
            "anthropic", requested_model
        )

        cost_saved = self._estimate_saving(
            requested_model, routed_model,
            classification.estimated_output_tokens
        )

        logger.info(
            "Routing caller=%s from=%s to=%s tier=%s override=%s score=%.2f saved=$%.5f",
            caller_id, requested_model, routed_model,
            tier.value, override_reason,
            classification.complexity_score, cost_saved,
        )

        return RoutingDecision(
            original_model=requested_model,
            routed_model=routed_model,
            tier=tier,
            provider=self._infer_provider(routed_model),
            override_reason=override_reason,
            classification=classification,
            estimated_cost_saved=cost_saved,
        )
    
    @staticmethod
    def _infer_provider(model_name: str) -> str:
        if model_name.startswith("claude"):
            return "anthropic"
        if model_name.startswith(("gpt-", "o1", "o3")):
            return "openai"
        if model_name.startswith("gemini"):
            return "google"
        return "anthropic"   # safe default

    def _pick_tier(
        self,
        classification: ClassificationResult,
        caller_id: str,
        tier_header: str | None,
    ) -> tuple[ModelTier, str | None]:

        # 1. Header override — caller forces a tier
        if tier_header:
            try:
                return ModelTier(tier_header.lower()), "header"
            except ValueError:
                logger.warning("Invalid X-Model-Tier header: %s", tier_header)

        # 2. Caller policy ceiling from SQLite
        ceiling = self._policies.get(caller_id)
        auto_tier = self._score_to_tier(classification.complexity_score)

        if ceiling and self._rank(auto_tier) > self._rank(ceiling):
            return ceiling, "caller_policy"

        # 3. Default — score based
        return auto_tier, None

    @staticmethod
    def _score_to_tier(score: float) -> ModelTier:
        for tier, (lo, hi) in TIER_THRESHOLDS.items():
            if lo <= score <= hi:
                return tier
        return ModelTier.CAPABLE

    @staticmethod
    def _rank(tier: ModelTier) -> int:
        return {
            ModelTier.CHEAP:    0,
            ModelTier.BALANCED: 1,
            ModelTier.CAPABLE:  2,
        }[tier]

    def _estimate_saving(
        self,
        original: str,
        routed: str,
        output_tokens: int,
    ) -> float:
        orig_rate   = self._pricing.get(original, self._default)["output"]
        routed_rate = self._pricing.get(routed,   self._default)["output"]
        saved_per_million = max(0.0, orig_rate - routed_rate)
        return round(saved_per_million * output_tokens / 1_000_000, 6)
    
    def calculate_actual_saving(
        self,
        original: str,
        routed: str,
        actual_output_tokens: int,
    ) -> float:
        return self._estimate_saving(original, routed, actual_output_tokens)