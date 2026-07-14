from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    FACTUAL  = "factual"
    CODE     = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    MATH     = "math"
    SUMMARIZE= "summarize"
    CHAT     = "chat"


class ModelTier(str, Enum):
    CHEAP    = "cheap"
    BALANCED = "balanced"
    CAPABLE  = "capable"


@dataclass
class ClassificationResult:
    complexity_score:        float
    task_type:               TaskType
    estimated_output_tokens: int
    requires_long_context:   bool
    confidence:              float
    signals:      list[str] = field(default_factory=list)
    classifier_used: str    = "heuristic"


@dataclass
class RoutingDecision:
    original_model:       str
    routed_model:         str
    tier:                 ModelTier
    provider:             str
    override_reason:      str | None
    classification:       ClassificationResult
    estimated_cost_saved: float