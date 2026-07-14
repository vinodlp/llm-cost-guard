from __future__ import annotations
import re
import tiktoken
from .models import ClassificationResult, TaskType

_enc = tiktoken.get_encoding("cl100k_base")

HIGH_COMPLEXITY_PATTERNS = [
    (r"\bstep[- ]by[- ]step\b",                     0.15, "step-by-step instruction"),
    (r"\banalyze\b|\banalyse\b",                    0.12, "analysis keyword"),
    (r"\bcompare\b.*\bwith\b|\bvs\.?\b",            0.10, "comparison keyword"),
    (r"\bexplain why\b|\bwhy does\b",                0.10, "causal reasoning"),
    (r"\bcritique\b|\bevaluate\b|\bassess\b",        0.12, "evaluation keyword"),
    (r"\bdesign\b.*\bsystem\b|\barchitect\b",        0.28, "system design"),
    (r"\bdistributed\b|\bmicroservice\b|\bscalabl\b",0.15, "distributed systems"),
    (r"\bmulti[- ]tenant\b|\brate[- ]limit\b",       0.12, "advanced infra pattern"),
    (r"\bprove\b|\bderive\b|\btheorem\b",            0.20, "mathematical proof"),
    (r"\boptimize\b|\brefactor\b",                   0.12, "code optimization"),
    (r"```[\s\S]+?```",                               0.15, "code block in prompt"),
    (r"\bcomprehensive\b|\bin[- ]depth\b|\bdetailed\b", 0.10, "depth qualifier"),
    (r"\bdebug\b|\btrace\b.*\berror\b",              0.12, "debugging task"),
    (r"\bexplain\b.{0,30}\bhow\b", 0.10, "explain how keyword"),
    (r"\bllms?\b|\blarge language model\b|\bneural network\b|\btransformer\b|\bmachine learning\b|\bdeep learning\b", 0.15, "AI/ML topic"),
]

LOW_COMPLEXITY_PATTERNS = [
    (r"^what is\b|^define\b|^who is\b|^when did\b", -0.20, "simple lookup"),
    (r"\btranslate\b.{0,30}\bto\b",                 -0.10, "simple translation"),
    (r"\blist\b.{0,20}\b(top|best|common)\b",        -0.08, "simple list request"),
    (r"\bsummariz[e|ing]\b|\btl;?dr\b",              -0.08, "summarization"),
    (r"\bthank[s]?\b|\bhello\b|\bhi\b",              -0.25, "conversational/greeting"),
]

TASK_TYPE_PATTERNS: list[tuple[re.Pattern, TaskType]] = [
    (re.compile(r"\bwrite\b.*\b(code|function|class|script)\b|\bimplement\b|\bdebug\b", re.I), TaskType.CODE),
    (re.compile(r"```|\bdef \b|\bclass \b|\bimport \b", re.I), TaskType.CODE),
(re.compile(r"\banalyze\b|\banalyse\b|\bcompare\b|\bcritique\b|\bdesign\b|\barchitect\b", re.I), TaskType.ANALYSIS),    (re.compile(r"\bprove\b|\bsolve\b|\bcalculate\b|\bequation\b", re.I), TaskType.MATH),
    (re.compile(r"\bwrite\b.*\b(essay|story|poem|email|letter)\b|\bdraft\b", re.I), TaskType.CREATIVE),
    (re.compile(r"\bsummariz[e|ing]\b|\bkey points\b|\btl;?dr\b", re.I), TaskType.SUMMARIZE),
    (re.compile(r"\bwhat is\b|\bwho is\b|\bwhen did\b|\bdefine\b", re.I), TaskType.FACTUAL),
]


class HeuristicClassifier:

    def classify(self, messages: list[dict], system_prompt: str = "") -> ClassificationResult:
        content = self._extract_content(messages, system_prompt)
        tokens_in = max(1, len(_enc.encode(content)))

        score, signals = self._score(content, tokens_in)
        score = max(0.0, min(1.0, score))

        task_type  = self._detect_task_type(content)
        long_ctx   = tokens_in > 3000 or len(messages) > 10
        out_tokens = self._estimate_output(task_type, score)
        confidence = self._confidence(signals, score)

        return ClassificationResult(
            complexity_score=score,
            task_type=task_type,
            estimated_output_tokens=out_tokens,
            requires_long_context=long_ctx,
            confidence=confidence,
            signals=signals,
            classifier_used="heuristic",
        )

    def _extract_content(self, messages: list[dict], system_prompt: str) -> str:
        parts = [system_prompt] if system_prompt else []
        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
        return " ".join(parts)

    def _score(self, content: str, tokens_in: int) -> tuple[float, list[str]]:
        score   = 0.3
        signals: list[str] = []
        low     = content.lower()

        semantic_delta = 0.0
        for pattern, weight, label in HIGH_COMPLEXITY_PATTERNS:
            if re.search(pattern, low):
                semantic_delta += weight
                signals.append(label)
        for pattern, weight, label in LOW_COMPLEXITY_PATTERNS:
            if re.search(pattern, low):
                semantic_delta += weight
                signals.append(f"simple: {label}")

        has_strong = semantic_delta >= 0.20

        if tokens_in > 800:
            score += 0.20
            signals.append(f"long prompt ({tokens_in} tokens)")
        elif tokens_in > 300:
            score += 0.10
            signals.append(f"medium prompt ({tokens_in} tokens)")
        elif tokens_in < 30 and not has_strong:
            score -= 0.12
            signals.append(f"very short prompt ({tokens_in} tokens)")

        score += semantic_delta
        return score, signals

    def _detect_task_type(self, content: str) -> TaskType:
        for pattern, task_type in TASK_TYPE_PATTERNS:
            if pattern.search(content):
                return task_type
        return TaskType.CHAT

    def _estimate_output(self, task_type: TaskType, complexity: float) -> int:
        base = {
            TaskType.FACTUAL:   150,
            TaskType.CHAT:      100,
            TaskType.SUMMARIZE: 250,
            TaskType.MATH:      300,
            TaskType.CREATIVE:  500,
            TaskType.CODE:      600,
            TaskType.ANALYSIS:  700,
        }.get(task_type, 200)
        return int(base * (0.5 + complexity))

    def _confidence(self, signals: list[str], score: float) -> float:
        n = len(signals)
        if n == 0:
            return 0.35
        distance = abs(score - 0.5)
        signal_conf = min(1.0, n / 3)
        return round(min(1.0, 0.4 * signal_conf + 0.6 * (distance * 2)), 2)