"""Prompt Analyzer — structured multi-dimensional prompt analysis.

Implements a lightweight five-task analysis pipeline:
  1. Intent understanding (one-sentence summary)
  2. Prompt comparison scoring (Clarity, Specificity, Context, Task Definition, Structure)
  3. Template detection with variable extraction
  4. Summary generation

All scoring is heuristic-based — no external LLM calls required.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data-classes for structured output
# ------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: str
    original_score: int       # 1–10
    optimized_score: int      # 1–10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "original_score": self.original_score,
            "optimized_score": self.optimized_score,
        }


@dataclass
class TemplateInfo:
    """Detected template information."""

    is_templatizable: bool
    template_name: str
    template_structure: str
    variables: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_templatizable": self.is_templatizable,
            "template_name": self.template_name,
            "template_structure": self.template_structure,
            "variables": self.variables,
        }


@dataclass
class ComputeScore:
    """Prompt compute complexity estimate."""

    token_length: int             # 1–10
    instruction_complexity: int   # 1–10
    reasoning_depth: int          # 1–10
    expected_output_size: int     # 1–10
    ambiguity: int                # 1–10
    overall: int                  # 1–10 (weighted average)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_length": self.token_length,
            "instruction_complexity": self.instruction_complexity,
            "reasoning_depth": self.reasoning_depth,
            "expected_output_size": self.expected_output_size,
            "ambiguity": self.ambiguity,
            "overall": self.overall,
        }


@dataclass
class AnalysisReport:
    """Full analysis report."""

    intent: str
    optimized_prompt: str
    comparison: List[DimensionScore]
    template: TemplateInfo
    summary: str
    compute_original: ComputeScore
    compute_optimized: ComputeScore
    compute_reduction_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "optimized_prompt": self.optimized_prompt,
            "comparison": [d.to_dict() for d in self.comparison],
            "template": self.template.to_dict(),
            "summary": self.summary,
            "compute_original": self.compute_original.to_dict(),
            "compute_optimized": self.compute_optimized.to_dict(),
            "compute_reduction_percent": self.compute_reduction_percent,
        }


# ------------------------------------------------------------------
# Keyword / pattern banks
# ------------------------------------------------------------------

_GOAL_VERBS = re.compile(
    r"\b(explain|write|build|create|compare|analyze|analyse|summarize|summarise|"
    r"debug|design|solve|answer|describe|list|generate|implement|evaluate|review|"
    r"translate|convert|plan|draft|outline|develop|refactor|optimise|optimize|"
    r"calculate|fix|improve|recommend|teach|test|deploy|configure|setup|"
    r"classify|extract|scrape|visualize|document|benchmark|profile)\b",
    re.IGNORECASE,
)

_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(step[- ]by[- ]step|example|bullet|table|format|json|markdown|concise|"
    r"detailed|tone|style|limit|maximum|minimum|under \d+|at least|no more than|"
    r"within|between|exactly|brief|comprehensive|professional|casual|formal|"
    r"numbered|in .{2,20} format|as a .{2,20}|using .{2,20})\b",
    re.IGNORECASE,
)

_CONTEXT_SIGNALS = re.compile(
    r"\b(context|background|given|assume|scenario|role|you are|act as|"
    r"for a|audience|target|purpose|domain|field|industry|scope|"
    r"regarding|about|concerning|related to|in the context of|"
    r"based on|considering|with respect to)\b",
    re.IGNORECASE,
)

_TASK_MARKERS = re.compile(
    r"\b(task|goal|objective|requirement|deliverable|output|produce|"
    r"result|expected|should|must|need to|have to|ensure|make sure|"
    r"the output should|return|respond with|provide)\b",
    re.IGNORECASE,
)

_STRUCTURE_SIGNALS = re.compile(
    r"(\n|^\s*[-*]\s|^\s*\d+\.\s|:\s|^#{1,3}\s|```|---|\|)",
    re.MULTILINE,
)

_FILLER_WORDS = re.compile(
    r"\b(please|kindly|i would like|can you|could you|would you|just|"
    r"maybe|perhaps|i think|basically|actually|literally|honestly|"
    r"feel free|that being said|in order to|as a matter of fact)\b",
    re.IGNORECASE,
)

# Template variable candidates: quoted strings, specific nouns, numbers, etc.
_QUOTED_STRINGS = re.compile(r'"[^"]{2,60}"|\'[^\']{2,60}\'')
_SPECIFIC_NUMBERS = re.compile(r"\b\d{2,}\b")
_PROPER_NOUNS = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")

# Common templatizable patterns
_TEMPLATE_PATTERNS: List[Tuple[re.Pattern, str, str, List[str]]] = [
    (
        re.compile(r"(?:write|create|draft|generate)\s+(?:a|an)\s+(\w+)\s+(?:about|on|for)", re.I),
        "Content Generator",
        "Write a {content_type} about {topic} for {audience}.",
        ["content_type", "topic", "audience"],
    ),
    (
        re.compile(r"(?:explain|describe|summarize)\s+(.+?)(?:\s+(?:to|for)\s+)", re.I),
        "Explanation Generator",
        "Explain {topic} to {audience} in {style} style.",
        ["topic", "audience", "style"],
    ),
    (
        re.compile(r"(?:compare|contrast)\s+(.+?)\s+(?:and|vs|with)\s+", re.I),
        "Comparison Generator",
        "Compare {item_a} and {item_b} based on {criteria}.",
        ["item_a", "item_b", "criteria"],
    ),
    (
        re.compile(r"(?:translate|convert)\s+(.+?)\s+(?:to|into|from)\s+", re.I),
        "Translation/Conversion",
        "Translate {source_content} from {source_format} to {target_format}.",
        ["source_content", "source_format", "target_format"],
    ),
    (
        re.compile(r"(?:analyze|analyse|review)\s+(?:this|the|my)\s+(\w+)", re.I),
        "Analysis Template",
        "Analyze {subject} focusing on {aspects} and provide {output_format}.",
        ["subject", "aspects", "output_format"],
    ),
    (
        re.compile(r"(?:build|implement|create|develop|design)\s+(?:a|an)\s+(\w+)", re.I),
        "Builder Template",
        "Build a {component_type} that {functionality} using {technology}.",
        ["component_type", "functionality", "technology"],
    ),
]


# ------------------------------------------------------------------
# Scoring functions (each returns 1–10)
# ------------------------------------------------------------------

def _score_clarity(text: str) -> int:
    """How clear the prompt is in communicating its intent."""
    score = 3  # baseline

    words = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    # Has explicit goal verb (+2)
    if _GOAL_VERBS.search(text):
        score += 2

    # Sentence count: 1-4 is ideal, deduct for too many or ambiguity
    if 1 <= len(sentences) <= 4:
        score += 2
    elif len(sentences) <= 7:
        score += 1

    # Word count: sweet spot 10-80
    if 10 <= len(words) <= 80:
        score += 1
    elif len(words) > 120:
        score -= 1

    # Penalise filler words
    fillers = len(_FILLER_WORDS.findall(text))
    score -= min(fillers, 3)

    # Has a question mark / imperative = clearer
    if re.search(r"[?]|^(explain|write|list|describe|compare|create)\b", text, re.I):
        score += 1

    return max(1, min(10, score))


def _score_specificity(text: str) -> int:
    """How specific and constrained the prompt is."""
    score = 2  # baseline

    # Constraint keywords
    constraints = len(_CONSTRAINT_KEYWORDS.findall(text))
    score += min(constraints * 2, 4)

    # Numeric values indicate precision
    numbers = len(_SPECIFIC_NUMBERS.findall(text))
    score += min(numbers, 2)

    # Quoted values / named entities
    quotes = len(_QUOTED_STRINGS.findall(text))
    score += min(quotes, 2)

    # Word count — more words often means more specific (up to a point)
    word_count = len(text.split())
    if word_count >= 20:
        score += 1
    if word_count >= 50:
        score += 1

    # Penalise vague-only prompts
    if not _CONSTRAINT_KEYWORDS.search(text) and not _GOAL_VERBS.search(text):
        score -= 2

    return max(1, min(10, score))


def _score_context(text: str) -> int:
    """How much context / background the prompt provides."""
    score = 2  # baseline

    # Context signals
    signals = len(_CONTEXT_SIGNALS.findall(text))
    score += min(signals * 2, 4)

    # Has role / persona definition
    if re.search(r"(you are|act as|role|persona|as a\s+\w+)", text, re.I):
        score += 2

    # Has audience / purpose
    if re.search(r"(for\s+(a|an|the)?\s*\w+|audience|beginner|expert|student|developer)", text, re.I):
        score += 1

    # Longer prompts typically carry more context
    if len(text) > 150:
        score += 1

    return max(1, min(10, score))


def _score_task_definition(text: str) -> int:
    """How well the task/outcome is defined."""
    score = 2  # baseline

    # Explicit task markers
    markers = len(_TASK_MARKERS.findall(text))
    score += min(markers * 2, 4)

    # Goal verb present
    if _GOAL_VERBS.search(text):
        score += 2

    # Has output format specification
    if re.search(r"(format|output|return|respond|provide)\s+.{3,}", text, re.I):
        score += 1

    # Multi-step instructions
    steps = len(re.findall(r"(?:^\s*\d+\.|^\s*[-*])\s", text, re.M))
    if steps >= 2:
        score += 1

    return max(1, min(10, score))


def _score_structure(text: str) -> int:
    """How well-structured / formatted the prompt is."""
    score = 2  # baseline

    # Newlines → some structure
    newlines = text.count("\n")
    if newlines >= 1:
        score += 1
    if newlines >= 3:
        score += 1

    # Bullet points or numbered lists
    lists = len(re.findall(r"(?:^\s*[-*]\s|^\s*\d+\.)\s", text, re.M))
    score += min(lists, 3)

    # Colons — section indicators
    if text.count(":") >= 1:
        score += 1

    # Headers / separators
    if re.search(r"^#{1,3}\s|^---", text, re.M):
        score += 1

    # Code blocks
    if "```" in text:
        score += 1

    # Single-line blob penalty
    if newlines == 0 and len(text) > 100:
        score -= 1

    return max(1, min(10, score))


# ------------------------------------------------------------------
# Intent summarisation
# ------------------------------------------------------------------

def _summarise_intent(text: str) -> str:
    """Generate a one-sentence intent summary from the prompt."""
    text_clean = text.strip()
    if not text_clean:
        return "Empty prompt with no identifiable intent."

    words = text_clean.split()

    # Try to detect the main verb
    verb_match = _GOAL_VERBS.search(text_clean)
    main_verb = verb_match.group(1).lower() if verb_match else None

    # Extract first meaningful sentence
    first_sentence = re.split(r"[.!?\n]", text_clean)[0].strip()
    if len(first_sentence) > 100:
        first_sentence = " ".join(first_sentence.split()[:15]) + "…"

    if main_verb:
        # Build: "The user wants to {verb} {topic}"
        verb_pos = verb_match.start()
        after_verb = text_clean[verb_match.end():].strip()
        topic_words = after_verb.split()[:8]
        topic = " ".join(topic_words).rstrip(".,;:!?")
        if topic:
            return f"The user wants to {main_verb} {topic}."
        return f"The user wants to {main_verb} something (unspecified target)."

    # Fallback — summarise first sentence
    if len(words) <= 5:
        return f'The user asks: "{text_clean}"'

    return f'The user wants: "{first_sentence}"'


# ------------------------------------------------------------------
# Template detection
# ------------------------------------------------------------------

def _detect_template(text: str) -> TemplateInfo:
    """Detect whether the prompt is templatizable.

    Returns a ``TemplateInfo`` with extracted template and variables,
    or a no-template marker.
    """
    if not text or len(text.strip()) < 15:
        return TemplateInfo(
            is_templatizable=False,
            template_name="",
            template_structure="Not Applicable",
            variables=[],
        )

    # 1. Check known templatizable patterns
    for pattern, name, structure, variables in _TEMPLATE_PATTERNS:
        if pattern.search(text):
            return TemplateInfo(
                is_templatizable=True,
                template_name=name,
                template_structure=structure,
                variables=variables,
            )

    # 2. Heuristic detection: if the prompt has quoted strings, specific
    #    nouns, or numbers that could be replaced with variables
    quoted = _QUOTED_STRINGS.findall(text)
    proper_nouns = [
        m for m in _PROPER_NOUNS.findall(text)
        if m.lower() not in {
            "the", "this", "that", "then", "than", "there", "these",
            "when", "what", "which", "while", "return", "provide",
            "please", "also", "use", "make", "sure",
        }
    ]

    replaceable_parts = []
    template_text = text
    variables: List[str] = []

    # Replace quoted strings with variable placeholders
    for i, q in enumerate(quoted[:3]):
        var_name = f"value_{i + 1}" if i > 0 else "topic"
        replaceable_parts.append((q, var_name))

    # Replace specific proper nouns
    seen_nouns = set()
    for noun in proper_nouns[:3]:
        if noun in seen_nouns:
            continue
        seen_nouns.add(noun)
        var_name = _noun_to_variable(noun)
        if var_name not in variables:
            replaceable_parts.append((noun, var_name))

    if not replaceable_parts:
        # Last resort: check if the prompt looks generic enough
        words = text.split()
        if len(words) >= 8 and _GOAL_VERBS.search(text):
            # Build a generic template from the verb + subject pattern
            verb_m = _GOAL_VERBS.search(text)
            verb = verb_m.group(1).lower()
            return TemplateInfo(
                is_templatizable=True,
                template_name=f"{verb.capitalize()} Template",
                template_structure=f"{verb.capitalize()} {{topic}} including {{details}} for {{audience}}.",
                variables=["topic", "details", "audience"],
            )
        return TemplateInfo(
            is_templatizable=False,
            template_name="",
            template_structure="No template suggested",
            variables=[],
        )

    # Build the template
    for original, var_name in replaceable_parts:
        template_text = template_text.replace(original, "{" + var_name + "}", 1)
        variables.append(var_name)

    # Generate a template name
    verb_m = _GOAL_VERBS.search(text)
    if verb_m:
        template_name = f"{verb_m.group(1).capitalize()} Template"
    else:
        template_name = "Reusable Prompt Template"

    return TemplateInfo(
        is_templatizable=True,
        template_name=template_name,
        template_structure=template_text.strip(),
        variables=variables,
    )


def _noun_to_variable(noun: str) -> str:
    """Convert a proper noun to a snake_case variable name."""
    return re.sub(r"\s+", "_", noun.strip()).lower()


# ------------------------------------------------------------------
# Compute complexity scoring
# ------------------------------------------------------------------

_REASONING_MARKERS = re.compile(
    r"\b(reason|why|because|cause|effect|consequence|implication|therefore|thus|hence|"
    r"if.{1,30}then|trade[- ]?off|pros and cons|advantages|disadvantages|evaluate|assess|"
    r"weigh|consider|decide|judge|critique|argue|debate|proof|prove|derive|deduce|infer|"
    r"hypothes[ie]s|logically|mathematically|calculate|compute|algorithm|complex|optimiz)",
    re.IGNORECASE,
)

_OUTPUT_SIZE_MARKERS = re.compile(
    r"\b(essay|article|blog post|report|document|guide|tutorial|paper|chapter|book|"
    r"comprehensive|detailed|in[- ]depth|thorough|complete|full|exhaustive|extended|"
    r"at least \d{3,}|\d{3,} words|long[- ]form|multi[- ]?page)",
    re.IGNORECASE,
)

_SMALL_OUTPUT_MARKERS = re.compile(
    r"\b(one[- ]?line|brief|short|concise|summary|tldr|tl;dr|quick|few words|"
    r"single sentence|one sentence|yes or no|true or false|name|list)",
    re.IGNORECASE,
)

_MULTI_STEP = re.compile(
    r"(?:^\s*\d+[\.\)]|^\s*[-*]\s|first.{0,40}then|step \d|phase \d|\bstep[- ]by[- ]step)",
    re.IGNORECASE | re.MULTILINE,
)


def _compute_token_length(text: str) -> int:
    """Score based on approximate token count (1 = very short, 10 = very long)."""
    # Rough estimate: 1 token ≈ 4 characters
    tokens_est = max(len(text.split()), len(text) // 4)
    if tokens_est <= 10:
        return 1
    if tokens_est <= 25:
        return 2
    if tokens_est <= 50:
        return 3
    if tokens_est <= 80:
        return 4
    if tokens_est <= 120:
        return 5
    if tokens_est <= 180:
        return 6
    if tokens_est <= 260:
        return 7
    if tokens_est <= 400:
        return 8
    if tokens_est <= 600:
        return 9
    return 10


def _compute_instruction_complexity(text: str) -> int:
    """Score based on number and nesting of instructions."""
    score = 2
    steps = len(_MULTI_STEP.findall(text))
    score += min(steps, 4)
    constraints = len(_CONSTRAINT_KEYWORDS.findall(text))
    score += min(constraints, 2)
    if re.search(r"(nested|recursive|iterative|chain|pipeline|multi[- ]?step)", text, re.I):
        score += 2
    return max(1, min(10, score))


def _compute_reasoning_depth(text: str) -> int:
    """Score based on reasoning / analytical complexity."""
    score = 1
    hits = len(_REASONING_MARKERS.findall(text))
    score += min(hits * 2, 6)
    if re.search(r"(compare|contrast|analyze|evaluate|critique)", text, re.I):
        score += 1
    if re.search(r"(mathematically|formally|rigorously|prove|derive)", text, re.I):
        score += 2
    return max(1, min(10, score))


def _compute_expected_output_size(text: str) -> int:
    """Score based on expected length of model output."""
    # Small output?
    if _SMALL_OUTPUT_MARKERS.search(text):
        return 2
    score = 3  # baseline medium
    if _OUTPUT_SIZE_MARKERS.search(text):
        score += 4
    steps = len(_MULTI_STEP.findall(text))
    if steps >= 3:
        score += 2
    elif steps >= 1:
        score += 1
    words = len(text.split())
    if words > 100:
        score += 1
    return max(1, min(10, score))


def _compute_ambiguity(text: str) -> int:
    """Score ambiguity (1 = crystal clear, 10 = very ambiguous)."""
    score = 3  # baseline
    fillers = len(_FILLER_WORDS.findall(text))
    score += min(fillers, 3)
    if not _GOAL_VERBS.search(text):
        score += 2
    if not _CONSTRAINT_KEYWORDS.search(text):
        score += 1
    if re.search(r"(something|stuff|things|whatever|anything|etc)", text, re.I):
        score += 1
    # specificity reduces ambiguity
    if _SPECIFIC_NUMBERS.search(text) or _QUOTED_STRINGS.search(text):
        score -= 2
    if len(text.split()) < 6:
        score += 1  # very short prompts tend to be ambiguous
    return max(1, min(10, score))


def _compute_score(text: str) -> ComputeScore:
    """Compute overall prompt complexity score."""
    tl = _compute_token_length(text)
    ic = _compute_instruction_complexity(text)
    rd = _compute_reasoning_depth(text)
    eos = _compute_expected_output_size(text)
    amb = _compute_ambiguity(text)
    # Weighted average — token_length and expected_output dominate compute cost
    overall = round(
        tl * 0.25 + ic * 0.20 + rd * 0.20 + eos * 0.25 + amb * 0.10
    )
    overall = max(1, min(10, overall))
    return ComputeScore(
        token_length=tl,
        instruction_complexity=ic,
        reasoning_depth=rd,
        expected_output_size=eos,
        ambiguity=amb,
        overall=overall,
    )


# ------------------------------------------------------------------
# Summary generation
# ------------------------------------------------------------------

def _generate_summary(
    original_scores: Dict[str, int],
    optimized_scores: Dict[str, int],
) -> str:
    """One-sentence summary of the improvement."""
    improved_dims = []
    for dim in original_scores:
        delta = optimized_scores[dim] - original_scores[dim]
        if delta > 0:
            improved_dims.append((dim, delta))

    if not improved_dims:
        return "The optimized prompt maintains similar quality across all dimensions."

    improved_dims.sort(key=lambda x: x[1], reverse=True)
    top = improved_dims[:3]
    parts = [f"{dim} (+{delta})" for dim, delta in top]
    return f"The optimized prompt improves {', '.join(parts)} while preserving semantic meaning."


# ------------------------------------------------------------------
# Main analyzer class
# ------------------------------------------------------------------

class PromptAnalyzer:
    """Lightweight prompt analysis engine.

    Performs five tasks:
      1. Intent understanding
      2. Comparison scoring (Clarity, Specificity, Context, Task Definition, Structure)
      3. Template detection
      4. Summary

    The *optimized_prompt* can be provided externally (from the pipeline)
    or omitted, in which case only the original is scored.
    """

    _DIMENSIONS = ["Clarity", "Specificity", "Context", "Task Definition", "Structure"]
    _SCORERS = {
        "Clarity": _score_clarity,
        "Specificity": _score_specificity,
        "Context": _score_context,
        "Task Definition": _score_task_definition,
        "Structure": _score_structure,
    }

    def analyze(
        self,
        original_prompt: str,
        optimized_prompt: str,
    ) -> AnalysisReport:
        """Run the full analysis pipeline.

        Args:
            original_prompt: the user's raw prompt.
            optimized_prompt: the pipeline-optimized version.

        Returns:
            An ``AnalysisReport`` with intent, comparison, template, and summary.
        """
        # Step 1 — identify intent
        intent = _summarise_intent(original_prompt)

        # Step 4 — prompt comparison scoring
        original_scores: Dict[str, int] = {}
        optimized_scores: Dict[str, int] = {}
        comparison: List[DimensionScore] = []

        for dim in self._DIMENSIONS:
            scorer = self._SCORERS[dim]
            o_score = scorer(original_prompt)
            c_score = scorer(optimized_prompt)
            original_scores[dim] = o_score
            optimized_scores[dim] = c_score
            comparison.append(DimensionScore(
                dimension=dim,
                original_score=o_score,
                optimized_score=c_score,
            ))

        # Step 3 — automatic template extraction (from the optimized prompt)
        template = _detect_template(optimized_prompt)

        # Step 5 — summary
        summary = _generate_summary(original_scores, optimized_scores)

        # Compute complexity scoring
        compute_orig = _compute_score(original_prompt)
        compute_opt = _compute_score(optimized_prompt)
        if compute_orig.overall > 0:
            reduction = ((compute_orig.overall - compute_opt.overall) / compute_orig.overall) * 100.0
        else:
            reduction = 0.0

        logger.info(
            "Analysis — intent='%s'  dims=%s  template=%s  compute=%d→%d (%.1f%% reduction)",
            intent[:60],
            {d: (original_scores[d], optimized_scores[d]) for d in self._DIMENSIONS},
            template.is_templatizable,
            compute_orig.overall,
            compute_opt.overall,
            reduction,
        )

        return AnalysisReport(
            intent=intent,
            optimized_prompt=optimized_prompt,
            comparison=comparison,
            template=template,
            summary=summary,
            compute_original=compute_orig,
            compute_optimized=compute_opt,
            compute_reduction_percent=round(reduction, 2),
        )

    def score_single(self, text: str) -> Dict[str, int]:
        """Score a single prompt on all five dimensions.

        Useful for the real-time quality card in the extension.
        """
        return {dim: self._SCORERS[dim](text) for dim in self._DIMENSIONS}
