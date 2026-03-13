"""Pydantic request / response schemas for the API layer."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Request
# ------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    """Request body for ``POST /optimize``."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The original prompt text to compress.",
    )
    mode: Literal["optimize", "enhance", "both"] = Field(
        default="optimize",
        description="Optimization mode: compression-focused, enhancement-focused, or both.",
    )
    aggressiveness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Pruning intensity (0 = conservative, 1 = aggressive). "
            "When omitted, auto-mode selects a value based on detected intent."
        ),
    )
    auto_aggressiveness: Optional[bool] = Field(
        default=True,
        description=(
            "When True (default), the system detects prompt intent and "
            "chooses aggressiveness automatically.  Ignored when an explicit "
            "aggressiveness value is provided."
        ),
    )
    test_query: Optional[str] = Field(
        default=None,
        description="Optional probe query used during evaluation.",
    )


class EvaluatePromptRequest(BaseModel):
    """Request body for ``POST /evaluate_prompt``."""

    prompt: str = Field(..., min_length=1)
    optimized_prompt: Optional[str] = Field(
        default=None,
        description="Optional already-optimized prompt. When omitted, the API optimizes first.",
    )
    mode: Literal["optimize", "enhance", "both"] = Field(default="both")
    aggressiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auto_aggressiveness: Optional[bool] = Field(default=True)
    evaluation_query: str = Field(
        default="Answer the request clearly with structure and examples.",
        description="Instruction used when generating comparison responses.",
    )
    store_result: bool = Field(
        default=True,
        description="Persist the evaluation outcome for future analysis.",
    )


class PromptCreateRequest(BaseModel):
    """Request body for ``POST /prompts`` — manual prompt storage."""

    original_prompt: str = Field(..., min_length=1)
    compressed_prompt: str = Field(default="")
    token_reduction_percent: float = Field(default=0.0)
    drift_score: float = Field(default=0.0)
    decision: str = Field(default="PENDING")


# ------------------------------------------------------------------
# Response sub-models
# ------------------------------------------------------------------

class DiffResponse(BaseModel):
    """Structured diff section of the response."""

    removed: List[str] = Field(default_factory=list)
    rewritten: List[Dict[str, str]] = Field(default_factory=list)
    preserved: List[str] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    """Quality-drift evaluation sub-section."""

    semantic_similarity: float
    length_difference: int
    length_ratio: float
    drift_score: float
    original_response: str
    compressed_response: str


class DecisionResponse(BaseModel):
    """Decision sub-section."""

    decision: str
    reason: str
    token_reduction_percent: float
    drift_score: float


class IntentResponse(BaseModel):
    """Intent detection sub-section."""

    intent: str
    intent_confidence: float
    aggressiveness_used: float
    auto_mode: bool


# ------------------------------------------------------------------
# New v2 sub-models — density, candidates, selection
# ------------------------------------------------------------------

class DensityResponse(BaseModel):
    """Information density metrics."""

    density_score: float
    semantic_similarity: float
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    information_per_token: float


class CandidateScoreResponse(BaseModel):
    """Score breakdown for a single candidate."""

    strategy: str
    similarity: float
    compression_ratio: float
    density_score: float
    composite_score: float


class CandidateResponse(BaseModel):
    """Summary of a generated candidate."""

    strategy: str
    text: str
    token_reduction_percent: Optional[float] = None
    phrase_rules_applied: Optional[int] = None
    was_restructured: Optional[bool] = None


class SelectionResponse(BaseModel):
    """Multi-candidate selection result."""

    decision: str
    selected_strategy: str
    composite_score: float
    reason: str
    all_scores: List[CandidateScoreResponse] = Field(default_factory=list)


class DensificationResponse(BaseModel):
    """Phrase-level densification metadata."""

    phrase_reduction_count: int
    rules_applied: List[Dict[str, str]] = Field(default_factory=list)


# ------------------------------------------------------------------
# Top-level response — v2 /optimize (superset of v1)
# ------------------------------------------------------------------

class OptimizeResponse(BaseModel):
    """Full response for ``POST /optimize``."""

    mode: str
    compressed_prompt: str
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    token_reduction_percent: float
    intent: IntentResponse
    diff: DiffResponse
    evaluation: EvaluationResponse
    decision: DecisionResponse
    # ---- v2 additions ----
    density: Optional[DensityResponse] = None
    densification: Optional[DensificationResponse] = None
    candidates: Optional[List[CandidateResponse]] = None
    selection: Optional[SelectionResponse] = None


# ------------------------------------------------------------------
# Prompt storage models
# ------------------------------------------------------------------

class PromptRecord(BaseModel):
    """A single stored prompt record."""

    id: str
    original_prompt: str
    compressed_prompt: str
    token_reduction_percent: float
    drift_score: float
    decision: str
    created_at: str
    version: int
    intent: Optional[str] = None
    aggressiveness_used: Optional[float] = None
    auto_mode: Optional[bool] = None


class PromptResponse(BaseModel):
    """Response wrapper for a single prompt (may contain version history)."""

    success: bool = True
    data: Optional[List[PromptRecord]] = None
    error: Optional[str] = None


class PromptListResponse(BaseModel):
    """Response wrapper for a list of prompts."""

    success: bool = True
    data: Optional[List[PromptRecord]] = None
    error: Optional[str] = None


class OptimizeAndStoreResponse(BaseModel):
    """Response for ``POST /optimize-and-store``."""

    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ResponseMetricsResponse(BaseModel):
    """Quality metrics for a generated response."""

    semantic_coverage: float
    structural_quality: float
    length_quality: float
    information_density: float
    overall_score: float


class EvaluatePromptResponse(BaseModel):
    """Response model for ``POST /evaluate_prompt``."""

    optimized_prompt: str
    original_response: str
    optimized_response: str
    original_metrics: ResponseMetricsResponse
    optimized_metrics: ResponseMetricsResponse
    improvement_score: float
    evaluation_query: str
    stored: bool = False


# ------------------------------------------------------------------
# Analyze endpoint models
# ------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for ``POST /analyze``."""

    prompt: str = Field(..., min_length=1, description="The original prompt to analyze.")
    mode: Literal["optimize", "enhance", "both"] = Field(default="both")
    aggressiveness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    auto_aggressiveness: Optional[bool] = Field(default=True)


class DimensionScoreResponse(BaseModel):
    """Score for a single quality dimension."""

    dimension: str
    original_score: int
    optimized_score: int


class TemplateInfoResponse(BaseModel):
    """Template detection result."""

    is_templatizable: bool
    template_name: str
    template_structure: str
    variables: List[str] = Field(default_factory=list)


class ComputeScoreResponse(BaseModel):
    """Prompt compute complexity estimate."""

    token_length: int
    instruction_complexity: int
    reasoning_depth: int
    expected_output_size: int
    ambiguity: int
    overall: int


class AnalyzeResponse(BaseModel):
    """Response for ``POST /analyze``."""

    intent: str
    optimized_prompt: str
    comparison: List[DimensionScoreResponse]
    template: TemplateInfoResponse
    summary: str
    # Compute complexity
    compute_original: ComputeScoreResponse
    compute_optimized: ComputeScoreResponse
    compute_reduction_percent: float
    # Pipeline metadata piggy-backed for convenience
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 1.0
    mode: str = "both"
    intent_detail: Optional[IntentResponse] = None
    density: Optional[DensityResponse] = None
