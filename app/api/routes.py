"""API route definitions."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    CandidateResponse,
    CandidateScoreResponse,
    DecisionResponse,
    DensificationResponse,
    DensityResponse,
    DiffResponse,
    EvaluatePromptRequest,
    EvaluatePromptResponse,
    EvaluationResponse,
    IntentResponse,
    OptimizeAndStoreResponse,
    OptimizeRequest,
    OptimizeResponse,
    PromptCreateRequest,
    PromptListResponse,
    PromptRecord,
    PromptResponse,
    SelectionResponse,
    ResponseMetricsResponse,
)
from app.core.candidate_generator import CandidateGenerator
from app.core.compressor import Compressor
from app.core.decision_engine import DecisionEngine
from app.core.densifier import Densifier
from app.core.density_metrics import DensityMetrics
from app.core.diff_engine import DiffEngine
from app.core.evaluator import Evaluator
from app.core.intent_engine import IntentEngine, get_intent_strategy
from app.core.model_loader import ModelLoader
from app.core.response_evaluator import ResponseEvaluator, infer_prompt_structure
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.prompt_repository import PromptRepository

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------
# Helper — run the full v2 optimisation pipeline, return raw dicts
# ------------------------------------------------------------------

def _run_pipeline(
    prompt: str,
    mode: str,
    aggressiveness: float | None,
    auto_aggressiveness: bool | None,
    test_query: str | None = None,
) -> dict:
    """Execute the full prompt-compiler pipeline.

    Flow:
      1. Intent detection → aggressiveness resolution
      2. Candidate generation (aggressive / balanced / semantic / structured)
         — each candidate internally applies:
           a. Token compression via surprisal
           b. Phrase-level densification (rule-based)
           c. Optional prompt structuring
      3. Density scoring of all candidates
      4. Decision engine selects the best candidate
      5. Diff + evaluation on the selected output
    """
    model_loader = ModelLoader.get_instance()

    # --- 1. Intent detection & aggressiveness resolution ----------------
    intent_engine = IntentEngine()
    intent_result = intent_engine.detect(prompt, model_loader=model_loader)

    auto_mode: bool
    effective_aggressiveness: float

    if aggressiveness is not None:
        effective_aggressiveness = aggressiveness
        auto_mode = False
    elif auto_aggressiveness is False:
        effective_aggressiveness = 0.3
        auto_mode = False
    else:
        effective_aggressiveness = intent_result.recommended_aggressiveness
        auto_mode = True

    intent_info = {
        "intent": intent_result.intent_label,
        "intent_confidence": intent_result.confidence_score,
        "aggressiveness_used": round(effective_aggressiveness, 4),
        "auto_mode": auto_mode,
    }

    strategy = intent_result.get_optimization_strategy()

    logger.info(
        "Pipeline — intent=%s  aggr=%.2f  auto=%s  min_sim=%.2f",
        intent_result.intent_label,
        effective_aggressiveness,
        auto_mode,
        strategy.min_similarity,
    )

    # --- 2. Candidate generation ----------------------------------------
    generator = CandidateGenerator(model_loader)
    candidate_set = generator.generate(
        original_text=prompt,
        intent_label=intent_result.intent_label,
        base_aggressiveness=effective_aggressiveness,
        mode=mode,
        prefer_structuring=strategy.prefer_structuring,
        skip_aggressive=strategy.skip_aggressive_candidate,
    )

    # --- 3. Density scoring ---------------------------------------------
    density_metrics = DensityMetrics(model_loader)
    density_reports = density_metrics.score_candidates(
        original_text=prompt,
        candidates=candidate_set.texts(),
    )

    # Build scoring input for the decision engine
    scored_candidates = []
    for cand, dreport in zip(candidate_set.candidates, density_reports):
        scored_candidates.append({
            "strategy": cand.strategy,
            "text": cand.text,
            "similarity": dreport.semantic_similarity,
            "compression_ratio": dreport.compression_ratio,
            "density_score": dreport.density_score,
        })

    # --- 4. Decision engine — multi-candidate selection -----------------
    decision_engine = DecisionEngine()
    selection = decision_engine.select_best(
        candidates=scored_candidates,
        original_text=prompt,
        min_similarity=strategy.min_similarity,
        similarity_weight=strategy.similarity_weight,
        compression_weight=strategy.compression_weight,
        density_weight=strategy.density_weight,
    )

    selected_text = selection["selected_text"]

    # --- 5. Diff + evaluation on the selected output --------------------
    diff_engine = DiffEngine()
    diff_result = diff_engine.compute_diff(original=prompt, compressed=selected_text)

    evaluator = Evaluator(model_loader)
    query = test_query or "Summarize the above context."
    evaluation = evaluator.evaluate(
        original_prompt=prompt,
        compressed_prompt=selected_text,
        test_query=query,
    )

    # Also run legacy single-candidate decision for backward-compat fields
    # Find the primary candidate (the selected one) to get token reduction %
    selected_cand = None
    selected_density = None
    for cand, dreport in zip(candidate_set.candidates, density_reports):
        if cand.strategy == selection["selected_strategy"]:
            selected_cand = cand
            selected_density = dreport
            break

    # Compute token_reduction_percent from the selected candidate
    if selected_cand and selected_cand.token_compression:
        token_reduction_pct = selected_cand.token_compression.token_reduction_percent
    elif selected_density:
        token_reduction_pct = (1.0 - selected_density.compression_ratio) * 100.0
    else:
        token_reduction_pct = 0.0

    if selected_density is None:
        selected_density = density_metrics.score(prompt, selected_text)

    legacy_decision = decision_engine.decide(
        token_reduction_percent=token_reduction_pct,
        drift_score=evaluation.drift_score,
    )

    # Build densification info from the selected candidate
    densification_info = None
    if selected_cand and selected_cand.densification:
        densification_info = selected_cand.densification.to_dict()

    return {
        "selected_text": selected_text,
        "mode": mode,
        "token_reduction_percent": token_reduction_pct,
        "diff": diff_result,
        "evaluation": evaluation,
        "decision": legacy_decision,
        "intent": intent_info,
        "density": selected_density,
        "densification": densification_info,
        "candidates": candidate_set,
        "selection": selection,
        "density_reports": density_reports,
    }


# ==================================================================
# POST /optimize  (original endpoint — unchanged contract)
# ==================================================================

@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Compress a prompt and return structured analysis."""
    try:
        result = _run_pipeline(
            request.prompt,
            request.mode,
            request.aggressiveness,
            request.auto_aggressiveness,
            request.test_query,
        )

        # Build candidate list for response
        candidate_responses = None
        if result["candidates"] is not None:
            candidate_responses = [
                CandidateResponse(**c.to_dict())
                for c in result["candidates"].candidates
            ]

        # Build density response
        density_resp = None
        if result["density"] is not None:
            density_resp = DensityResponse(**result["density"].to_dict())

        # Build densification response
        densification_resp = None
        if result["densification"] is not None:
            densification_resp = DensificationResponse(
                phrase_reduction_count=result["densification"]["phrase_reduction_count"],
                rules_applied=result["densification"]["rules_applied"],
            )

        # Build selection response
        selection_resp = None
        sel = result["selection"]
        if sel is not None:
            selection_resp = SelectionResponse(
                decision=sel["decision"],
                selected_strategy=sel["selected_strategy"],
                composite_score=round(sel.get("composite_score", 0.0), 6),
                reason=sel["reason"],
                all_scores=[
                    CandidateScoreResponse(**s)
                    for s in sel.get("all_scores", [])
                ],
            )

        return OptimizeResponse(
            mode=result["mode"],
            compressed_prompt=result["selected_text"],
            original_token_count=result["density"].original_token_count if result["density"] is not None else 0,
            compressed_token_count=result["density"].compressed_token_count if result["density"] is not None else 0,
            compression_ratio=round(result["density"].compression_ratio, 4) if result["density"] is not None else 1.0,
            token_reduction_percent=round(result["token_reduction_percent"], 4),
            intent=IntentResponse(**result["intent"]),
            diff=DiffResponse(**result["diff"].to_dict()),
            evaluation=EvaluationResponse(**result["evaluation"].to_dict()),
            decision=DecisionResponse(**result["decision"]),
            density=density_resp,
            densification=densification_resp,
            candidates=candidate_responses,
            selection=selection_resp,
        )
    except Exception as exc:
        logger.exception("Optimization pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==================================================================
# POST /optimize-and-store
# ==================================================================

@router.post("/optimize-and-store", response_model=OptimizeAndStoreResponse)
async def optimize_and_store(request: OptimizeRequest) -> OptimizeAndStoreResponse:
    """Optimise a prompt **and** persist both versions to storage."""
    try:
        result = _run_pipeline(
            request.prompt,
            request.mode,
            request.aggressiveness,
            request.auto_aggressiveness,
            request.test_query,
        )
        evaluation = result["evaluation"]
        decision = result["decision"]
        intent_info = result["intent"]

        repo = PromptRepository.get_instance()
        record = repo.store(
            original_prompt=request.prompt,
            compressed_prompt=result["selected_text"],
            token_reduction_percent=result["token_reduction_percent"],
            drift_score=evaluation.drift_score,
            decision=decision["decision"],
            intent=intent_info["intent"],
            aggressiveness_used=intent_info["aggressiveness_used"],
            auto_mode=intent_info["auto_mode"],
        )

        density_dict = None
        if result["density"] is not None:
            density_dict = result["density"].to_dict()

        selection_dict = result["selection"]

        return OptimizeAndStoreResponse(
            success=True,
            data={
                "record": record,
                "mode": result["mode"],
                "compressed_prompt": result["selected_text"],
                "original_token_count": result["density"].original_token_count if result["density"] is not None else 0,
                "compressed_token_count": result["density"].compressed_token_count if result["density"] is not None else 0,
                "token_reduction_percent": round(result["token_reduction_percent"], 4),
                "intent": intent_info,
                "diff": result["diff"].to_dict(),
                "evaluation": evaluation.to_dict(),
                "decision": decision,
                "density": density_dict,
                "selection": {
                    "selected_strategy": selection_dict["selected_strategy"],
                    "composite_score": selection_dict.get("composite_score", 0.0),
                },
            },
            error=None,
        )
    except Exception as exc:
        logger.exception("Optimize-and-store failed")
        return OptimizeAndStoreResponse(success=False, data=None, error=str(exc))


# ==================================================================
# GET /prompts
# ==================================================================

@router.get("/prompts", response_model=PromptListResponse)
async def list_prompts() -> PromptListResponse:
    """Return every stored prompt record."""
    try:
        repo = PromptRepository.get_instance()
        records = repo.list_all()
        return PromptListResponse(
            success=True,
            data=[PromptRecord(**r) for r in records],
            error=None,
        )
    except Exception as exc:
        logger.exception("Failed to list prompts")
        return PromptListResponse(success=False, data=None, error=str(exc))


# ==================================================================
# POST /prompts
# ==================================================================

@router.post("/prompts", response_model=PromptResponse)
async def create_prompt(request: PromptCreateRequest) -> PromptResponse:
    """Manually store a prompt record."""
    try:
        repo = PromptRepository.get_instance()
        record = repo.store(
            original_prompt=request.original_prompt,
            compressed_prompt=request.compressed_prompt,
            token_reduction_percent=request.token_reduction_percent,
            drift_score=request.drift_score,
            decision=request.decision,
        )
        return PromptResponse(
            success=True,
            data=[PromptRecord(**record)],
            error=None,
        )
    except Exception as exc:
        logger.exception("Failed to store prompt")
        return PromptResponse(success=False, data=None, error=str(exc))


# ==================================================================
# GET /prompts/{prompt_id}
# ==================================================================

@router.get("/prompts/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: str) -> PromptResponse:
    """Return all versions of a single prompt."""
    try:
        repo = PromptRepository.get_instance()
        records = repo.get_by_id(prompt_id)
        if not records:
            return PromptResponse(success=False, data=None, error=f"Prompt '{prompt_id}' not found.")
        return PromptResponse(
            success=True,
            data=[PromptRecord(**r) for r in records],
            error=None,
        )
    except Exception as exc:
        logger.exception("Failed to retrieve prompt %s", prompt_id)
        return PromptResponse(success=False, data=None, error=str(exc))


# ==================================================================
# POST /evaluate_prompt
# ==================================================================

@router.post("/evaluate_prompt", response_model=EvaluatePromptResponse)
async def evaluate_prompt(request: EvaluatePromptRequest) -> EvaluatePromptResponse:
    """Evaluate original vs optimised prompt responses and persist the outcome."""
    try:
        optimized_prompt = request.optimized_prompt
        if not optimized_prompt:
            optimization = _run_pipeline(
                request.prompt,
                request.mode,
                request.aggressiveness,
                request.auto_aggressiveness,
                request.evaluation_query,
            )
            optimized_prompt = optimization["selected_text"]
        else:
            optimization = None

        model_loader = ModelLoader.get_instance()
        response_evaluator = ResponseEvaluator(model_loader)
        report = response_evaluator.evaluate(
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            evaluation_query=request.evaluation_query,
        )

        stored = False
        if request.store_result:
            repo = EvaluationRepository.get_instance()
            compression_ratio = 1.0
            if optimization and optimization["density"] is not None:
                compression_ratio = optimization["density"].compression_ratio

            repo.store({
                "original_prompt": request.prompt,
                "optimized_prompt": optimized_prompt,
                "mode": request.mode,
                "intent": optimization["intent"]["intent"] if optimization else None,
                "prompt_structure": infer_prompt_structure(optimized_prompt),
                "compression_ratio": round(compression_ratio, 4),
                "original_score": report.original_metrics.overall_score,
                "optimized_score": report.optimized_metrics.overall_score,
                "improvement_score": report.improvement_score,
                "evaluation_query": request.evaluation_query,
            })
            stored = True

        return EvaluatePromptResponse(
            optimized_prompt=optimized_prompt,
            original_response=report.original_response,
            optimized_response=report.optimized_response,
            original_metrics=ResponseMetricsResponse(**report.original_metrics.to_dict()),
            optimized_metrics=ResponseMetricsResponse(**report.optimized_metrics.to_dict()),
            improvement_score=round(report.improvement_score, 6),
            evaluation_query=report.evaluation_query,
            stored=stored,
        )
    except Exception as exc:
        logger.exception("Prompt evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
