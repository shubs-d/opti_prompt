"""API route definitions."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CandidateResponse,
    CandidateScoreResponse,
    ComputeScoreResponse,
    CostResponse,
    DecisionResponse,
    DensificationResponse,
    DensityResponse,
    DiffResponse,
    DimensionScoreResponse,
    EfficiencyMetricsResponse,
    EvaluatePromptRequest,
    EvaluatePromptResponse,
    EvaluationResponse,
    EvolutionVariantResponse,
    GepaRepairResponse,
    IntentResponse,
    OptimizeAndStoreResponse,
    OptimizeRequest,
    OptimizeResponse,
    PipelineRequest,
    PipelineResponse,
    PipelineTemplateResponse,
    PromptCreateRequest,
    PromptListResponse,
    PromptRecord,
    PredictRequest,
    PredictResponse,
    PromptResponse,
    SelectionResponse,
    ResponseMetricsResponse,
    TemplateInfoResponse,
)
from app.core.candidate_generator import CandidateGenerator
from app.core.compressor import Compressor, compress_with_genome, score_compression_variant
from app.core.decision_engine import DecisionEngine
from app.core.densifier import Densifier
from app.core.density_metrics import DensityMetrics
from app.core.diff_engine import DiffEngine
from app.core.evaluator import Evaluator
from app.core.gepa.mutator import GepaMutator
from app.core.gepa.optimizer import GepaOptimizer
from app.core.intent_engine import (
    IntentEngine,
    get_default_aggressiveness_for_intent,
    get_intent_strategy,
    normalize_intent_label,
)
from app.core.model_loader import ModelLoader
from app.core.prompt_analyzer import PromptAnalyzer
from app.core.response_evaluator import ResponseEvaluator, infer_prompt_structure
from app.config import (
    CONTROLLED_ENFORCE_COMPRESSION_WINDOW,
    CONTROLLED_MAX_TOTAL_COMPRESSION_PERCENT,
    CONTROLLED_MIN_TOTAL_COMPRESSION_PERCENT,
)
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.prompt_repository import PromptRepository
from genome_loader import load_best_genome

logger = logging.getLogger(__name__)
router = APIRouter()


def _reduction_percent_from_ratio(compression_ratio: float) -> float:
    return max(0.0, min(100.0, (1.0 - compression_ratio) * 100.0))


def _select_window_compliant_output(
    *,
    original_prompt: str,
    selected_text: str,
    variant_texts: dict[str, str],
    evaluator: Evaluator,
    density_metrics: DensityMetrics,
    test_query: str,
    min_total_compression_percent: float,
    max_total_compression_percent: float,
) -> tuple[str, str]:
    """Choose the best output while enforcing a compression-window preference.

    Preference order:
      1) Candidate inside [min, max] compression window, highest reduction.
      2) Otherwise nearest-to-window candidate with low drift.
    """
    # Deduplicate equivalent texts while keeping the first label.
    deduped: dict[str, str] = {"selected_pre_control": selected_text}
    seen_texts = {selected_text}
    for label, text in variant_texts.items():
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        deduped[label] = text

    # Always keep the original as a strict fallback.
    if original_prompt not in seen_texts:
        deduped["original_fallback"] = original_prompt

    best_label = "selected_pre_control"
    best_text = selected_text
    best_rank = (-1.0, -1.0, -1.0, -1.0)

    for label, text in deduped.items():
        density = density_metrics.score(original_prompt, text)
        reduction_pct = _reduction_percent_from_ratio(density.compression_ratio)
        evaluation = evaluator.evaluate(
            original_prompt=original_prompt,
            compressed_prompt=text,
            test_query=test_query,
        )
        drift = evaluation.drift_score

        in_window = min_total_compression_percent <= reduction_pct <= max_total_compression_percent
        if in_window:
            # Prefer higher compression inside the safe band, then lower drift.
            rank = (1.0, reduction_pct, -drift, density.density_score)
        else:
            distance = (
                (min_total_compression_percent - reduction_pct)
                if reduction_pct < min_total_compression_percent
                else (reduction_pct - max_total_compression_percent)
            )
            rank = (0.0, -distance, -drift, density.density_score)

        if rank > best_rank:
            best_rank = rank
            best_label = label
            best_text = text

    logger.info(
        "Controlled window selector picked '%s' (target %.1f%%-%.1f%%)",
        best_label,
        min_total_compression_percent,
        max_total_compression_percent,
    )
    return best_text, best_label


# ------------------------------------------------------------------
# Helper — run the full v2 optimisation pipeline, return raw dicts
# ------------------------------------------------------------------

async def _run_pipeline(
    prompt: str,
    mode: str,
    aggressiveness: float | None,
    auto_aggressiveness: bool | None,
    test_query: str | None = None,
    intent_override: str | None = None,
    use_gepa_repair: bool = False,
    gepa_candidate_count: int = 3,
    gepa_token_budget_ratio: float = 0.72,
    use_gepa: bool = True,
    gepa_generations: int = 6,
    gepa_population_size: int = 6,
    gepa_time_budget_seconds: float = 1.5,
    enforce_compression_window: bool = CONTROLLED_ENFORCE_COMPRESSION_WINDOW,
    min_total_compression_percent: float = CONTROLLED_MIN_TOTAL_COMPRESSION_PERCENT,
    max_total_compression_percent: float = CONTROLLED_MAX_TOTAL_COMPRESSION_PERCENT,
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
    detected_intent_result = intent_engine.detect(prompt, model_loader=model_loader)

    if intent_override:
        resolved_intent = normalize_intent_label(intent_override)
        intent_label = resolved_intent
        intent_confidence = 1.0
        recommended_aggressiveness = get_default_aggressiveness_for_intent(resolved_intent)
    else:
        intent_label = detected_intent_result.intent_label
        intent_confidence = detected_intent_result.confidence_score
        recommended_aggressiveness = detected_intent_result.recommended_aggressiveness

    auto_mode: bool
    effective_aggressiveness: float

    if aggressiveness is not None:
        effective_aggressiveness = aggressiveness
        auto_mode = False
    elif auto_aggressiveness is False:
        effective_aggressiveness = 0.3
        auto_mode = False
    else:
        effective_aggressiveness = recommended_aggressiveness
        auto_mode = True

    intent_info = {
        "intent": intent_label,
        "intent_confidence": intent_confidence,
        "aggressiveness_used": round(effective_aggressiveness, 4),
        "auto_mode": auto_mode,
    }

    strategy = get_intent_strategy(intent_label)

    logger.info(
        "Pipeline — intent=%s  aggr=%.2f  auto=%s  min_sim=%.2f",
        intent_label,
        effective_aggressiveness,
        auto_mode,
        strategy.min_similarity,
    )

    # --- 2. Candidate generation ----------------------------------------
    generator = CandidateGenerator(model_loader)
    candidate_set = generator.generate(
        original_text=prompt,
        intent_label=intent_label,
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
    # Keep thresholds aligned with configured total-compression guard when enabled.
    if min_total_compression_percent > max_total_compression_percent:
        min_total_compression_percent, max_total_compression_percent = (
            max_total_compression_percent,
            min_total_compression_percent,
        )

    decision_engine = DecisionEngine(
        min_reduction=min_total_compression_percent if enforce_compression_window else 5.0,
        max_reduction=max_total_compression_percent if enforce_compression_window else 70.0,
    )
    selection = decision_engine.select_best(
        candidates=scored_candidates,
        original_text=prompt,
        min_similarity=strategy.min_similarity,
        similarity_weight=strategy.similarity_weight,
        compression_weight=strategy.compression_weight,
        density_weight=strategy.density_weight,
    )

    selected_text = selection["selected_text"]

    # --- 5. GEPA evolutionary optimization on selected baseline ----------
    evaluator = Evaluator(model_loader)
    query = test_query or "Summarize the above context."

    gepa_summary = None
    if use_gepa:
        gepa_optimizer = GepaOptimizer(
            model_loader=model_loader,
            compressor=Compressor(model_loader),
            evaluator=evaluator,
        )
        gepa_result = gepa_optimizer.optimize(
            original_prompt=prompt,
            baseline_prompt=selected_text,
            test_query=query,
            base_aggressiveness=effective_aggressiveness,
            generations=gepa_generations,
            population_size=gepa_population_size,
            time_budget_seconds=gepa_time_budget_seconds,
        )
        selected_text = gepa_result.best_prompt
        gepa_summary = {
            "generations_run": gepa_result.generations_run,
            "frontier_size": gepa_result.frontier_size,
            "summaries": gepa_result.summaries,
        }

    # --- 6. Evolutionary genome integration (fallback + hybrid mode) ----
    method_used = "gepa"
    hybrid_scores = None
    evolved_text = None
    genome_rules = load_best_genome()
    if genome_rules:
        evolved_text = compress_with_genome(prompt, genome_rules, use_cache=True)
        method_used = "evolution"

        # Hybrid mode: compare GEPA-selected output and genome output, keep best.
        if use_gepa:
            gepa_score = score_compression_variant(
                original_prompt=prompt,
                candidate_prompt=selected_text,
                evaluator=evaluator,
                density_metrics=density_metrics,
                test_query=query,
            )
            evo_score = score_compression_variant(
                original_prompt=prompt,
                candidate_prompt=evolved_text,
                evaluator=evaluator,
                density_metrics=density_metrics,
                test_query=query,
            )

            hybrid_scores = {
                "gepa": gepa_score,
                "evolution": evo_score,
            }

            if evo_score["fitness"] >= gepa_score["fitness"]:
                selected_text = evolved_text
                method_used = "evolution"
            else:
                method_used = "gepa"
        else:
            selected_text = evolved_text

    # --- 6b. Controlled second-stage window enforcement -----------------
    controlled_changed_selection = False
    if enforce_compression_window:
        pre_control_text = selected_text
        variant_texts = {
            f"candidate_{cand.strategy}": cand.text
            for cand in candidate_set.candidates
        }
        if evolved_text:
            variant_texts["evolution_genome"] = evolved_text
        variant_texts["selected_hybrid"] = selected_text

        selected_text, controlled_label = _select_window_compliant_output(
            original_prompt=prompt,
            selected_text=selected_text,
            variant_texts=variant_texts,
            evaluator=evaluator,
            density_metrics=density_metrics,
            test_query=query,
            min_total_compression_percent=min_total_compression_percent,
            max_total_compression_percent=max_total_compression_percent,
        )
        controlled_changed_selection = selected_text != pre_control_text

        # Emit explicit method marker for the controlled hybrid path.
        if use_gepa and genome_rules:
            method_used = "genome_plus_controlled_gepa"
        elif controlled_label.startswith("candidate_"):
            method_used = "controlled_candidate"

    # --- 7. Diff + evaluation on the selected output --------------------
    diff_engine = DiffEngine()
    diff_result = diff_engine.compute_diff(original=prompt, compressed=selected_text)

    evaluation = evaluator.evaluate(
        original_prompt=prompt,
        compressed_prompt=selected_text,
        test_query=query,
    )

    # --- 8. Optional GEPA reflective repair -----------------------------
    gepa_repair = None
    gepa_applied = False
    if use_gepa_repair and evaluator.should_trigger_gepa_repair(evaluation):
        logger.info(
            "GEPA repair triggered for drift %.4f (band %.2f-%.2f)",
            evaluation.drift_score,
            decision_engine.conservative_drift,
            decision_engine.max_drift,
        )
        mutator = GepaMutator(model_loader=model_loader, evaluator=evaluator)
        gepa_repair = await mutator.repair_prompt(
            original_prompt=prompt,
            broken_prompt=selected_text,
            test_query=query,
            max_candidates=gepa_candidate_count,
            token_budget_ratio=gepa_token_budget_ratio,
        )

        if gepa_repair.applied:
            gepa_applied = True
            selected_text = gepa_repair.repaired_prompt
            evaluation = gepa_repair.final_report
            diff_result = diff_engine.compute_diff(original=prompt, compressed=selected_text)

    # Also run legacy single-candidate decision for backward-compat fields
    # Find the primary candidate (the selected one) to get token reduction %
    selected_cand = None
    selected_density = None
    if not controlled_changed_selection:
        for cand, dreport in zip(candidate_set.candidates, density_reports):
            if cand.strategy == selection["selected_strategy"]:
                selected_cand = cand
                selected_density = dreport
                break

    # GEPA output is outside initial candidate_set; recompute metrics from text.
    if gepa_applied:
        selected_cand = None
        selected_density = None

    if selected_density is None:
        selected_density = density_metrics.score(prompt, selected_text)

    # Compute token reduction from selected candidate metadata when available,
    # otherwise derive it from the final selected text ratio.
    if selected_cand and selected_cand.token_compression:
        token_reduction_pct = selected_cand.token_compression.token_reduction_percent
    else:
        token_reduction_pct = (1.0 - selected_density.compression_ratio) * 100.0

    legacy_decision = decision_engine.decide(
        token_reduction_percent=token_reduction_pct,
        drift_score=evaluation.drift_score,
        use_gepa_repair=(use_gepa_repair and not gepa_applied),
    )

    # Build densification info from the selected candidate
    densification_info = None
    if selected_cand and selected_cand.densification:
        densification_info = selected_cand.densification.to_dict()

    return {
        "original_prompt": prompt,
        "selected_text": selected_text,
        "mode": mode,
        "method_used": method_used,
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
        "gepa": gepa_summary,
        "hybrid_scores": hybrid_scores,
        "gepa_repair": gepa_repair.to_dict() if gepa_repair is not None else None,
    }


# ==================================================================
# POST /predict
# ==================================================================

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Return a short deterministic continuation for inline ghost text."""
    try:
        model_loader = ModelLoader.get_instance()
        prediction = model_loader.predict_next_tokens(
            text=request.text,
            max_new_tokens=10,
        )
        return PredictResponse(prediction=prediction)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==================================================================
# POST /optimize  (original endpoint — unchanged contract)
# ==================================================================

@router.post("/optimize")
async def optimize(request: OptimizeRequest) -> dict:
    """Compress a prompt and return structured analysis."""
    try:
        result = await _run_pipeline(
            request.prompt,
            request.mode,
            request.aggressiveness,
            request.auto_aggressiveness,
            request.test_query,
            request.intent_override,
            request.use_gepa_repair,
            request.gepa_candidate_count,
            request.gepa_token_budget_ratio,
            request.use_gepa,
            request.gepa_generations,
            request.gepa_population_size,
            request.gepa_time_budget_seconds,
            request.enforce_compression_window,
            request.min_total_compression_percent,
            request.max_total_compression_percent,
        )

        selected_text = result["selected_text"]
        original_prompt = result["original_prompt"]

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

        gepa_resp = None
        if result.get("gepa_repair") is not None:
            gepa_resp = GepaRepairResponse(**result["gepa_repair"])

        # ---- v3: Evaluation Engine (CQS metrics) ----
        from app.evaluation.semantic import compute_semantic_similarity
        from app.evaluation.metrics import instruction_retention_score, information_density
        from app.evaluation.scoring import compression_quality_score

        sem_sim = compute_semantic_similarity(original_prompt, selected_text)
        retention = instruction_retention_score(original_prompt, selected_text)
        info_density = information_density(selected_text)
        comp_ratio = result["density"].compression_ratio if result["density"] is not None else 1.0
        cqs = compression_quality_score(sem_sim, retention, info_density, comp_ratio)

        metrics_resp = EfficiencyMetricsResponse(
            token_reduction_percent=round(result["token_reduction_percent"], 4),
            semantic_similarity=round(sem_sim, 6),
            instruction_retention=round(retention, 6),
            information_density=round(info_density, 6),
            compression_quality_score=cqs,
        )

        # ---- v3: Cost Predictor ----
        from app.cost.cost_model import compare_costs

        cost_data = compare_costs(original_prompt, selected_text)
        cost_resp = CostResponse(**cost_data)

        # ---- v3: 4-class Intent Classifier ----
        from app.intent.classifier import classify_intent

        intent_config = classify_intent(original_prompt)
        prompt_intent = intent_config.label

        # ---- v3: Evolutionary Optimization Variants ----
        from app.evolution.engine import EvolutionaryOptimizer

        evo_optimizer = EvolutionaryOptimizer()
        evo_result = evo_optimizer.optimize(original_prompt, intent_label=None)

        evolution_variants = [
            EvolutionVariantResponse(
                strategy=v.strategy,
                optimized_prompt=v.optimized_prompt,
                semantic_similarity=v.semantic_similarity,
                instruction_retention=v.instruction_retention,
                information_density=v.information_density,
                compression_ratio=v.compression_ratio,
                cqs=v.cqs,
            )
            for v in evo_result.variants
        ]

        # If evolutionary optimizer found a better candidate, use it
        if evo_result.best.cqs > cqs:
            selected_text = evo_result.best.optimized_prompt
            sem_sim = evo_result.best.semantic_similarity
            retention = evo_result.best.instruction_retention
            info_density = evo_result.best.information_density
            cqs = evo_result.best.cqs
            metrics_resp = EfficiencyMetricsResponse(
                token_reduction_percent=round((1.0 - evo_result.best.compression_ratio) * 100.0, 4),
                semantic_similarity=sem_sim,
                instruction_retention=retention,
                information_density=info_density,
                compression_quality_score=cqs,
            )
            cost_data = compare_costs(original_prompt, selected_text)
            cost_resp = CostResponse(**cost_data)

        response = OptimizeResponse(
            mode=result["mode"],
            compressed_prompt=selected_text,
            original_token_count=result["density"].original_token_count if result["density"] is not None else 0,
            compressed_token_count=result["density"].compressed_token_count if result["density"] is not None else 0,
            compression_ratio=round(comp_ratio, 4),
            token_reduction_percent=round(result["token_reduction_percent"], 4),
            intent=IntentResponse(**result["intent"]),
            diff=DiffResponse(**result["diff"].to_dict()),
            evaluation=EvaluationResponse(**result["evaluation"].to_dict()),
            decision=DecisionResponse(**result["decision"]),
            density=density_resp,
            densification=densification_resp,
            candidates=candidate_responses,
            selection=selection_resp,
            gepa_repair=gepa_resp,
            # v3 fields
            metrics=metrics_resp,
            cost=cost_resp,
            prompt_intent=prompt_intent,
            evolution_variants=evolution_variants,
        )
        payload = response.model_dump()
        payload["original_prompt"] = original_prompt
        payload["optimized_prompt"] = selected_text
        payload["method_used"] = result["method_used"]
        return payload
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
        result = await _run_pipeline(
            request.prompt,
            request.mode,
            request.aggressiveness,
            request.auto_aggressiveness,
            request.test_query,
            request.intent_override,
            request.use_gepa_repair,
            request.gepa_candidate_count,
            request.gepa_token_budget_ratio,
            request.use_gepa,
            request.gepa_generations,
            request.gepa_population_size,
            request.gepa_time_budget_seconds,
            request.enforce_compression_window,
            request.min_total_compression_percent,
            request.max_total_compression_percent,
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
# POST /analyze
# ==================================================================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the five-task prompt analysis pipeline.

    Returns intent summary, optimized prompt, comparison scores,
    template detection, and a one-line improvement summary.
    """
    try:
        # 1. Run the optimisation pipeline to get a compressed prompt
        result = await _run_pipeline(
            request.prompt,
            request.mode,
            request.aggressiveness,
            request.auto_aggressiveness,
            intent_override=request.intent_override,
        )

        optimized = result["selected_text"]

        # 2. Run the prompt analyzer
        analyzer = PromptAnalyzer()
        report = analyzer.analyze(
            original_prompt=request.prompt,
            optimized_prompt=optimized,
        )

        # Build density sub-response
        density_resp = None
        if result["density"] is not None:
            density_resp = DensityResponse(**result["density"].to_dict())

        return AnalyzeResponse(
            intent=report.intent,
            optimized_prompt=report.optimized_prompt,
            comparison=[
                DimensionScoreResponse(**d.to_dict()) for d in report.comparison
            ],
            template=TemplateInfoResponse(**report.template.to_dict()),
            summary=report.summary,
            compute_original=ComputeScoreResponse(**report.compute_original.to_dict()),
            compute_optimized=ComputeScoreResponse(**report.compute_optimized.to_dict()),
            compute_reduction_percent=report.compute_reduction_percent,
            original_token_count=(
                result["density"].original_token_count
                if result["density"] is not None else 0
            ),
            compressed_token_count=(
                result["density"].compressed_token_count
                if result["density"] is not None else 0
            ),
            compression_ratio=round(
                result["density"].compression_ratio, 4
            ) if result["density"] is not None else 1.0,
            mode=result["mode"],
            intent_detail=IntentResponse(**result["intent"]),
            density=density_resp,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ==================================================================
# POST /evaluate_prompt
# ==================================================================

@router.post("/evaluate_prompt", response_model=EvaluatePromptResponse)
async def evaluate_prompt(request: EvaluatePromptRequest) -> EvaluatePromptResponse:
    """Evaluate original vs optimised prompt responses and persist the outcome."""
    try:
        optimized_prompt = request.optimized_prompt
        if not optimized_prompt:
            optimization = await _run_pipeline(
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


# ==================================================================
# POST /optimize-pipeline  (multi-stage pipeline)
# ==================================================================

@router.post("/optimize-pipeline", response_model=PipelineResponse)
async def optimize_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Run the full 9-stage prompt optimisation pipeline.

    Stages: regex cleaning → structural simplification → tokenization +
    surprisal → GEPA scoring → adaptive pruning → reconstruction →
    semantic validation → metrics → template extraction.
    """
    try:
        from app.core.pipeline import PromptPipeline

        model_loader = ModelLoader.get_instance()
        pipeline = PromptPipeline(model_loader)
        result = pipeline.run(
            text=request.prompt,
            intent_label=request.intent_override,
        )

        return PipelineResponse(
            original_prompt=result.original_prompt,
            optimized_prompt=result.optimized_prompt,
            original_token_count=result.original_token_count,
            optimized_token_count=result.optimized_token_count,
            compression_ratio=round(result.compression_ratio, 4),
            information_density=round(result.information_density, 4),
            semantic_similarity=round(result.semantic_similarity, 4),
            pipeline_accepted=result.pipeline_accepted,
            template=PipelineTemplateResponse(
                template=result.template.get("template", ""),
                variables=result.template.get("variables", []),
            ),
            stages_applied=result.stages_applied,
        )
    except Exception as exc:
        logger.exception("Pipeline optimisation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
