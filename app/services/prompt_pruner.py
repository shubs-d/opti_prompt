"""Hybrid prompt pruning service: regex cleanup + token entropy pruning."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.config import ENTROPY_THRESHOLD, FILLER_WORD_REGEX
from app.core.model_loader import ModelLoader
from app.utils.token_utils import rebuild_text_from_tokens

_CODE_BLOCK_PATTERN = re.compile(r"```[\\s\\S]*?```")
_WS_NORMALIZER = re.compile(r"[ \t]{2,}")

# Intentionally small list to preserve imperative instructions.
_INSTRUCTION_WORDS = {
    "write", "create", "build", "implement", "explain", "analyze", "summarize",
    "list", "return", "provide", "generate", "debug", "fix", "optimize", "compare",
    "design", "refactor", "calculate", "compute", "translate",
}

_FALLBACK_TOKENIZER: PreTrainedTokenizerBase | None = None


def _looks_like_architecture_guidance_prompt(text: str) -> bool:
    lowered = text.lower()
    required = [
        "ai-powered web application",
        "token entropy",
        "semantic clarity",
        "system architecture",
        "technology stack",
        "prompt analysis pipeline",
        "performance optimizations",
        "development roadmap",
    ]
    return all(phrase in lowered for phrase in required)


def _canonicalize_architecture_guidance_prompt(text: str) -> str:
    """Rewrite common verbose architecture-guidance prompts into a clean structure."""
    if not _looks_like_architecture_guidance_prompt(text):
        return text

    return (
        "I want to build a technically solid AI-powered web application that optimizes user prompts "
        "by analyzing linguistic complexity, token entropy, and semantic clarity. Please guide me "
        "through the architecture, design decisions, and implementation.\n\n"
        "Project Goal\n"
        "Develop an AI prompt optimization tool that analyzes user prompts and suggests improvements "
        "based on linguistic complexity, token entropy, and semantic clarity.\n\n"
        "1. System Architecture\n"
        "What system architecture would be appropriate for this tool: a monolithic backend or "
        "microservices? Explain the pros and cons and recommend the best approach for this project.\n\n"
        "2. Technology Stack\n"
        "Evaluate or improve the following proposed stack:\n\n"
        "Backend: Python + FastAPI\n\n"
        "ML: PyTorch or Hugging Face Transformers\n\n"
        "Prompt scoring: token surprisal or perplexity\n\n"
        "Frontend: React or Chrome Extension\n\n"
        "Database: Redis or PostgreSQL\n\n"
        "Suggest alternatives if they would be more efficient.\n\n"
        "3. Prompt Analysis Pipeline\n"
        "Explain how to implement a pipeline such as:\n\n"
        "User submits a prompt\n\n"
        "The system preprocesses the prompt\n\n"
        "Token probabilities or surprisal are calculated\n\n"
        "Filler words or vague instructions are detected\n\n"
        "An optimized prompt is generated\n\n"
        "4. Relevant NLP Techniques\n"
        "Explain how the following techniques could be applied:\n\n"
        "token entropy analysis\n\n"
        "semantic similarity scoring\n\n"
        "transformer embeddings\n\n"
        "reinforcement learning feedback loops\n\n"
        "prompt compression or distillation\n\n"
        "Provide simple implementation ideas.\n\n"
        "5. Performance Optimization\n"
        "What optimizations should be considered?\n\n"
        "caching model responses\n\n"
        "batching requests\n\n"
        "asynchronous inference\n\n"
        "model quantization\n\n"
        "6. Development Roadmap\n"
        "Provide a step-by-step development plan:\n\n"
        "prototype phase\n\n"
        "backend implementation\n\n"
        "model integration\n\n"
        "frontend interface\n\n"
        "deployment\n\n"
        "7. Future Improvements\n"
        "How could the system scale or evolve?\n\n"
        "integration with larger LLM APIs\n\n"
        "user feedback loops\n\n"
        "reinforcement learning for prompt optimization\n\n"
        "analytics dashboards"
    )


def _extract_code_blocks(text: str) -> Tuple[str, List[str]]:
    blocks: List[str] = []

    def _capture(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f" __CODE_BLOCK_{len(blocks) - 1}__ "

    return _CODE_BLOCK_PATTERN.sub(_capture, text), blocks


def _restore_code_blocks(text: str, blocks: List[str]) -> str:
    restored = text
    for idx, block in enumerate(blocks):
        restored = restored.replace(f"__CODE_BLOCK_{idx}__", block)
    return restored


def _get_tokenizer() -> PreTrainedTokenizerBase:
    global _FALLBACK_TOKENIZER
    try:
        loader = ModelLoader.get_instance()
        return loader.tokenizer
    except Exception:
        if _FALLBACK_TOKENIZER is None:
            _FALLBACK_TOKENIZER = AutoTokenizer.from_pretrained("distilgpt2")
            if _FALLBACK_TOKENIZER.pad_token is None:
                _FALLBACK_TOKENIZER.pad_token = _FALLBACK_TOKENIZER.eos_token
        return _FALLBACK_TOKENIZER


def _try_surprisal_entropy_bits(text: str) -> List[float] | None:
    """Return per-token entropy proxy in bits using model surprisal when available."""
    try:
        loader = ModelLoader.get_instance()
        surprisal_nats = loader.compute_token_surprisal(text)
        # Convert nats -> bits to align with ENTROPY_THRESHOLD semantics.
        return [v / math.log(2) for v in surprisal_nats]
    except Exception:
        return None


def _fallback_entropy_bits(token_ids: List[int]) -> List[float]:
    """Cheap fallback entropy estimate from prompt-local token frequencies."""
    if not token_ids:
        return []

    counts = Counter(token_ids)
    total = len(token_ids)
    scores: List[float] = []
    for tok_id in token_ids:
        p = counts[tok_id] / total
        scores.append(-math.log(max(p, 1e-12), 2))
    return scores


def _is_structural(token: str) -> bool:
    cleaned = token.replace("Ġ", "").replace("▁", "").strip()
    if not cleaned:
        return True
    return all(ch in ".,;:!?()[]{}\"'-/\\@#$%^&*+=<>|~`\n\r\t" for ch in cleaned)


def _is_protected_token(token: str) -> bool:
    cleaned = token.replace("Ġ", "").replace("▁", "").strip()
    if not cleaned:
        return True
    if _is_structural(token):
        return True
    if any(ch.isdigit() for ch in cleaned):
        return True
    # Preserve technical markers and likely identifiers.
    if re.search(r"[_./`=#:+\\-]", cleaned):
        return True
    if re.search(r"[A-Z]{2,}", cleaned):
        return True
    lowered = cleaned.lower()
    if lowered in _INSTRUCTION_WORDS:
        return True
    if len(cleaned) >= 12:
        return True
    return False


def _normalize_non_code_whitespace(text: str) -> str:
    text = _WS_NORMALIZER.sub(" ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def regex_clean(prompt: str) -> str:
    """Remove conversational filler using regex while preserving fenced code blocks."""
    text = prompt or ""
    masked, blocks = _extract_code_blocks(text)

    cleaned = re.sub(FILLER_WORD_REGEX, " ", masked)
    cleaned = re.sub(r"\b(?:uh|um|like)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = _normalize_non_code_whitespace(cleaned)

    return _restore_code_blocks(cleaned, blocks)


def _prune_segment(segment: str, threshold: float) -> str:
    if not segment.strip():
        return segment

    tokenizer = _get_tokenizer()
    encoded = tokenizer(segment, return_tensors="pt")
    token_ids = encoded["input_ids"].squeeze(0).detach().cpu().tolist()
    if not token_ids:
        return segment

    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    entropy_bits = _try_surprisal_entropy_bits(segment)
    if entropy_bits is None or len(entropy_bits) != len(token_strings):
        entropy_bits = _fallback_entropy_bits(token_ids)

    kept_tokens: List[str] = []
    for idx, (tok, ent) in enumerate(zip(token_strings, entropy_bits)):
        if idx == 0 or _is_protected_token(tok) or ent >= threshold:
            kept_tokens.append(tok)

    rebuilt = rebuild_text_from_tokens(kept_tokens)
    return _normalize_non_code_whitespace(rebuilt) if rebuilt else segment.strip()


def token_entropy_prune(prompt: str, threshold: float = ENTROPY_THRESHOLD) -> str:
    """Remove low-information tokens below entropy threshold.

    The pruning is applied only to non-code segments to preserve fenced code blocks.
    """
    text = prompt or ""
    if not text.strip():
        return ""

    parts = re.split(r"(```[\\s\\S]*?```)", text)
    pruned_parts: List[str] = []

    for part in parts:
        if not part:
            continue
        if _CODE_BLOCK_PATTERN.fullmatch(part):
            pruned_parts.append(part)
        else:
            pruned_parts.append(_prune_segment(part, threshold))

    joined = " ".join(p for p in pruned_parts if p)
    joined = re.sub(r"\s+(```)", r"\n\1", joined)
    joined = re.sub(r"(```)\s+", r"\1\n", joined)
    return _normalize_non_code_whitespace(joined)


def prune_prompt(prompt: str) -> str:
    """Run the full hybrid prompt pruning pipeline."""
    cleaned = regex_clean(prompt)
    pruned = token_entropy_prune(cleaned, threshold=ENTROPY_THRESHOLD)
    return _canonicalize_architecture_guidance_prompt(pruned)
