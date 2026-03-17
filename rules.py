from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Dict, List


RuleFn = Callable[[str], str]


@lru_cache(maxsize=200000)
def _cached_apply_rule(rule_name: str, text: str) -> str:
    return RULE_FUNCTIONS[rule_name](text)


def apply_rule(rule_name: str, text: str, use_cache: bool = True) -> str:
    if use_cache:
        return _cached_apply_rule(rule_name, text)
    return RULE_FUNCTIONS[rule_name](text)


def remove_greeting(text: str) -> str:
    return re.sub(
        r"^\s*(hi|hello|hey|dear\s+\w+|greetings)[,!.\s]+",
        "",
        text,
        flags=re.IGNORECASE,
    )


def remove_filler(text: str) -> str:
    fillers = [
        r"\bin order to\b",
        r"\bbasically\b",
        r"\bactually\b",
        r"\bjust\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\bliterally\b",
        r"\bvery\b",
        r"\breally\b",
        r"\bI think\b",
        r"\bI believe\b",
        r"\bI would say\b",
        r"\bto be honest\b",
        r"\bto tell you\s+the truth\b",
        r"\bfairly\b",
        r"\bquite\b",
        r"\brather\b",
        r"\bsomewhat\b",
        r"\bapparently\b",
        r"\bseemingly\b",
        r"\bin my opinion\b",
        r"\bin my view\b",
        r"\bmore or less\b",
    ]
    out = text
    for pat in fillers:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def remove_hedging(text: str) -> str:
    hedges = [
        r"\bit seems\b",
        r"\bit appears\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bprobably\b",
        r"\bI think\b",
        r"\bI believe\b",
    ]
    out = text
    for pat in hedges:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def merge_duplicates(text: str) -> str:
    # Remove immediate repeated words and duplicate adjacent sentences.
    out = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", out)
    deduped: List[str] = []
    prev_norm = ""
    for s in sentences:
        norm = re.sub(r"\W+", "", s.lower())
        if norm and norm != prev_norm:
            deduped.append(s)
        prev_norm = norm
    return " ".join(deduped).strip()


def remove_adverbs(text: str) -> str:
    # Remove adverbs that don't change core meaning (especially -ly adverbs and temporal modifiers).
    out = re.sub(r"\b\w+ly\b", "", text, flags=re.IGNORECASE)
    out = re.sub(r"\b(frequently|occasionally|rarely|often|seldom|typically|generally|usually|always|never)\b", "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def clause_level_prune(text: str) -> str:
    # Aggressively drop parenthetical, relative clauses, and appositive phrases.
    out = re.sub(r"\s*\([^)]{0,200}\)", "", text)
    out = re.sub(r",\s+(which|that|who|where|when|as|because|since|although)\b[^,.;:!?]{0,150}", "", out, flags=re.IGNORECASE)
    out = re.sub(r";\s*[a-z][^.!?]{0,120}", "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def preserve_entities(text: str) -> str:
    # Marker rule for composition ordering; no-op transformation.
    return text


def preserve_numbers(text: str) -> str:
    # Marker rule for composition ordering; no-op transformation.
    return text


def keep_list_structure(text: str) -> str:
    # Normalize bullet/list spacing while preserving list semantics.
    out = re.sub(r"\n\s*[-*]\s*", "\n- ", text)
    out = re.sub(r"\n\s*\d+[.)]\s*", "\n1. ", out)
    return out


def aggressive_sentence_filter(text: str) -> str:
    """Remove sentences that are primarily examples, elaborations, or redundant filler."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept: List[str] = []
    
    # Patterns indicating removable sentences
    remove_patterns = [
        r"^(for example|e\.g\.|for instance|such as|including|like)",
        r"(e\.g\.|for example|for instance)" ,
        r"^(note that|note:|please note)",
        r"^(as mentioned|as discussed|as noted)",
        r"^(in this context|in this case|in other words)",
    ]
    
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        
        # Skip if sentence matches removable pattern
        skip = False
        for pat in remove_patterns:
            if re.search(pat, s_clean, flags=re.IGNORECASE):
                skip = True
                break
        
        if not skip:
            kept.append(s_clean)
    
    return " ".join(kept).strip()


RULE_LIBRARY: List[str] = [
    "REMOVE_GREETING",
    "REMOVE_FILLER",
    "REMOVE_HEDGING",
    "MERGE_DUPLICATES",
    "REMOVE_ADVERBS",
    "CLAUSE_LEVEL_PRUNE",
    "PRESERVE_ENTITIES",
    "PRESERVE_NUMBERS",
    "KEEP_LIST_STRUCTURE",
    "AGGRESSIVE_SENTENCE_FILTER",
]


RULE_FUNCTIONS: Dict[str, RuleFn] = {
    "REMOVE_GREETING": remove_greeting,
    "REMOVE_FILLER": remove_filler,
    "REMOVE_HEDGING": remove_hedging,
    "MERGE_DUPLICATES": merge_duplicates,
    "REMOVE_ADVERBS": remove_adverbs,
    "CLAUSE_LEVEL_PRUNE": clause_level_prune,
    "PRESERVE_ENTITIES": preserve_entities,
    "PRESERVE_NUMBERS": preserve_numbers,
    "KEEP_LIST_STRUCTURE": keep_list_structure,
    "AGGRESSIVE_SENTENCE_FILTER": aggressive_sentence_filter,
}
