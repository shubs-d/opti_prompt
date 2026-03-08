<<<<<<< HEAD
# OptiPrompt - Prompt Compressor and Optimiser

Agentic AI backend that compresses prompts using **token-level surprisal scoring**, evaluates quality drift, and produces structured diff output.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | PyTorch, Hugging Face Transformers |
| API | FastAPI, Uvicorn |
| Validation | Pydantic v2 |
| CLI | Typer |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload

# Test the endpoint
curl -s -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the theory of relativity in simple terms for a beginner student who has no background in physics.", "aggressiveness": 0.3}' | python -m json.tool
```

## CLI Usage

```bash
# Write a prompt to a file
echo "Explain the theory of relativity in simple terms." > prompt.txt

# Run the optimizer
python cli.py optimize --file prompt.txt --aggressiveness 0.4
```

## API Reference

### `POST /optimize`

**Request body:**

```json
{
  "prompt": "Your prompt text here.",
  "aggressiveness": 0.3
}
```

**Response:**

```json
{
  "compressed_prompt": "...",
  "token_reduction_percent": 18.75,
  "diff": {
    "removed": [],
    "rewritten": [],
    "preserved": []
  },
  "evaluation": {
    "semantic_similarity": 0.9812,
    "length_difference": 24,
    "length_ratio": 0.82,
    "drift_score": 0.0188,
    "original_response": "...",
    "compressed_response": "..."
  },
  "decision": {
    "decision": "APPROVE",
    "reason": "...",
    "token_reduction_percent": 18.75,
    "drift_score": 0.0188
  }
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Architecture

```
context_optimizer/
├── app/
│   ├── main.py                 # FastAPI entry + lifespan
│   ├── api/
│   │   ├── routes.py           # POST /optimize endpoint
│   │   └── schemas.py          # Pydantic request/response models
│   ├── core/
│   │   ├── model_loader.py     # Causal LM loader + surprisal computation
│   │   ├── compressor.py       # Token-level prompt compression
│   │   ├── diff_engine.py      # Structured diff generation
│   │   ├── evaluator.py        # Quality-drift evaluation
│   │   └── decision_engine.py  # Approve / Reject gate
│   └── utils/
│       ├── token_utils.py      # Token manipulation helpers
│       └── similarity.py       # Cosine similarity + embeddings
├── cli.py                      # Typer CLI wrapper
├── requirements.txt
└── README.md
```

## How It Works

1. **Tokenize** the input prompt using a causal language model (default: `distilgpt2`).
2. **Score** each token by surprisal (negative log-probability given left context).
3. **Prune** low-surprisal tokens that carry little information, controlled by the `aggressiveness` parameter.
4. **Diff** the original and compressed prompts (line-level + token-level).
5. **Evaluate** semantic drift via cosine similarity of mean-pooled hidden states.
6. **Decide** whether to APPROVE, REJECT, or flag as CONSERVATIVE_REQUIRED.

## Decision Thresholds

| Metric | Threshold | Outcome |
|--------|-----------|---------|
| Drift > 0.15 | Hard ceiling | REJECT |
| Reduction > 70% | Too aggressive | REJECT |
| Reduction < 5% | Not worth it | REJECT |
| Drift > 0.08 | Caution zone | CONSERVATIVE_REQUIRED |
| Otherwise | All clear | APPROVE |
=======
# opti_prompt
>>>>>>> 71fd456b (first commit)
# opti_prompt
# opti_prompt
