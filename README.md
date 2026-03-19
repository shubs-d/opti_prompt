# ⚡ OptiPrompt — Adaptive LLM Efficiency Engine

OptiPrompt is a **production-grade prompt optimization engine** that compresses, evaluates, and restructures LLM prompts using information-theoretic methods. It combines entropy-based token pruning (GEPA), evolutionary optimization, semantic evaluation, and cost estimation into a single API.

> **TL;DR** — Send a verbose prompt, get back a compressed one with measurable quality scores, cost savings, and full evaluation metrics.

---

## 🧠 Architecture

```text
Input Prompt
  → Regex Cleaning (filler removal)
  → Structural Simplification
  → Intent Detection (coding / creative / qa / general)
  → Adaptive GEPA (entropy-based token pruning, intent-aware thresholds)
  → Multi-Candidate Generation (aggressive / balanced / structure-focused)
  → Evolutionary Optimization (CQS-scored variant selection)
  → Evaluation Engine (semantic similarity + instruction retention + info density)
  → Cost Estimation (GPT-4 / GPT-3.5 / Claude pricing)
  → Best Variant Selection
```

---

## ✨ Features

### Core Compression
- **9-stage pipeline** — regex cleaning, structural simplification, tokenization, surprisal scoring, GEPA pruning, reconstruction, semantic validation, metrics, template extraction
- **Intent-aware GEPA** — adjusts pruning aggressiveness per prompt type (preserves syntax tokens for code, adjectives for creative)
- **Multi-candidate generation** — aggressive, balanced, semantic, and structured strategies scored against each other
- **GEPA evolutionary post-optimization** — Pareto frontier on drift vs. reduction

### Advanced Evaluation Engine
- **Semantic Similarity** — cosine similarity via `all-MiniLM-L6-v2` (sentence-transformers)
- **Instruction Retention Score** — measures % of key verbs/nouns preserved after compression
- **Information Density** — ratio of meaningful tokens to total tokens
- **Compression Quality Score (CQS)** — weighted composite:
  ```
  CQS = 0.4 × semantic_similarity + 0.2 × instruction_retention
      + 0.2 × information_density + 0.2 × (1 - compression_ratio)
  ```

### Compute Cost Predictor
- Token-accurate cost estimation using `tiktoken` (cl100k_base encoding)
- Configurable pricing for GPT-4, GPT-3.5-turbo, Claude
- Returns original cost, optimized cost, and savings percentage

### Evolutionary Optimization Loop
- Generates 3 compression variants at different aggressiveness levels
- Evaluates each with CQS
- Selects the highest-scoring candidate automatically

### Chrome Extension (MV3)
- Popup UI with configurable backend URL
- Grammarly-style inline suggestions via content script
- GEPA controls (generations, population size, time budget)

---

## 🔌 API Response Format

`POST /optimize` returns:

```json
{
  "original_prompt": "Can you please help me write a Python function...",
  "optimized_prompt": "write Python function list integers sum even...",

  "metrics": {
    "token_reduction_percent": 57.5,
    "semantic_similarity": 0.9259,
    "instruction_retention": 0.5238,
    "information_density": 0.8571,
    "compression_quality_score": 0.7616
  },

  "cost": {
    "original_cost": 0.0012,
    "optimized_cost": 0.00051,
    "savings_percent": 57.5,
    "model": "gpt-4"
  },

  "prompt_intent": "coding",

  "intent": { "intent": "CONVERSATIONAL", "intent_confidence": 0.7, "..." : "..." },
  "evaluation": { "semantic_similarity": 0.98, "drift_score": 0.02, "..." : "..." },
  "density": { "density_score": 0.87, "compression_ratio": 0.425, "..." : "..." },
  "evolution_variants": [ "..." ]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10+, FastAPI, Uvicorn, Gunicorn |
| ML Models | PyTorch, HuggingFace Transformers (distilgpt2), sentence-transformers (all-MiniLM-L6-v2) |
| Tokenization | tiktoken (cl100k_base) |
| Validation | Pydantic v2 |
| Frontend | Chrome Extension (Manifest V3) |

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd opti_prompt
pip install -r requirements.txt

# 2. Run the API server
uvicorn app.main:app --reload

# 3. Test the /optimize endpoint
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you please help me write a Python function that takes a list of integers and returns the sum of all even numbers?"}'

# 4. Health check
curl http://localhost:8000/health
```

API docs available at `http://localhost:8000/docs`.

---

## 📁 Project Structure

```text
opti_prompt/
├── app/
│   ├── main.py                  # FastAPI entry point + lifespan
│   ├── config.py                # Global configuration constants
│   ├── api/
│   │   ├── routes.py            # All API endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   ├── core/
│   │   ├── pipeline.py          # 9-stage optimization pipeline
│   │   ├── model_loader.py      # distilgpt2 singleton loader
│   │   ├── compressor.py        # Token-level compression
│   │   ├── candidate_generator.py  # Multi-strategy variant generation
│   │   ├── decision_engine.py   # Candidate selection logic
│   │   ├── density_metrics.py   # Information density scoring
│   │   ├── evaluator.py         # Semantic drift evaluation
│   │   ├── intent_engine.py     # 5-class intent detection (existing)
│   │   ├── diff_engine.py       # Structured diff output
│   │   ├── template_extractor.py
│   │   └── gepa/               # Evolutionary optimization
│   │       ├── optimizer.py     # Main GEPA evolutionary loop
│   │       ├── population.py    # Candidate data models
│   │       ├── mutation.py      # Mutation operators
│   │       ├── pareto.py        # Pareto frontier selection
│   │       └── reflection.py    # Convergence/adaptation logic
│   ├── evaluation/              # ⭐ Advanced Evaluation Engine
│   │   ├── semantic.py          # Sentence-transformer similarity
│   │   ├── metrics.py           # Instruction retention + info density
│   │   └── scoring.py           # CQS composite scoring
│   ├── cost/                    # ⭐ Compute Cost Predictor
│   │   └── cost_model.py        # Token-based pricing (GPT-4/3.5/Claude)
│   ├── intent/                  # ⭐ 4-Class Intent Classifier
│   │   └── classifier.py        # coding / creative / qa / general
│   ├── evolution/               # ⭐ Evolutionary Optimization Loop
│   │   └── engine.py            # Variant generation + CQS selection
│   ├── services/
│   │   └── prompt_pruner.py
│   ├── storage/
│   │   ├── prompt_repository.py
│   │   └── evaluation_repository.py
│   └── utils/
│       ├── similarity.py
│       └── token_utils.py
├── extension/                   # Chrome Extension (MV3)
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   └── popup/
├── frontend/                    # Bundled web frontend
├── cli.py                       # CLI interface
├── requirements.txt
└── README.md
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/optimize` | Full optimization with metrics, cost, and intent |
| `POST` | `/optimize-pipeline` | 9-stage pipeline (lightweight) |
| `POST` | `/optimize-and-store` | Optimize + persist to storage |
| `POST` | `/analyze` | Five-task prompt analysis |
| `POST` | `/evaluate_prompt` | Compare original vs. optimized responses |
| `POST` | `/predict` | Inline ghost-text prediction |
| `GET` | `/prompts` | List all stored prompts |
| `POST` | `/prompts` | Manually store a prompt |
| `GET` | `/prompts/{id}` | Retrieve prompt by ID |

### Minimal Request

```json
{
  "prompt": "Explain relativity simply for a beginner.",
  "mode": "optimize",
  "auto_aggressiveness": true
}
```

---

## ☁️ Deployment

### Azure App Service

```bash
# Create resources
az group create --name <rg> --location <region>
az appservice plan create --name <plan> --resource-group <rg> --sku B1 --is-linux
az webapp create --name <app> --resource-group <rg> --plan <plan> --runtime "PYTHON|3.12"

# Configure startup
az webapp config set --resource-group <rg> --name <app> \
  --startup-file "gunicorn -k uvicorn.workers.UvicornWorker --bind=0.0.0.0:\$PORT app.main:app"

# Deploy
az webapp up --name <app> --resource-group <rg>
```

### Modal

```bash
modal deploy modal_deploy.py
```

---

## 🧩 Chrome Extension Setup

1. Open `chrome://extensions` → Enable **Developer mode**
2. Click **Load unpacked** → Select `extension/`
3. Start backend: `uvicorn app.main:app --reload`
4. Open ChatGPT or Claude and use the extension popup

---

## 🔧 CORS Configuration

The backend allows Chrome/Firefox extension origins via regex:

```python
ALLOWED_ORIGIN_REGEX = r"^(chrome-extension|moz-extension)://.*$"
```

Add your deployment URL to `ALLOWED_ORIGINS` in `app/main.py` (must include protocol):

```python
"https://your-app.azurewebsites.net"
```

---

## 📜 License

Add your preferred license information here.
