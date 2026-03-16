# OptiPrompt

OptiPrompt is a FastAPI-powered prompt optimization engine with a Chrome extension frontend.
It compresses and restructures prompts, generates multiple candidates, scores them for density and drift,
and returns the best version with traceable evaluation data.

## Features

- Prompt optimization with configurable or auto aggressiveness
- Multi-candidate generation and selection
- GEPA evolutionary post-optimization (Pareto frontier on drift vs reduction)
- Semantic drift and response-quality evaluation
- Structured diff output
- Prompt storage and retrieval endpoints
- Chrome extension integration (Manifest V3)

## GEPA Architecture

The optimize pipeline now includes a GEPA layer after baseline candidate selection:

`Prompt -> Baseline Compression -> GEPA Evolution -> Evaluation -> Decision`

GEPA module layout:

```text
app/core/gepa/
├── optimizer.py      # Main evolutionary loop
├── population.py     # Candidate and metrics data models
├── mutation.py       # Mutation operators
├── pareto.py         # Pareto frontier selection
├── reflection.py     # Lightweight convergence/adaptation logic
├── mutator.py        # Existing reflective repair module (kept for compatibility)
└── reflection_llm.py # Existing reflection provider abstraction
```

## Tech Stack

- Python 3.10+
- FastAPI + Uvicorn
- Pydantic v2
- PyTorch + Hugging Face Transformers
- Chrome Extension (MV3)

## Quick Start

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run API
uvicorn app.main:app --reload

# 3) Smoke test
curl -s http://127.0.0.1:8000/health
```

API docs are available at `http://127.0.0.1:8000/docs`.

## Extension Setup (Existing)

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click `Load unpacked` and select `extension/`.
4. Start the backend (`uvicorn app.main:app --reload`).
5. Open ChatGPT or Claude and use the extension popup.

By default, `extension/background.js` points to your configured backend.

The extension popup now includes GEPA controls (`use_gepa`, generations,
population size, time budget) and a configurable backend URL.

## API Endpoints

- `GET /health`
- `POST /optimize`
- `POST /analyze`
- `POST /optimize-and-store`
- `GET /prompts`
- `POST /prompts`
- `GET /prompts/{prompt_id}`
- `POST /evaluate_prompt`

Minimal optimize request:

```json
{
  "prompt": "Explain relativity simply for a beginner.",
  "mode": "balanced",
  "auto_aggressiveness": true
}
```

## Project Structure

```text
opti_prompt/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   ├── storage/
│   └── utils/
├── extension/
│   ├── background.js
│   ├── content.js
│   ├── popup/
│   └── manifest.json
├── frontend/
├── cli.py
├── requirements.txt
└── README.md
```

## CORS Troubleshooting (Chrome Extension)

If you see an error like:

```text
Access to fetch at 'https://<your-app>.azurewebsites.net/health' from origin
'chrome-extension://<id>' has been blocked by CORS policy
```

check the following:

1. Ensure backend CORS allows Chrome extension origins.
2. Ensure remote web origins include protocol (`https://...`) in `ALLOWED_ORIGINS`.
3. Ensure extension `host_permissions` include your API host.
4. Reload the extension after manifest or code changes.

Current backend supports extension origins via regex:

```python
ALLOWED_ORIGIN_REGEX = r"^chrome-extension://.*$"
```

Important: In `app/main.py`, origin values in `ALLOWED_ORIGINS` must be full origins,
for example:

```python
"https://optiprompt-gqd9hqf6dffvaacb.eastasia-01.azurewebsites.net"
```

not just:

```python
"optiprompt-gqd9hqf6dffvaacb.eastasia-01.azurewebsites.net"
```

## CLI

```bash
python cli.py --help
```

## License

Add your preferred license information here.
