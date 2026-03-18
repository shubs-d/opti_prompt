"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.routes import router
from app.config import ENTROPY_THRESHOLD
from app.core.model_loader import ModelLoader
from app.services.prompt_pruner import prune_prompt, regex_clean, token_entropy_prune
from app.storage.evaluation_repository import EvaluationRepository
from app.storage.prompt_repository import PromptRepository

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

# ------------------------------------------------------------------
# CORS — origins allowed for local frontend + browser extension
# ------------------------------------------------------------------
ALLOWED_ORIGINS = [
    
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://chatgpt.com",
    "https://optiprompt-gqd9hqf6dffvaacb.eastasia-01.azurewebsites.net",
    "https://claude.ai",
]

# Chrome extensions send Origin: chrome-extension://<id>
# Firefox extensions send Origin: moz-extension://<id>
ALLOWED_ORIGIN_REGEX = r"^(chrome-extension|moz-extension)://.*$"


# ------------------------------------------------------------------
# Lifespan — load model + init storage once on startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Pre-load the language model and initialise storage."""
    preload_model = _is_truthy(os.getenv("PRELOAD_MODEL_ON_STARTUP", "false"))
    if preload_model:
        logger.info("Loading model on startup …")
        try:
            ModelLoader.get_instance()
            logger.info("Model ready.")
        except Exception:
            # Keep the process alive so /health works; model can load lazily later.
            logger.exception("Model preload failed; continuing without preloaded model")
    else:
        logger.info("Skipping model preload (PRELOAD_MODEL_ON_STARTUP=false)")

    logger.info("Initialising prompt storage …")
    PromptRepository.get_instance()
    logger.info("Storage ready.")

    logger.info("Initialising evaluation storage …")
    EvaluationRepository.get_instance()
    logger.info("Evaluation storage ready.")

    yield
    logger.info("Shutting down.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="LLM Context Optimization Engine",
    version="3.0.0",
    description=(
        "Prompt compiler that compresses, densifies, and restructures "
        "prompts using token-level surprisal scoring, phrase-level "
        "rewriting, multi-candidate generation, and information-density "
        "scoring.  Selects the optimal candidate via a composite "
        "quality gate informed by intent-specific strategies."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


class OptimizePromptRequest(BaseModel):
    """Request body for lightweight prompt preprocessing endpoint."""

    prompt: str = Field(..., min_length=1)
    entropy_threshold: float = Field(default=ENTROPY_THRESHOLD, ge=0.0)


class OptimizePromptResponse(BaseModel):
    """Response body for lightweight prompt preprocessing endpoint."""

    optimized_prompt: str


@app.post("/optimize_prompt", response_model=OptimizePromptResponse)
async def optimize_prompt(request: OptimizePromptRequest) -> OptimizePromptResponse:
    """Apply regex filler cleanup + token entropy pruning."""
    # Keep explicit pipeline stages for threshold-aware pruning.
    cleaned = regex_clean(request.prompt)
    _ = token_entropy_prune(cleaned, threshold=request.entropy_threshold)
    optimized = prune_prompt(request.prompt)
    return OptimizePromptResponse(optimized_prompt=optimized)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the bundled frontend app."""
    return RedirectResponse(url="/frontend/")


@app.get("/health")
async def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}
