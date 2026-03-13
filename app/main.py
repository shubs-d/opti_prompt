"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.model_loader import ModelLoader
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

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

# ------------------------------------------------------------------
# CORS — origins allowed for local frontend + browser extension
# ------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://optiprompt-gqd9hqf6dffvaacb.eastasia-01.azurewebsites.net",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "https://chatgpt.com",
    "https://claude.ai",
]

# Chrome extensions send Origin: chrome-extension://<id>
ALLOWED_ORIGIN_REGEX = r"^chrome-extension://.*$"


# ------------------------------------------------------------------
# Lifespan — load model + init storage once on startup
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Pre-load the language model and initialise storage."""
    logger.info("Loading model on startup …")
    ModelLoader.get_instance()
    logger.info("Model ready.")

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
