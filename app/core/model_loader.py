"""
Model loader module — loads and caches a causal language model,
exposes token-level surprisal computation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _looks_like_device_error(exc: RuntimeError) -> bool:
    """Return True when *exc* looks like a CUDA / device-placement failure."""
    msg = str(exc).lower()
    needles = (
        "expected all tensors to be on the same device",
        "cuda",
        "cublas",
        "cudnn",
        "device-side assert",
        "index_select",
    )
    return any(n in msg for n in needles)


class ModelLoader:
    """Singleton-style wrapper around a Hugging Face causal LM.

    The model and tokenizer are loaded once and cached for the lifetime
    of the process.  All heavy lifting runs on the best available device
    (CUDA ➜ MPS ➜ CPU).
    """

    _instance: Optional["ModelLoader"] = None

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or self._resolve_device()
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer()
        self.model: PreTrainedModel = self._load_model()
        logger.info(
            "Model '%s' loaded on device '%s'",
            self.model_name,
            self.device,
        )

    # ------------------------------------------------------------------
    # Class-level caching
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
    ) -> "ModelLoader":
        """Return the cached instance, creating it on first call."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            try:
                # Smoke-test: create a tiny tensor on the GPU to verify
                # the driver / runtime actually works.
                t = torch.tensor([1.0], device="cuda")
                del t
                return "cuda"
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CUDA reported as available but failed a smoke-test "
                    "(%s). Falling back to CPU.",
                    exc,
                )
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _fallback_to_cpu(self) -> None:
        """Move model to CPU after a CUDA runtime error."""
        if self.device != "cpu":
            logger.warning(
                "CUDA runtime error detected — migrating model to CPU."
            )
            self.device = "cpu"
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_model(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        try:
            model.to(self.device)
        except RuntimeError as exc:
            logger.warning(
                "Failed to move model to '%s' (%s). Falling back to CPU.",
                self.device,
                exc,
            )
            self.device = "cpu"
            model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize *text* and move all returned tensors onto the active device."""
        encoding = self.tokenizer(text, return_tensors="pt")
        moved: Dict[str, torch.Tensor] = {}
        for key, value in encoding.items():
            try:
                moved[key] = value.to(self.device)
            except RuntimeError as exc:
                if _looks_like_device_error(exc):
                    self._fallback_to_cpu()
                    moved[key] = value.to(self.device)
                else:
                    raise
        return moved

    def tokenize(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Tokenize *text* and return (input_ids, token_strings)."""
        encoding = self.encode(text)
        input_ids: torch.Tensor = encoding["input_ids"]
        token_strings: List[str] = self.tokenizer.convert_ids_to_tokens(
            input_ids.squeeze().detach().cpu().tolist()
        )
        return input_ids, token_strings

    @torch.no_grad()
    def forward(self, **model_inputs: torch.Tensor) -> Any:
        """Run the model with automatic CPU fallback on device errors."""
        prepared = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in model_inputs.items()
        }
        try:
            return self.model(**prepared)
        except RuntimeError as exc:
            if not _looks_like_device_error(exc):
                raise
            self._fallback_to_cpu()
            prepared = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }
            return self.model(**prepared)

    @torch.no_grad()
    def compute_token_surprisal(self, text: str) -> List[float]:
        """Return per-token surprisal (negative log-probability) for *text*.

        Surprisal is computed as:
            s_t = -log P(x_t | x_{<t})
        using the model's causal logits.

        The first token has no left context, so its surprisal is set to 0.0.
        """
        encoded = self.encode(text)
        input_ids = encoded["input_ids"]
        logits: torch.Tensor = self.forward(**encoded).logits  # (1, seq_len, vocab)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (1, seq_len, vocab)

        # Shift: logits at position t predict token at position t+1.
        shifted_log_probs = log_probs[:, :-1, :]  # (1, seq_len-1, vocab)
        target_ids = input_ids[:, 1:]              # (1, seq_len-1)

        # Gather the log-prob assigned to the actual next token.
        token_log_probs = shifted_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)  # (1, seq_len-1)

        surprisal_values = (-token_log_probs.squeeze(0)).cpu().tolist()

        # Handle single-token edge case (tolist returns a scalar, not a list)
        if isinstance(surprisal_values, float):
            surprisal_values = [surprisal_values]

        # Prepend 0.0 for the first token (no prediction context).
        return [0.0] + surprisal_values

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back to a string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """Mean-pool the last hidden states for *text* and return a Python list."""
        encoded = self.encode(text)
        outputs = self.forward(**encoded, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden.mean(dim=1).squeeze(0).detach().cpu().tolist()
        return embedding

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
    ) -> str:
        """Generate a short completion for *prompt* with device-safe fallback."""
        encoded = self.encode(prompt)
        try:
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except RuntimeError as exc:
            if not _looks_like_device_error(exc):
                raise
            self._fallback_to_cpu()
            encoded = self.encode(prompt)
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output_ids[0].detach().cpu(), skip_special_tokens=True)
        if decoded.startswith(prompt):
            return decoded[len(prompt):].strip() or decoded.strip()
        return decoded.strip()
