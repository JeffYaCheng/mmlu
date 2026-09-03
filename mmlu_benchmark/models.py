"""
Model Inference Interface & Adapter Architecture (Pure Log-Likelihood Edition)
=============================================================================
Provides abstract BaseModelAdapter and 4 focused adapter implementations:
1. HuggingFaceModelAdapter ("huggingface"): Local HuggingFace / fine-tuned checkpoints with 4-bit NF4 support.
2. LlamaCppModelAdapter ("llamacpp" / "gguf"): Local GGUF quantized models running via llama-cpp-python.
3. CustomFunctionAdapter ("custom"): User-defined Python inference / scoring function.
4. MockModelAdapter ("mock"): Deterministic Mock for unit tests & zero-dependency CI verification.

All adapters implement academic-standard Next-Token Log-Likelihood probability evaluation:
    choice* = argmax_{c in {A, B, C, D}} log P(token = c | prompt)
"""

import abc
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class BaseModelAdapter(abc.ABC):
    """
    Abstract Model Adapter defining the standard Log-Likelihood MMLU evaluation contract.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, timeout_sec: float = 30.0):
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    @abc.abstractmethod
    async def evaluate_loglikelihood_async(self, prompt: str) -> Tuple[str, Dict[str, float], float]:
        """
        Evaluates next-token log-likelihood for choices A, B, C, D.
        Returns (predicted_choice, score_dict, latency_ms).
        """
        pass

    async def generate_async(self, prompt: str) -> Tuple[str, float]:
        """
        Standard interface returning (predicted_choice_string, latency_ms) for pipeline compatibility.
        """
        pred_choice, scores, latency_ms = await self.evaluate_loglikelihood_async(prompt)
        return pred_choice, latency_ms


class HuggingFaceModelAdapter(BaseModelAdapter):
    """
    Adapter 1: Local HuggingFace Transformer models or fine-tuned checkpoints.
    Computes exact Next-Token Log-Likelihood (log P(choice | prompt)) for choices A, B, C, D.
    Supports 4-bit NF4 quantization (BitsAndBytes) and CPU / Metal (MPS) / CUDA GPU.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.load_in_4bit = load_in_4bit
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "HuggingFace dependencies not found. Please install: "
                "pip install transformers torch accelerate bitsandbytes"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if device is None:
            if torch.cuda.is_available():
                self.device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_str = "mps"
            else:
                self.device_str = "cpu"
        else:
            self.device_str = device

        quant_config = None
        if self.load_in_4bit and self.device_str == "cuda":
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                quant_config = None

        dtype = torch.float16 if self.device_str in ["cuda", "mps"] else torch.float32
        if torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto" if self.device_str == "cuda" else None,
            trust_remote_code=True
        )
        if self.device_str != "cuda" and quant_config is None:
            self.model.to(self.device_str)

        self.model.eval()

    def evaluate_choice_loglikelihood(self, prompt: str) -> Tuple[str, Dict[str, float]]:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_logits = outputs.logits[0, -1, :]
            log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

        scores = {}
        for c in ["A", "B", "C", "D"]:
            tok_raw = self.tokenizer.encode(c, add_special_tokens=False)
            tok_space = self.tokenizer.encode(" " + c, add_special_tokens=False)
            id_raw = tok_raw[-1] if tok_raw else None
            id_space = tok_space[-1] if tok_space else None

            cand_vals = []
            if id_raw is not None:
                cand_vals.append(log_probs[id_raw].item())
            if id_space is not None:
                cand_vals.append(log_probs[id_space].item())
            scores[c] = max(cand_vals) if cand_vals else -float("inf")

        pred_choice = max(scores, key=scores.get)
        return pred_choice, scores

    async def evaluate_loglikelihood_async(self, prompt: str) -> Tuple[str, Dict[str, float], float]:
        import asyncio
        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()

        def _run():
            return self.evaluate_choice_loglikelihood(prompt)

        pred_choice, scores = await loop.run_in_executor(None, _run)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return pred_choice, scores, latency_ms


class LlamaCppModelAdapter(BaseModelAdapter):
    """
    Adapter 2: Local GGUF quantized models running via llama-cpp-python.
    Fast CPU/Metal/CUDA inference. Evaluates next-token log-likelihood over choices A, B, C, D.
    """

    def __init__(
        self,
        model_name: str = "./my-model.gguf",
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        logits_all: bool = True,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Please run: pip install llama-cpp-python")

        import threading
        # Thread lock to protect llama.cpp C++ context and KV-cache from concurrent access across threads
        self._lock = threading.Lock()

        # CRITICAL FIX: logits_all=True is required by llama-cpp-python to support logprobs in create_completion()
        # for academic next-token log-likelihood calculation. Without this, llama-cpp raises:
        # ValueError: logprobs is not supported for models created with logits_all=False
        self.llm = Llama(
            model_path=model_name,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=logits_all,
            verbose=False
        )

    def evaluate_choice_loglikelihood(self, prompt: str) -> Tuple[str, Dict[str, float]]:
        """
        Academic standard next-token log-likelihood evaluation over choices A, B, C, D via llama-cpp.
        Strictly computes log P(choice | prompt) using token logprobs.
        Thread-safe execution protected by self._lock. All scores strictly cast to native Python float.
        """
        try:
            with self._lock:
                res = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=50,
                )
            top_logprobs = res["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0]
            scores = {}
            for c in ["A", "B", "C", "D"]:
                cand_vals = []
                for tok, lp in top_logprobs.items():
                    if tok.strip() == c:
                        # Convert numpy.float32 to native Python float to avoid JSON serialization errors
                        cand_vals.append(float(lp))
                scores[c] = float(max(cand_vals)) if cand_vals else -float("inf")

            pred_choice = max(scores, key=scores.get)
            return pred_choice, scores
        except Exception as e:
            raise RuntimeError(
                f"llama-cpp-python Log-Likelihood evaluation failed: {e}. "
                "Ensure model is initialized with logits_all=True."
            ) from e

    async def evaluate_loglikelihood_async(self, prompt: str) -> Tuple[str, Dict[str, float], float]:
        import asyncio
        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()

        def _run():
            return self.evaluate_choice_loglikelihood(prompt)

        pred_choice, scores = await loop.run_in_executor(None, _run)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return pred_choice, scores, latency_ms


class CustomFunctionAdapter(BaseModelAdapter):
    """
    Adapter 3: Allows engineers to plug in ANY custom Python scoring function.
    The function MUST return a log-likelihood / logit score dictionary for choices A, B, C, D:
      {"A": -1.2, "B": -0.1, "C": -3.4, "D": -2.1}
    In accordance with pure academic Log-Likelihood evaluation, string text output
    and parse_choice are strictly prohibited.
    """

    def __init__(
        self,
        custom_fn: Optional[Callable[[str], Any]] = None,
        model_name: str = "my-custom-llm",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        if custom_fn is None:
            raise ValueError("`custom_fn` callable must be provided to CustomFunctionAdapter.")
        self.custom_fn = custom_fn

    async def evaluate_loglikelihood_async(self, prompt: str) -> Tuple[str, Dict[str, float], float]:
        import asyncio
        import inspect
        start_time = time.perf_counter()

        if inspect.iscoroutinefunction(self.custom_fn):
            output = await self.custom_fn(prompt)
        else:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(None, self.custom_fn, prompt)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not isinstance(output, dict):
            raise TypeError(
                f"CustomFunctionAdapter expects a dict of log-likelihood scores for choices ['A', 'B', 'C', 'D'], "
                f"e.g. {{'A': -1.5, 'B': -0.2, 'C': -3.1, 'D': -4.0}}, but received {type(output).__name__}: {output}. "
                f"Free-form text generation and parse_choice are strictly disabled; pure Log-Likelihood is required."
            )

        scores = {k.upper(): float(v) for k, v in output.items() if k.upper() in ["A", "B", "C", "D"]}
        for c in ["A", "B", "C", "D"]:
            if c not in scores:
                scores[c] = -float("inf")

        pred_choice = max(scores, key=scores.get)
        return pred_choice, scores, latency_ms


class MockModelAdapter(BaseModelAdapter):
    """
    Adapter 4: Deterministic Mock Adapter for unit tests and local pipeline sanity checks.
    Requires 0 GPU, 0 network, runs in milliseconds.
    """

    def __init__(self, model_name: str = "mock-baseline", fixed_choice: str = "B", **kwargs):
        super().__init__(model_name, **kwargs)
        self.fixed_choice = fixed_choice

    async def evaluate_loglikelihood_async(self, prompt: str) -> Tuple[str, Dict[str, float], float]:
        import asyncio
        await asyncio.sleep(0.005)
        scores = {"A": -2.5, "B": -0.1, "C": -1.8, "D": -3.0}
        if self.fixed_choice in scores:
            scores[self.fixed_choice] = 0.0
        return self.fixed_choice, scores, 5.0


def create_model_adapter(provider: str, model_name: str, **kwargs) -> BaseModelAdapter:
    """
    Factory helper to instantiate one of the 4 supported model adapters:
    - "huggingface": HuggingFaceModelAdapter
    - "llamacpp" / "gguf": LlamaCppModelAdapter
    - "custom": CustomFunctionAdapter
    - "mock": MockModelAdapter
    """
    provider_map = {
        "huggingface": HuggingFaceModelAdapter,
        "llamacpp": LlamaCppModelAdapter,
        "gguf": LlamaCppModelAdapter,
        "custom": CustomFunctionAdapter,
        "mock": MockModelAdapter,
    }
    prov_key = provider.lower().strip()
    if prov_key not in provider_map:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported 4 adapters: "
            "['huggingface', 'llamacpp', 'custom', 'mock']"
        )
    return provider_map[prov_key](model_name=model_name, **kwargs)
