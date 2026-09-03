#!/usr/bin/env python3
"""
Example 3: Evaluate User Custom Python Inference / Logit Function
=================================================================
Allows engineers to evaluate ANY in-house model architecture, PyTorch module,
or custom scoring pipeline without modifying the benchmark library.
The custom function must return:
  A log-likelihood / score dictionary: {"A": -1.2, "B": -0.1, "C": -3.4, "D": -2.1}
"""

import asyncio
import os
import sys
from typing import Dict

# Ensure root package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS
from mmlu_benchmark.report_generator import ReportGenerator


# Define your custom model / inference logic here:
def my_custom_model_scoring_fn(prompt: str) -> Dict[str, float]:
    """
    Example Custom Model scoring function adhering to academic Next-Token Log-Likelihood.
    Engineers can run PyTorch forward pass, ONNX runtime, TensorRT-LLM, etc.,
    and return next-token log-likelihoods / logits for ['A', 'B', 'C', 'D'].
    """
    # Pure Log-Likelihood evaluation (ArgMax P(Choice | Prompt)):
    return {"A": -2.4, "B": -0.2, "C": -1.8, "D": -3.5}


async def main():
    config = BenchmarkConfig(
        model_name="my-custom-research-model",
        provider="custom",
        subjects=BALANCED_BENCHMARK_SUBJECTS,
        shots=5,
        max_samples_per_subject=5,
        report_dir="./report",
    )

    # Pass your custom function into the pipeline:
    pipeline = MMLUBenchmarkPipeline(config=config, custom_fn=my_custom_model_scoring_fn)
    metrics = await pipeline.run_benchmark()

    print("\n" + "=" * 60)
    print(f"Custom Model Benchmark Completed: {metrics.micro_accuracy * 100:.2f}%")
    print("=" * 60)

    rg = ReportGenerator("./report")
    rg.generate_html_report(metrics)


if __name__ == "__main__":
    asyncio.run(main())
