#!/usr/bin/env python3
"""
Example 2: Evaluate Local GGUF Model with llama.cpp Log-Likelihood
==================================================================
Evaluates local GGUF quantized models (e.g. Gemma 3, Llama 3, Mistral, Qwen)
using llama-cpp-python with Log-Likelihood token logprob scoring.
Supports full GPU layer offloading (CUDA / Metal).
"""

import asyncio
import os
import sys

# Ensure root package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS
from mmlu_benchmark.report_generator import ReportGenerator


async def main():
    # 1. Configure Benchmark for GGUF model
    # Tip: You can test with Qwen2.5-1.5B GGUF:
    # wget -O Qwen2.5-1.5B-Instruct-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
    config = BenchmarkConfig(
        model_name="./Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",  # Local GGUF file path
        provider="llamacpp",                               # Or "gguf"
        preset="balanced",                                 # 12 subjects across STEM/Humanities/Social Sci/Other
        batch_size=1,                                      # Enforce batch_size=1 for native C++ KV-cache thread safety
        shots=5,                                           # 5-shot prompt context
        preload_dataset=True,                              # Preload dataset to prevent per-subject network stalls
        report_dir="./report",
    )

    # 2. Run benchmark pipeline
    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()

    # 3. Print results
    print("\n" + "=" * 60)
    print("MMLU llama.cpp Evaluation Completed!")
    print(f"Model: {metrics.model_name}")
    print(f"Micro Accuracy: {metrics.micro_accuracy * 100:.2f}%")
    print(f"Macro Accuracy: {metrics.macro_accuracy * 100:.2f}%")
    print("=" * 60)

    # 4. Generate visual HTML report and 300 DPI PNG charts
    rg = ReportGenerator("./report")
    rg.generate_html_report(metrics)
    rg.generate_png_charts(metrics)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (ImportError, FileNotFoundError) as e:
        print(f"\n[Environment Notice] {e}")
        sys.exit(0)
