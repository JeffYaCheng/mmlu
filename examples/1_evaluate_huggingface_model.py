#!/usr/bin/env python3
"""
Example 1: Evaluate Local HuggingFace Checkpoint or Hub Model with Log-Likelihood
=================================================================================
Evaluates local fine-tuned checkpoints (LoRA / SFT) or HuggingFace Hub models
using exact Next-Token Log-Likelihood:
    P(Choice | Prompt) for Choices in ['A', 'B', 'C', 'D']
Supports 4-bit BitsAndBytes quantization on GPU, Apple Silicon MPS, or CPU.
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
    # 1. Configure Benchmark
    config = BenchmarkConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",  # Or local path e.g. "./checkpoints/my_lora_model"
        provider="huggingface",
        preset="balanced",                                # Or preset="full", preset="quick", or explicit subjects=[...]
        shots=5,                                          # 5-shot prompt context
        load_in_4bit=True,                                # 4-bit NF4 quantization for GPU memory efficiency
        preload_dataset=True,                             # Download all subjects in one batch (prevents rate limits)
        report_dir="./report",
    )

    # 2. Initialize and execute evaluation pipeline
    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()

    # 3. Print summary results
    print("\n" + "=" * 60)
    print("MMLU Evaluation Completed!")
    print(f"Model: {metrics.model_name}")
    print(f"Micro Accuracy: {metrics.micro_accuracy * 100:.2f}%")
    print(f"Macro Accuracy: {metrics.macro_accuracy * 100:.2f}%")
    print("Category Breakdown:")
    for cat, acc in metrics.category_summary.items():
        print(f"  - {cat:16s}: {acc * 100:.2f}%")
    print("=" * 60)

    # 4. Generate visual HTML report and 300 DPI PNG charts
    rg = ReportGenerator("./report")
    rg.generate_html_report(metrics)
    rg.generate_png_charts(metrics)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"\n[Environment Notice] {e}")
        sys.exit(0)
