#!/usr/bin/env python3
"""
Example 4: Evaluate Mock Model Adapter (0-Dependency Local Verification)
========================================================================
Runs deterministic mock evaluation in milliseconds without requiring GPU or network.
Ideal for continuous integration (CI/CD) and verifying report formatting.
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
    config = BenchmarkConfig(
        model_name="mock-baseline",
        provider="mock",
        subjects=BALANCED_BENCHMARK_SUBJECTS,
        shots=5,
        max_samples_per_subject=5,  # Quick test: 5 samples per subject
        report_dir="./report",
    )

    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()

    print("\n" + "=" * 60)
    print("Mock Benchmark Completed Successfully!")
    print(f"Total Samples: {metrics.total_samples}")
    print(f"Accuracy: {metrics.micro_accuracy * 100:.2f}%")
    print("=" * 60)

    rg = ReportGenerator("./report")
    rg.generate_html_report(metrics)


if __name__ == "__main__":
    asyncio.run(main())
