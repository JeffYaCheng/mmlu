#!/usr/bin/env python3
"""
Pegatron MMLU Benchmark Suite - Google Colab / GPU & Local Runner
==================================================================
Standardized on Next-Token Log-Likelihood evaluation across 4 Model Adapters:
  1. "huggingface": HuggingFace Transformers / Fine-tuned checkpoints with 4-bit NF4 quantization.
  2. "llamacpp": Local GGUF models with llama-cpp-python GPU offload.
  3. "custom": User-defined Python inference / scoring function.
  4. "mock": Instant zero-dependency baseline for pipeline verification.

Auto-generates Radar Charts, Grouped Bar Charts, HTML Reports, and Markdown tables.
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError:
    plt = None
    np = None
    pd = None

from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import (
    ALL_57_SUBJECTS,
    BALANCED_BENCHMARK_SUBJECTS,
    MMLU_CATEGORIES,
    SUBJECT_TO_CATEGORY,
    MMLUDatasetLoader,
    normalize_answer_to_letter,
)
from mmlu_benchmark.models import (
    BaseModelAdapter,
    CustomFunctionAdapter,
    HuggingFaceModelAdapter,
    LlamaCppModelAdapter,
    MockModelAdapter,
    create_model_adapter,
)
from mmlu_benchmark.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MMLU-Runner")


def plot_comparison_figures(all_results: Dict[str, Any], output_dir: str):
    """
    Generates 4 high-resolution 300 DPI visualization figures:
      1. mmlu_comparison_radar.png: 4-Domain Radar Chart
      2. mmlu_overall_accuracy_bar.png: Overall Micro/Macro Accuracy Bar Chart
      3. mmlu_category_comparison_bar.png: 4-Domain Grouped Bar Chart
      4. mmlu_individual_model_breakdown.png: Per-model category performance subplots
    """
    os.makedirs(output_dir, exist_ok=True)
    if plt is None or np is None:
        logger.warning("matplotlib/numpy not available. Skipping static PNG export.")
        return

    categories = ["STEM", "Humanities", "Social Sciences", "Other"]
    model_keys = list(all_results.keys())
    model_labels = [m.split("/")[-1] for m in model_keys]
    palette = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b"]
    markers = ["o", "s", "^", "D", "v", "p", "*"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    # 1. Radar Chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    for idx, (m_key, res) in enumerate(all_results.items()):
        vals = [res["category_summary"].get(c, 0.0) * 100 for c in categories]
        vals += vals[:1]
        c = palette[idx % len(palette)]
        m = markers[idx % len(markers)]
        ls = linestyles[idx % len(linestyles)]
        lbl = model_labels[idx]
        ax.plot(angles, vals, color=c, linewidth=2.5, linestyle=ls, marker=m, markersize=8, label=lbl)
        ax.fill(angles, vals, color=c, alpha=0.12)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=10, color="#6b7280")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.title("MMLU 4-Domain Capability Radar Chart\n(Next-Token Log-Likelihood)", fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mmlu_comparison_radar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Overall Accuracy Bar
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(model_labels))
    width = 0.35
    micro_accs = [all_results[m]["micro_accuracy"] * 100 for m in model_keys]
    macro_accs = [all_results[m]["macro_accuracy"] * 100 for m in model_keys]

    b1 = ax.bar(x - width / 2, micro_accs, width, label="Micro Accuracy", color="#2563eb", edgecolor="#1e40af", alpha=0.9)
    b2 = ax.bar(x + width / 2, macro_accs, width, label="Macro Accuracy", color="#f97316", edgecolor="#c2410c", alpha=0.9)

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("MMLU Benchmark: Overall Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11, rotation=15)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    for bar in list(b1) + list(b2):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mmlu_overall_accuracy_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Category Comparison Bar
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(categories))
    total_models = len(model_keys)
    width = 0.8 / max(1, total_models)

    for i, m_key in enumerate(model_keys):
        c_vals = [all_results[m_key]["category_summary"].get(c, 0.0) * 100 for c in categories]
        pos = x - 0.4 + (i + 0.5) * width
        c = palette[i % len(palette)]
        bars = ax.bar(pos, c_vals, width, label=model_labels[i], color=c, alpha=0.88, edgecolor="#1f2937", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("MMLU Performance by Domain Category", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mmlu_category_comparison_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Individual Model Breakdown Subplots
    n_models = len(model_keys)
    cols = 2 if n_models > 1 else 1
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), dpi=300, squeeze=False)
    for idx, (m_key, res) in enumerate(all_results.items()):
        r = idx // cols
        c = idx % cols
        ax_sub = axes[r][c]
        c_vals = [res["category_summary"].get(cat, 0.0) * 100 for cat in categories]
        bars = ax_sub.bar(categories, c_vals, color=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"], edgecolor="#374151")
        ax_sub.set_title(f"{model_labels[idx]} (Micro: {res['micro_accuracy']*100:.1f}%)", fontsize=11, fontweight="bold")
        ax_sub.set_ylim(0, 100)
        ax_sub.set_ylabel("Accuracy (%)")
        ax_sub.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax_sub.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for idx in range(n_models, rows * cols):
        fig.delaxes(axes[idx // cols][idx % cols])

    plt.suptitle("Individual Model Domain Breakdown", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mmlu_individual_model_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Visual figures exported to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pegatron MMLU Benchmark Runner (Log-Likelihood Standard)")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["huggingface", "llamacpp", "gguf", "custom", "mock"],
        default="huggingface",
        help="Model adapter: 'huggingface', 'llamacpp', 'custom', or 'mock'",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
        help="Model IDs / GGUF paths to evaluate",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["balanced", "full", "quick"],
        default="balanced",
        help="Preset: 'balanced' (12 subjects across 4 domains), 'full' (all 57), 'quick' (2 subjects)",
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="Explicit subjects to evaluate")
    parser.add_argument("--shots", type=int, default=5, help="Few-shot count (0 to 5)")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["loglikelihood"],
        default="loglikelihood",
        help="Evaluation protocol: 'loglikelihood' (Next-Token Log-Likelihood)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples per subject")
    parser.add_argument("--batch-size", type=int, default=8, help="Concurrency / batch size")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization for HuggingFace")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory for JSON results")
    parser.add_argument("--report-dir", type=str, default="./report", help="Directory for HTML report and charts")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    if args.subjects:
        eval_subjects = args.subjects
    elif args.preset == "balanced":
        eval_subjects = BALANCED_BENCHMARK_SUBJECTS
    elif args.preset == "full":
        eval_subjects = ALL_57_SUBJECTS
    elif args.preset == "quick":
        eval_subjects = ["machine_learning", "philosophy"]
    else:
        eval_subjects = BALANCED_BENCHMARK_SUBJECTS

    all_reports = {}

    async def _run_all():
        for model_id in args.models:
            cfg = BenchmarkConfig(
                model_name=model_id,
                provider=args.provider,
                subjects=args.subjects,
                preset=args.preset,
                shots=args.shots,
                max_samples_per_subject=args.max_samples,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                report_dir=args.report_dir,
                load_in_4bit=not args.no_4bit,
                preload_dataset=True,
                eval_mode="loglikelihood",
            )
            pipeline = MMLUBenchmarkPipeline(cfg)
            metrics = await pipeline.run_benchmark()
            all_reports[model_id] = metrics.to_dict()

    asyncio.run(_run_all())

    # Save multi-model summary JSON
    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    # Plot PNG charts
    plot_comparison_figures(all_reports, args.report_dir)

    # Print Markdown Summary Table
    print("\n" + "=" * 70)
    print("📊 MMLU Benchmark Comparison Report (Academic Log-Likelihood Standard)")
    print("=" * 70)
    rows = []
    for m, r in all_reports.items():
        rows.append({
            "Model": m.split("/")[-1],
            "Provider": args.provider,
            "Shots": r["shots"],
            "Micro (%)": f"{r['micro_accuracy']*100:.2f}%",
            "Macro (%)": f"{r['macro_accuracy']*100:.2f}%",
            "STEM (%)": f"{r['category_summary']['STEM']*100:.2f}%",
            "Humanities (%)": f"{r['category_summary']['Humanities']*100:.2f}%",
            "Social Sci (%)": f"{r['category_summary']['Social Sciences']*100:.2f}%",
            "Other (%)": f"{r['category_summary']['Other']*100:.2f}%",
        })

    if pd is not None:
        df = pd.DataFrame(rows)
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df.to_string(index=False))
    else:
        headers = ["Model", "Provider", "Shots", "Micro (%)", "Macro (%)", "STEM (%)", "Humanities (%)", "Social Sci (%)", "Other (%)"]
        col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        divider_line = "-|-".join("-" * col_widths[h] for h in headers)
        print(f"| {header_line} |")
        print(f"| {divider_line} |")
        for r in rows:
            row_line = " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
            print(f"| {row_line} |")
    print("=" * 70)
    logger.info(f"Benchmark finished. All files saved to {args.report_dir} and {args.output_dir}")


if __name__ == "__main__":
    main()
