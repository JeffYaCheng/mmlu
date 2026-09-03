#!/usr/bin/env python3
"""
Pegatron ML Benchmark Pipeline - MMLU Evaluation Runner
======================================================
Objective:
    Automated benchmark pipeline to evaluate Foundation LLMs on the Massive
    Multitask Language Understanding (MMLU) benchmark across 57 subjects.
    Supports 0-shot and 5-shot evaluation, exact match parsing, confidence
    intervals, and markdown/JSON report exports.

Dataset: cais/mmlu (Hugging Face Datasets)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mmlu_benchmark.dataset import (
    ALL_57_SUBJECTS,
    BALANCED_BENCHMARK_SUBJECTS,
    MMLUDatasetLoader,
    SUBJECT_TO_CATEGORY,
)
from mmlu_benchmark.metrics import BenchmarkMetrics, SubjectResult
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
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MMLU-Benchmark")


@dataclass
class BenchmarkConfig:
    model_name: str
    provider: str
    subjects: Optional[List[str]] = None
    preset: Optional[str] = "balanced"
    shots: int = 5
    batch_size: int = 8
    max_samples_per_subject: Optional[int] = None
    output_dir: str = "./benchmark_results"
    report_dir: str = "./report"
    temperature: float = 0.0
    timeout_sec: float = 30.0
    eval_mode: str = "loglikelihood"
    load_in_4bit: bool = True
    preload_dataset: bool = True

    def __post_init__(self):
        """
        Resolves subjects based on explicit subjects list or preset name:
          - 'balanced' (default): 12 subjects across STEM, Humanities, Social Sciences, Other.
          - 'full': all official 57 MMLU subjects.
          - 'quick': 2 subjects ('machine_learning', 'philosophy') for fast smoke testing.
          - 'stem': subjects under STEM category.
          - 'humanities': subjects under Humanities category.
          - 'social_sciences': subjects under Social Sciences category.
          - 'other': subjects under Other category.
        """
        if not self.subjects:
            if self.preset == "full":
                self.subjects = list(ALL_57_SUBJECTS)
            elif self.preset == "quick":
                self.subjects = ["machine_learning", "philosophy"]
            elif self.preset == "stem":
                self.subjects = [s for s, c in SUBJECT_TO_CATEGORY.items() if c == "STEM"]
            elif self.preset == "humanities":
                self.subjects = [s for s, c in SUBJECT_TO_CATEGORY.items() if c == "Humanities"]
            elif self.preset == "social_sciences":
                self.subjects = [s for s, c in SUBJECT_TO_CATEGORY.items() if c == "Social Sciences"]
            elif self.preset == "other":
                self.subjects = [s for s, c in SUBJECT_TO_CATEGORY.items() if c == "Other"]
            elif self.preset == "balanced" or self.preset is None:
                self.subjects = list(BALANCED_BENCHMARK_SUBJECTS)
            else:
                self.subjects = list(BALANCED_BENCHMARK_SUBJECTS)

        # Ensure llamacpp / gguf provider enforces batch_size=1 for native C++ KV-cache safety
        if self.provider in ("llamacpp", "gguf") and self.batch_size > 1:
            logger.info(f"LlamaCpp provider detected: adjusting batch_size from {self.batch_size} to 1 for native C++ KV-cache safety.")
            self.batch_size = 1


class MMLUBenchmarkPipeline:
    """
    Main pipeline orchestrator for MMLU benchmark evaluation.
    All evaluations use academic standard Log-Likelihood next-token scoring.
    """

    def __init__(self, config: BenchmarkConfig, custom_fn: Optional[Callable[[str], Any]] = None):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.report_generator = ReportGenerator(output_dir=config.report_dir)

        logger.info("Initializing Model Adapter: %s (%s)", config.model_name, config.provider)
        adapter_kwargs: Dict[str, Any] = {
            "temperature": config.temperature,
            "timeout_sec": config.timeout_sec,
        }
        if custom_fn is not None:
            adapter_kwargs["custom_fn"] = custom_fn
        if config.provider == "huggingface":
            adapter_kwargs["load_in_4bit"] = config.load_in_4bit

        self.model: BaseModelAdapter = create_model_adapter(
            provider=config.provider,
            model_name=config.model_name,
            **adapter_kwargs,
        )

        logger.info("Initializing MMLU Dataset Loader...")
        self.dataset_loader = MMLUDatasetLoader(auto_preload=config.preload_dataset)

    def format_mmlu_prompt(self, question: Dict[str, Any], few_shots: List[Dict[str, Any]]) -> str:
        """
        Formats MMLU multiple choice questions into structured prompt.
        """
        prompt = f"The following are multiple choice questions (with answers) about {question['subject'].replace('_', ' ')}.\n\n"

        # Add Few-shot examples if specified
        for shot in few_shots:
            prompt += f"Question: {shot['question']}\n"
            for label, opt in zip(["A", "B", "C", "D"], shot["choices"]):
                prompt += f"{label}. {opt}\n"
            prompt += f"Answer: {shot['answer']}\n\n"

        # Target Question
        prompt += f"Question: {question['question']}\n"
        for label, opt in zip(["A", "B", "C", "D"], question["choices"]):
            prompt += f"{label}. {opt}\n"
        prompt += "\nAnswer:"
        return prompt

    async def evaluate_single_question(
        self, question: Dict[str, Any], few_shots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates a single question asynchronously using Next-Token Log-Likelihood.
        Enforces pure log-likelihood without string parsing.
        """
        prompt = self.format_mmlu_prompt(question, few_shots)

        predicted_choice, scores, latency_ms = await self.model.evaluate_loglikelihood_async(prompt)

        is_correct = (predicted_choice == question["answer"])

        return {
            "id": question["id"],
            "subject": question["subject"],
            "category": SUBJECT_TO_CATEGORY.get(question["subject"], "Other"),
            "question": question["question"],
            "choices": question.get("choices", []),
            "ground_truth": question["answer"],
            "predicted": predicted_choice,
            "is_correct": is_correct,
            "latency_ms": round(latency_ms, 2),
            "log_likelihood_scores": scores,
            "raw_output": f"Scores: {scores} -> Choice: {predicted_choice}",
        }

    async def evaluate_subject(self, subject: str) -> SubjectResult:
        """
        Evaluates all questions for a specific MMLU subject using bounded concurrency.
        Accurately enforces few-shot context matching and logs explicit warnings if data is truncated.
        """
        category = SUBJECT_TO_CATEGORY.get(subject, "Other")
        test_samples, dev_samples = self.dataset_loader.load_subject(
            subject=subject,
            max_samples=self.config.max_samples_per_subject,
        )

        effective_shots = min(len(dev_samples), self.config.shots) if self.config.shots > 0 else 0
        few_shots = dev_samples[:effective_shots]

        if self.config.shots > 0 and len(dev_samples) < self.config.shots:
            logger.warning(
                f"[{subject}] Few-shot discrepancy: Requested {self.config.shots}-shot prompt, "
                f"but only {len(dev_samples)} dev example(s) were available. "
                f"Evaluating with {effective_shots}-shot context."
            )
            logger.info(
                f"==> Evaluating Subject: {subject} [{category}] "
                f"({effective_shots}-shot actual / {self.config.shots}-shot requested)"
            )
        else:
            logger.info(f"==> Evaluating Subject: {subject} [{category}] ({effective_shots}-shot)")

        semaphore = asyncio.Semaphore(self.config.batch_size)
        total_q = len(test_samples)
        completed_q = 0

        async def _bounded_eval(idx, q):
            nonlocal completed_q
            async with semaphore:
                res = await self.evaluate_single_question(q, few_shots)
                completed_q += 1
                if completed_q % max(1, total_q // 5) == 0 or completed_q == total_q:
                    logger.info(f"  [{subject}] Progress: {completed_q}/{total_q} questions evaluated...")
                return res

        tasks = [_bounded_eval(i, q) for i, q in enumerate(test_samples)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        correct_count = sum(1 for r in results if r["is_correct"])
        total_count = len(results)
        accuracy = (correct_count / total_count) if total_count > 0 else 0.0
        avg_latency = sum(r["latency_ms"] for r in results) / total_count if total_count > 0 else 0.0

        return SubjectResult(
            subject=subject,
            category=category,
            shots_used=effective_shots,
            total_samples=total_count,
            correct_samples=correct_count,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            sample_details=results,
        )

    async def run_benchmark(self) -> BenchmarkMetrics:
        """
        Runs the full benchmark suite across all requested subjects.
        """
        start_all = time.time()
        logger.info(f"Starting MMLU Benchmark on {len(self.config.subjects)} subjects...")

        subject_results = []
        for subject in self.config.subjects:
            res = await self.evaluate_subject(subject)
            subject_results.append(res)
            logger.info(f"Subject [{subject}] ({res.category}) Accuracy: {res.accuracy * 100:.2f}% ({res.correct_samples}/{res.total_samples})")

        total_samples = sum(s.total_samples for s in subject_results)
        total_correct = sum(s.correct_samples for s in subject_results)
        macro_accuracy = sum(s.accuracy for s in subject_results) / len(subject_results) if subject_results else 0.0
        micro_accuracy = (total_correct / total_samples) if total_samples > 0 else 0.0
        total_time = time.time() - start_all

        metrics = BenchmarkMetrics(
            model_name=self.config.model_name,
            provider=self.config.provider,
            shots=self.config.shots,
            total_samples=total_samples,
            total_correct=total_correct,
            macro_accuracy=macro_accuracy,
            micro_accuracy=micro_accuracy,
            total_runtime_sec=total_time,
            subjects=subject_results,
        )

        self._export_results(metrics)
        return metrics

    def _export_results(self, metrics: BenchmarkMetrics):
        """
        Exports benchmark results to structured JSON, Markdown summary, interactive HTML, and static PNG charts in ./report/.
        """
        timestamp = int(time.time())
        safe_model_name = self.config.model_name.replace("/", "_").replace(":", "_")
        json_file = self.output_path / f"mmlu_{safe_model_name}_{self.config.shots}shot_{timestamp}.json"
        md_file = self.output_path / f"mmlu_{safe_model_name}_{self.config.shots}shot_{timestamp}.md"

        def _json_serializable(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "__float__"):
                return float(obj)
            if hasattr(obj, "__int__"):
                return int(obj)
            return str(obj)

        # Save JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False, default=_json_serializable)
        logger.info(f"JSON Benchmark Report saved to: {json_file}")

        # Save Markdown Summary
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(metrics.to_markdown_summary())
        logger.info(f"Markdown Summary saved to: {md_file}")

        # Save Interactive HTML Report with charts to ./report/
        html_report_path = self.report_generator.generate_html_report(
            metrics,
            filename=f"mmlu_report_{safe_model_name}_{self.config.shots}shot_{timestamp}.html",
        )
        logger.info(f"Visual HTML Report generated at: {html_report_path}")

        # Also save static PNG charts (Radar, Overall Bar, Category Bar, Breakdown)
        png_charts = self.report_generator.generate_png_charts(metrics)
        if png_charts:
            logger.info(f"Generated {len(png_charts)} visualization charts in {self.config.report_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pegatron MMLU Benchmark Pipeline (Log-Likelihood Standard)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name or huggingface checkpoint or GGUF path")
    parser.add_argument(
        "--provider",
        type=str,
        choices=["huggingface", "llamacpp", "gguf", "custom", "mock"],
        default="huggingface",
        help="Inference provider: 'huggingface', 'llamacpp' (or 'gguf'), 'custom', 'mock'",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["balanced", "full", "quick"],
        default="balanced",
        help="Subject preset: 'balanced' (12 subjects across STEM/Humanities/Social Sci/Other), 'full' (all 57), 'quick' (2 subjects)",
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="Explicit MMLU subjects to evaluate (overrides --preset)")
    parser.add_argument("--shots", type=int, default=5, help="Few-shot count (0 to 5)")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["loglikelihood"],
        default="loglikelihood",
        help="Evaluation protocol: 'loglikelihood' (Next-Token Log-Likelihood)",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization for HuggingFace models")
    parser.add_argument("--no-preload", action="store_true", help="Disable single-batch dataset preloading from HuggingFace")
    parser.add_argument("--batch-size", type=int, default=8, help="Async concurrency limit")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per subject (useful for testing)")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Directory to save raw logs & json")
    parser.add_argument("--report-dir", type=str, default="./report", help="Directory to save visual charts and HTML reports")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve subjects based on preset or explicit flag
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

    cfg = BenchmarkConfig(
        model_name=args.model,
        provider=args.provider,
        subjects=args.subjects,
        preset=args.preset,
        shots=args.shots,
        batch_size=args.batch_size,
        max_samples_per_subject=args.max_samples,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        eval_mode="loglikelihood",
        load_in_4bit=not args.no_4bit,
        preload_dataset=not args.no_preload,
    )
    pipeline = MMLUBenchmarkPipeline(cfg)
    return asyncio.run(pipeline.run_benchmark())


if __name__ == "__main__":
    main()
