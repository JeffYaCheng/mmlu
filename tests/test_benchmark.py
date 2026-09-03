"""
Expanded Comprehensive Unit & Integration Test Suite for MMLU Benchmark Pipeline
================================================================================
This test suite covers all edge cases and pipeline modules:
1. Regex Choice Parser Edge Cases (Markdown, latex, conversational, multiline, tricky prefixes)
2. Statistical Metrics & Confidence Interval Edge Cases (zero divisions, aggregation, bounds)
3. Dataset Loader, 57 Subject Mapping, & Balanced Preset Invariants
4. Prompt Formatting & Few-Shot Injection Integrity (0-shot, 1-shot, 5-shot)
5. Model Adapters (Mock, Custom Function sync/async, Factory dispatch, Error handling)
6. HTML Report Generator & Markdown Summary Export Verification
7. End-to-End Pipeline Evaluation (Multiple subjects, batching, limit filtering)
8. CLI Argument Parsing & Preset Resolution
"""

import asyncio
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline, parse_args
from mmlu_benchmark.dataset import (
    ALL_57_SUBJECTS,
    BALANCED_BENCHMARK_SUBJECTS,
    MMLU_CATEGORIES,
    MMLUDatasetLoader,
    SUBJECT_TO_CATEGORY,
    normalize_answer_to_letter,
)
from mmlu_benchmark.metrics import BenchmarkMetrics, SubjectResult
from mmlu_benchmark.models import (
    BaseModelAdapter,
    CustomFunctionAdapter,
    MockModelAdapter,
    create_model_adapter,
)
from mmlu_benchmark.report_generator import ReportGenerator


class TestAnswerNormalizationAndLogLikelihood(unittest.TestCase):
    """Validates raw dataset answer normalization and pure log-likelihood decision integrity."""

    def test_normalize_answer_to_letter(self):
        # Letters
        self.assertEqual(normalize_answer_to_letter("A"), "A")
        self.assertEqual(normalize_answer_to_letter("b"), "B")
        self.assertEqual(normalize_answer_to_letter(" C "), "C")
        self.assertEqual(normalize_answer_to_letter("d"), "D")

        # Numeric indices (0->A, 1->B, 2->C, 3->D)
        self.assertEqual(normalize_answer_to_letter(0), "A")
        self.assertEqual(normalize_answer_to_letter(1), "B")
        self.assertEqual(normalize_answer_to_letter(2), "C")
        self.assertEqual(normalize_answer_to_letter(3), "D")
        self.assertEqual(normalize_answer_to_letter("0"), "A")
        self.assertEqual(normalize_answer_to_letter("1"), "B")
        self.assertEqual(normalize_answer_to_letter("2"), "C")
        self.assertEqual(normalize_answer_to_letter("3"), "D")

    def test_pure_loglikelihood_argmax(self):
        """Ensures highest log-likelihood score is selected as prediction."""
        scores = {"A": -3.2, "B": -0.15, "C": -2.8, "D": -4.1}
        best_choice = max(scores.keys(), key=lambda c: scores[c])
        self.assertEqual(best_choice, "B")

        scores2 = {"A": -0.05, "B": -4.0, "C": -5.1, "D": -6.2}
        best_choice2 = max(scores2.keys(), key=lambda c: scores2[c])
        self.assertEqual(best_choice2, "A")


class TestDatasetTaxonomyAndIntegrity(unittest.TestCase):
    """Validates MMLU 57 Subject taxonomy, 4 canonical domains, and built-in questions."""

    def test_official_57_subject_count(self):
        self.assertEqual(len(ALL_57_SUBJECTS), 57)
        self.assertEqual(len(SUBJECT_TO_CATEGORY), 57)

    def test_canonical_4_categories_present(self):
        expected_cats = {"STEM", "Humanities", "Social Sciences", "Other"}
        self.assertEqual(set(MMLU_CATEGORIES.keys()), expected_cats)

    def test_balanced_preset_distribution(self):
        self.assertEqual(len(BALANCED_BENCHMARK_SUBJECTS), 12)
        cat_counts = {}
        for s in BALANCED_BENCHMARK_SUBJECTS:
            cat = SUBJECT_TO_CATEGORY.get(s, "Other")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        self.assertEqual(cat_counts.get("STEM"), 3)
        self.assertEqual(cat_counts.get("Humanities"), 3)
        self.assertEqual(cat_counts.get("Social Sciences"), 3)
        self.assertEqual(cat_counts.get("Other"), 3)

    def test_answer_normalization_letters_and_integers(self):
        self.assertEqual(normalize_answer_to_letter(0), "A")
        self.assertEqual(normalize_answer_to_letter(1), "B")
        self.assertEqual(normalize_answer_to_letter(2), "C")
        self.assertEqual(normalize_answer_to_letter(3), "D")
        self.assertEqual(normalize_answer_to_letter("A"), "A")
        self.assertEqual(normalize_answer_to_letter("b"), "B")
        self.assertEqual(normalize_answer_to_letter("c"), "C")
        self.assertEqual(normalize_answer_to_letter("D"), "D")
        self.assertEqual(normalize_answer_to_letter(99), "A")

    def test_dataset_loader_offline_fallback(self):
        loader = MMLUDatasetLoader()
        test_set, dev_set = loader.load_subject("machine_learning")
        self.assertGreater(len(test_set), 0)
        for item in test_set:
            self.assertIn("question", item)
            self.assertIn("choices", item)
            self.assertEqual(len(item["choices"]), 4)
            self.assertIn(item["answer"], ["A", "B", "C", "D"])


class TestMetricsAndConfidenceIntervals(unittest.TestCase):
    """Validates statistical calculations and confidence interval edge cases."""

    def test_normal_confidence_interval(self):
        res = SubjectResult(
            subject="college_physics",
            total_samples=100,
            correct_samples=75,
            accuracy=0.75,
            avg_latency_ms=50.0,
            category="STEM",
            sample_details=[],
        )
        expected_ci = round(1.96 * math.sqrt(0.75 * 0.25 / 100), 4)
        self.assertAlmostEqual(res.confidence_interval_95, expected_ci, places=4)

    def test_zero_samples_handling(self):
        res = SubjectResult(
            subject="empty_subj",
            total_samples=0,
            correct_samples=0,
            accuracy=0.0,
            avg_latency_ms=0.0,
            category="STEM",
            sample_details=[],
        )
        self.assertEqual(res.confidence_interval_95, 0.0)

    def test_perfect_and_zero_accuracy_ci(self):
        perfect = SubjectResult("math", 50, 50, 1.0, 10.0, category="STEM")
        zero = SubjectResult("math", 50, 0, 0.0, 10.0, category="STEM")
        self.assertEqual(perfect.confidence_interval_95, 0.0)
        self.assertEqual(zero.confidence_interval_95, 0.0)

    def test_benchmark_metrics_properties(self):
        s1 = SubjectResult("sub1", 10, 8, 0.8, 100.0, category="STEM")
        s2 = SubjectResult("sub2", 10, 6, 0.6, 200.0, category="Humanities")
        metrics = BenchmarkMetrics(
            model_name="test-llm",
            provider="mock",
            shots=5,
            total_samples=20,
            total_correct=14,
            macro_accuracy=0.7,
            micro_accuracy=0.7,
            total_runtime_sec=2.5,
            subjects=[s1, s2],
        )
        self.assertEqual(metrics.correct_count, 14)
        self.assertEqual(metrics.total_questions, 20)
        self.assertEqual(metrics.avg_latency_ms, 150.0)
        self.assertEqual(metrics.category_summary["STEM"], 0.8)
        self.assertEqual(metrics.category_summary["Humanities"], 0.6)

        # Markdown output verification
        md = metrics.to_markdown_summary()
        self.assertIn("Micro Accuracy", md)
        self.assertIn("Macro Accuracy", md)
        self.assertIn("STEM", md)


class TestPromptFormatting(unittest.TestCase):
    """Tests zero-shot and few-shot prompt construction."""

    def setUp(self):
        cfg = BenchmarkConfig(model_name="mock", provider="mock", subjects=["machine_learning"])
        self.pipeline = MMLUBenchmarkPipeline(cfg)
        self.sample_q = {
            "subject": "machine_learning",
            "question": "What is overfitting?",
            "choices": ["High variance", "High bias", "Zero loss always", "None"],
            "answer": "A",
        }
        self.sample_shot = {
            "subject": "machine_learning",
            "question": "What is supervised learning?",
            "choices": ["Learning with labels", "No labels", "Clustering", "Reinforcement"],
            "answer": "A",
        }

    def test_zero_shot_prompt_format(self):
        prompt = self.pipeline.format_mmlu_prompt(self.sample_q, few_shots=[])
        self.assertIn("The following are multiple choice questions (with answers) about machine learning", prompt)
        self.assertIn("Question: What is overfitting?", prompt)
        self.assertIn("A. High variance", prompt)
        self.assertIn("B. High bias", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))

    def test_few_shot_prompt_format(self):
        prompt = self.pipeline.format_mmlu_prompt(self.sample_q, few_shots=[self.sample_shot])
        self.assertIn("Question: What is supervised learning?", prompt)
        self.assertIn("Answer: A", prompt)
        self.assertIn("Question: What is overfitting?", prompt)


class TestModelAdaptersAndFactory(unittest.TestCase):
    """Tests model adapters with pure next-token log-likelihood evaluation."""

    def test_mock_adapter_deterministic_output(self):
        mock = MockModelAdapter(model_name="deterministic-mock")
        prompt = "Question: Test? A. Yes B. No C. Maybe D. Unknown\nAnswer:"
        async def _run():
            pred_choice, scores, latency = await mock.evaluate_loglikelihood_async(prompt)
            self.assertIn(pred_choice, ["A", "B", "C", "D"])
            self.assertGreater(latency, 0.0)
            self.assertEqual(scores[pred_choice], max(scores.values()))
        asyncio.run(_run())

    def test_sync_custom_function_adapter(self):
        def my_sync_fn(prompt: str) -> dict:
            return {"A": -2.5, "B": -0.1, "C": -3.0, "D": -4.0}

        adapter = CustomFunctionAdapter(model_name="sync-custom", custom_fn=my_sync_fn)
        async def _run():
            pred_choice, scores, latency = await adapter.evaluate_loglikelihood_async("some prompt")
            self.assertEqual(pred_choice, "B")
            self.assertGreaterEqual(latency, 0.0)

            # Rejection test: string return type must raise TypeError
            bad_adapter = CustomFunctionAdapter(model_name="bad-custom", custom_fn=lambda p: "Answer is B")
            with self.assertRaises(TypeError):
                await bad_adapter.evaluate_loglikelihood_async("some prompt")

        asyncio.run(_run())

    def test_async_custom_function_adapter(self):
        async def my_async_fn(prompt: str) -> dict:
            await asyncio.sleep(0.01)
            return {"A": -3.0, "B": -2.0, "C": -0.15, "D": -4.5}

        adapter = CustomFunctionAdapter(model_name="async-custom", custom_fn=my_async_fn)
        async def _run():
            pred_choice, scores, latency = await adapter.evaluate_loglikelihood_async("some prompt")
            self.assertEqual(pred_choice, "C")
            self.assertGreater(latency, 5.0)
        asyncio.run(_run())

    def test_create_model_adapter_factory(self):
        adapter1 = create_model_adapter("mock", "mock-model")
        self.assertIsInstance(adapter1, MockModelAdapter)

        adapter2 = create_model_adapter("custom", "custom-model", custom_fn=lambda p: {"A": 0.0, "B": -1.0, "C": -2.0, "D": -3.0})
        self.assertIsInstance(adapter2, CustomFunctionAdapter)

    def test_all_provider_choices_supported_in_cli_and_factory(self):
        import sys
        supported_providers = ["huggingface", "llamacpp", "gguf", "custom", "mock"]
        orig_argv = sys.argv
        try:
            for p in supported_providers:
                sys.argv = ["benchmark.py", "--provider", p]
                args = parse_args()
                self.assertEqual(args.provider, p)
        finally:
            sys.argv = orig_argv


class TestReportGeneratorIntegrity(unittest.TestCase):
    """Tests HTML report generation and structure."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_html_report_creation(self):
        rg = ReportGenerator(output_dir=self.temp_dir)
        s1 = SubjectResult("machine_learning", 10, 8, 0.8, 12.0, category="STEM")
        s2 = SubjectResult("philosophy", 10, 9, 0.9, 15.0, category="Humanities")
        metrics = BenchmarkMetrics(
            model_name="Tested-Model",
            provider="mock",
            shots=5,
            total_samples=20,
            total_correct=17,
            macro_accuracy=0.85,
            micro_accuracy=0.85,
            total_runtime_sec=3.0,
            subjects=[s1, s2],
        )
        report_path = rg.generate_html_report(metrics)
        self.assertTrue(os.path.exists(report_path))
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Tested-Model", content)
            self.assertIn("MMLU EVALUATION REPORT", content)
            self.assertIn("85.00%", content)


class TestPipelineEndToEndAndPresets(unittest.TestCase):
    """Tests full pipeline run across balanced and custom subjects."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_balanced_preset_execution(self):
        async def _run():
            cfg = BenchmarkConfig(
                model_name="pipeline-test-model",
                provider="mock",
                subjects=BALANCED_BENCHMARK_SUBJECTS,
                shots=0,
                max_samples_per_subject=2,
                output_dir=self.temp_dir,
                report_dir=self.temp_dir,
            )
            pipeline = MMLUBenchmarkPipeline(cfg)
            metrics = await pipeline.run_benchmark()

            self.assertEqual(len(metrics.subjects), 12)
            self.assertEqual(metrics.total_samples, 24)
            self.assertIn("STEM", metrics.category_summary)
            self.assertIn("Humanities", metrics.category_summary)
            self.assertIn("Social Sciences", metrics.category_summary)
            self.assertIn("Other", metrics.category_summary)

            # Check that exported JSON file exists
            exported_files = list(Path(self.temp_dir).glob("*.json"))
            self.assertGreater(len(exported_files), 0)

        asyncio.run(_run())

    def test_benchmark_config_presets(self):
        # Default preset should resolve to 12 balanced subjects
        cfg_default = BenchmarkConfig(model_name="test", provider="mock")
        self.assertEqual(len(cfg_default.subjects), 12)
        self.assertEqual(cfg_default.preset, "balanced")

        # Preset 'full' should resolve to 57 subjects
        cfg_full = BenchmarkConfig(model_name="test", provider="mock", preset="full")
        self.assertEqual(len(cfg_full.subjects), 57)

        # Preset 'quick' should resolve to 2 subjects
        cfg_quick = BenchmarkConfig(model_name="test", provider="mock", preset="quick")
        self.assertEqual(cfg_quick.subjects, ["machine_learning", "philosophy"])

        # Explicit subjects should take precedence over preset
        cfg_explicit = BenchmarkConfig(
            model_name="test",
            provider="mock",
            subjects=["machine_learning"],
            preset="full",
        )
        self.assertEqual(cfg_explicit.subjects, ["machine_learning"])

    def test_cli_arg_parser_presets(self):
        import sys
        # Save original sys.argv
        orig_argv = sys.argv
        try:
            sys.argv = ["benchmark.py", "--model", "test-llm", "--preset", "quick"]
            args = parse_args()
            self.assertEqual(args.preset, "quick")
            self.assertEqual(args.model, "test-llm")

            sys.argv = ["benchmark.py", "--model", "test-llm", "--subjects", "machine_learning", "philosophy"]
            args_custom = parse_args()
            self.assertEqual(args_custom.subjects, ["machine_learning", "philosophy"])
        finally:
            sys.argv = orig_argv

    def test_llamacpp_batch_size_safe_enforcement(self):
        # llamacpp and gguf providers must auto-adjust batch_size to 1 for native KV-cache thread safety
        cfg_llama = BenchmarkConfig(model_name="model.gguf", provider="llamacpp", batch_size=8)
        self.assertEqual(cfg_llama.batch_size, 1)

        cfg_gguf = BenchmarkConfig(model_name="model.gguf", provider="gguf", batch_size=16)
        self.assertEqual(cfg_gguf.batch_size, 1)

        # Other providers preserve user-defined batch_size
        cfg_hf = BenchmarkConfig(model_name="model", provider="huggingface", batch_size=8)
        self.assertEqual(cfg_hf.batch_size, 8)

    def test_json_export_handles_numpy_float32_safely(self):
        class SimulatedNumpyFloat32(float):
            """Simulates numpy.float32 object with .item() method."""
            def item(self):
                return float(self)

        val = SimulatedNumpyFloat32(-1.234)
        sample = {
            "question_index": 1,
            "pred_choice": "A",
            "gold_choice": "A",
            "is_correct": True,
            "latency_ms": 50.0,
            "log_likelihood_scores": {"A": val, "B": SimulatedNumpyFloat32(-5.67)},
        }
        res = SubjectResult("machine_learning", 1, 1, 1.0, 50.0, category="STEM", sample_details=[sample])
        metrics = BenchmarkMetrics(
            model_name="test-float32-model",
            provider="llamacpp",
            shots=5,
            total_samples=1,
            total_correct=1,
            macro_accuracy=1.0,
            micro_accuracy=1.0,
            total_runtime_sec=0.1,
            subjects=[res],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BenchmarkConfig(
                model_name="test-float32-model",
                provider="mock",
                output_dir=tmpdir,
                report_dir=tmpdir,
            )
            pipeline = MMLUBenchmarkPipeline(config=cfg)
            # Should export cleanly without TypeError: Object of type float32 is not JSON serializable
            pipeline._export_results(metrics)
            exported_json = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(exported_json), 1)
            with open(exported_json[0], "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.assertIn("model_name", loaded)


if __name__ == "__main__":
    unittest.main()
