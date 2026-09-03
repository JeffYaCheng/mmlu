#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🎯 Pegatron MMLU Benchmark Pipeline - Automated Runner"
echo "=========================================================="

echo "[Step 1/3] Running Unit Test Suite..."
python3 -m unittest discover -s tests -p "test_*.py" -v

echo ""
echo "[Step 2/3] Running Benchmark Sanity Check (--preset quick)..."
python3 -m mmlu_benchmark.benchmark \
    --provider mock \
    --model mock-baseline-7b \
    --preset quick \
    --shots 5 \
    --max-samples 5 \
    --report-dir ./report

echo ""
echo "[Step 3/3] Running Custom Function and Mock Adapter Examples..."
python3 examples/3_evaluate_custom_function.py
python3 examples/4_evaluate_mock_sanity_check.py

echo "=========================================================="
echo "✅ All Automated Benchmark Tests Finished Successfully!"
echo "📂 Reports and visual PNG charts available in ./report and ./benchmark_results"
echo "=========================================================="
