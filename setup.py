from setuptools import setup, find_packages

setup(
    name="mmlu_benchmark",
    version="1.0.0",
    description="Pegatron ML Benchmark Pipeline for MMLU Evaluation",
    author="Pegatron ML Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.44.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tabulate>=0.9.0",
        "pydantic>=2.8.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "gguf": ["llama-cpp-python>=0.2.90"],
        "test": ["pytest>=8.3.0", "pytest-asyncio>=0.24.0", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "mmlu-bench=mmlu_benchmark.benchmark:main",
        ],
    },
)
