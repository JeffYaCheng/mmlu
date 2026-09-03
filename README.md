# ML Benchmark Pipeline - MMLU Evaluation Suite
=========================================================
> **針對自研與微調大型語言模型（In-House & Fine-Tuned LLMs）打造的現代化、高擴充 MMLU (Massive Multitask Language Understanding) 標準評測工具箱。**

本專案旨在提供工程師最快速、最標準化的方式，在本地環境（Local Workstation / GPU Server / Google Colab）評測自研 LLM 模型的推理與多學科能力。

全面採用 **學術標準 Next-Token Log-Likelihood 條件對數機率評測**（消除生成隨機性與解析誤差，與 Dan Hendrycks et al. 原論文及 Open LLM Leaderboard 標準完全一致），支援 **4 大核心 Adapter 接入架構**、**0-Shot 至 5-Shot 提示工程**、**四大領域平衡子集評測 (STEM、Humanities、Social Sciences、Other)**、**4-Bit NF4 量化**、**單批次預載杜絕卡頓**，並自動輸出 **JSON / Markdown / HTML 報告 / 4 種 300 DPI 視覺化圖表 (Radar Chart, Grouped Bar Charts)**。

---

## 🌟 核心特色與工程優勢 (Key Features)

1. **4 大專屬工程師接入 Adapter (The 4 Core Model Adapters)**：
   - 🧩 **`huggingface` (HuggingFaceModelAdapter)**：本地 Hugging Face 權重資料夾、LoRA/SFT Checkpoint 或 Hub 模型 ID，支援 4-bit NF4 量化。
   - 🦙 **`llamacpp` (LlamaCppModelAdapter)**：本地 `.gguf` 格式邊緣量化模型，支援 GPU Layer Offload 與 Logprob 評測。
   - ⚙️ **`custom` (CustomFunctionAdapter)**：工程師自訂任意 Python 推論/Scoring 函式（回傳各選項 Next-Token Log-Likelihood 對數機率字典 `{"A": ..., "B": ..., "C": ..., "D": ...}`，由管線執行確定性 ArgMax 判定）。
   - 🧪 **`mock` (MockModelAdapter)**：0 顯存、0 網路依賴的秒級驗證基準，適合 CI/CD 與格式除錯。
2. **學術標準條件對數機率 (Academic Standard Log-Likelihood)**：
   - 計算候選選項 `['A', 'B', 'C', 'D']` 之條件對數機率 $P(\text{choice} \mid \text{prompt})$：
     $$\text{Choice}^* = \arg\max_{c \in \{A, B, C, D\}} \log P(\text{token} = c \mid \text{prompt})$$
   - 完全杜絕自由生成的格式解析誤差與隨機採樣漂移。
3. **四大領域平衡代表子集 (4-Domain Balanced Preset)**：
   - `--preset balanced`：在 STEM、人文、社科、其他領域各精選 3 門代表性學科（共 12 門），保證跨領域雷達圖與長條圖完整展開，不重疊擠壓。
   - `--preset full`：完整評測 57 個官方學科。
   - `--preset quick`：2 門學科快速驗證（適用於 CI/CD 與 Smoke Test）。
4. **單批次全量預載技術 (Single-Batch Preload)**：
   - 一次性下載與快取 MMLU 題庫，杜絕 HuggingFace 頻繁連線的 Rate Limit 與網路延遲問題。
5. **4 種 300 DPI 高解析度圖表自動生成 (`report/`)**：
   - 🎯 `mmlu_comparison_radar.png`：跨四大領域能力雷達圖。
   - 📊 `mmlu_overall_accuracy_bar.png`：整體 Micro/Macro 準確率長條圖。
   - 📊 `mmlu_category_comparison_bar.png`：跨模型四大領域分組長條圖。
   - 📊 `mmlu_individual_model_breakdown.png`：各模型領域獨立子圖。

---

## 🚀 快速開始 (Quick Start)

### 1. Clone 專案與安裝環境

```bash
# 1. 複製 GitHub 專案庫
git clone https://github.com/JeffYaCheng/mmlu.git
cd mmlu

# 2. 建立並啟動虛擬環境 (建議 Python 3.10+)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安裝相依套件
pip install -r requirements.txt

# 4. 安裝為本地可編輯套件 (啟用 mmlu-bench CLI 指令)
pip install -e .
```

---

## 🛠️ 開發工程師 4 大 Adapter 接入範例 (Developer Guide)

本專案在 `examples/` 目錄下提供 4 個立即可執行的範例腳本：

### 1. Hugging Face Adapter (`examples/1_evaluate_huggingface_model.py`)
> 適用於評測本地微調權重資料夾（LoRA / SFT）或 HuggingFace Hub 上的模型。

```bash
# CLI 指令快速評測 (支援 4-bit 量化與 Log-Likelihood)
mmlu-bench \
    --provider huggingface \
    --model ./my_finetuned_checkpoint \
    --preset balanced \
    --shots 5 \
    --report-dir ./report
```

```python
import asyncio
from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS

async def main():
    config = BenchmarkConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="huggingface",
        subjects=BALANCED_BENCHMARK_SUBJECTS,
        shots=5,
        load_in_4bit=True,
        report_dir="./report",
    )
    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()
    print(metrics.to_markdown_summary())

asyncio.run(main())
```

---

### 2. llama.cpp / GGUF Adapter (`examples/2_evaluate_llamacpp_gguf.py`)
> 適用於在邊緣端或本機直接評測量化好的 `.gguf` 檔案。

```bash
# 需先安裝 llama-cpp-python
pip install llama-cpp-python

mmlu-bench \
    --provider llamacpp \
    --model ./gemma-3-12b-it-Q4_0.gguf \
    --preset balanced \
    --shots 5 \
    --report-dir ./report
```

```python
import asyncio
from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS

async def main():
    config = BenchmarkConfig(
        model_name="./gemma-3-12b-it-Q4_0.gguf",
        provider="llamacpp",
        subjects=BALANCED_BENCHMARK_SUBJECTS,
        shots=5,
        report_dir="./report",
    )
    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()
    print(metrics.to_markdown_summary())

asyncio.run(main())
```

---

### 3. 自訂 Python 函式 Adapter (`examples/3_evaluate_custom_function.py`)
> 適用於非標準結構、自研類別、C++ 封裝或自定義 Logit Scoring 邏輯。

```python
import asyncio
from typing import Dict
from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS

# 1. 定義您的自訂推論/打分函式 (回傳 Next-Token Log-Likelihood 字典)
def my_custom_scoring_fn(prompt: str) -> Dict[str, float]:
    # 純對數概似值 (Log-Likelihood) 評測：回傳各選項對數機率或 logits，由管線取 ArgMax
    return {"A": -1.5, "B": -0.2, "C": -2.8, "D": -3.1}

# 2. 啟動評測管線 (可使用 preset="balanced" 或直接省略 subjects)
async def main():
    config = BenchmarkConfig(
        model_name="My-InHouse-LLM",
        provider="custom",
        preset="balanced",   # 支援 "balanced" (12科), "full" (57科), "quick" (2科)，或自訂 subjects=[...]
        shots=5,
        report_dir="./report",
    )
    pipeline = MMLUBenchmarkPipeline(config=config, custom_fn=my_custom_scoring_fn)
    metrics = await pipeline.run_benchmark()
    print(metrics.to_markdown_summary())

asyncio.run(main())
```

---

### 4. Mock 基準 Adapter (`examples/4_evaluate_mock_sanity_check.py`)
> 0 GPU、0 網路需求，毫秒級完成管線驗證與 CI/CD 測試。

```bash
mmlu-bench --provider mock --model mock-baseline --preset quick
```

```python
import asyncio
from mmlu_benchmark.benchmark import BenchmarkConfig, MMLUBenchmarkPipeline
from mmlu_benchmark.dataset import BALANCED_BENCHMARK_SUBJECTS

async def main():
    config = BenchmarkConfig(
        model_name="mock-baseline",
        provider="mock",
        subjects=BALANCED_BENCHMARK_SUBJECTS,
        shots=5,
        max_samples_per_subject=5,
        report_dir="./report",
    )
    pipeline = MMLUBenchmarkPipeline(config=config)
    metrics = await pipeline.run_benchmark()
    print(metrics.to_markdown_summary())

asyncio.run(main())
```

---

## ⚡ Google Colab 評測指南

若本地缺乏 GPU，可直接利用 Google Colab 免費 GPU：

1. 開啟專案根目錄的 `mmlu_colab_benchmark.ipynb`，內建 4 大 Adapter 的即用程式碼區塊。
2. 或在 Colab 終端機執行：
```bash
python3 colab_run_full_benchmark.py \
    --provider huggingface \
    --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct \
    --preset balanced \
    --shots 5 \
    --report-dir ./report
```

---

## 🧪 單元測試與自動化驗證 (CI/CD)

專案內建完整單元測試與端到端測試套件（涵蓋 40+ 種極端邊界測試案例）：

```bash
# 執行所有單元與整合測試
python3 -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📁 專案目錄結構 (Repository Structure)

```text
├── mmlu_benchmark/                       # 核心評測 Python 套件
│   ├── __init__.py                       # 模組匯出 (4 大 Adapter 與 Pipeline)
│   ├── benchmark.py                      # 評測調度器與 CLI 進入點 (Log-Likelihood)
│   ├── models.py                         # 4 大模型適配器 (HuggingFace, LlamaCpp, Custom, Mock)
│   ├── dataset.py                        # 57 學科分類庫、四大領域平衡子集、預載資料載入器
│   ├── metrics.py                        # Micro/Macro 準確率計算與 95% 信賴區間
│   └── report_generator.py               # HTML 互動報告與 4 種 300 DPI Matplotlib 圖表生成器
├── examples/                             # 開發工程師 4 大 Adapter 範例
│   ├── 1_evaluate_huggingface_model.py   # 範例 1: HuggingFace Checkpoint 評測
│   ├── 2_evaluate_llamacpp_gguf.py       # 範例 2: llama.cpp / GGUF 模型評測
│   ├── 3_evaluate_custom_function.py     # 範例 3: 自訂 Python 推論函式評測
│   └── 4_evaluate_mock_sanity_check.py   # 範例 4: Mock 快速驗證
├── colab_run_full_benchmark.py           # Google Colab / GPU 叢集獨立執行腳本
├── mmlu_colab_benchmark.ipynb            # Google Colab 互動式 Jupyter Notebook
├── tests/
│   └── test_benchmark.py                 # 單元測試套件
├── scripts/
│   └── run_benchmark.sh                  # 自動化建置與測試腳本
├── requirements.txt                      # Python 相依套件清單
├── setup.py                              # Python Package 定義 (提供 mmlu-bench 指令)
└── README.md                             # 專案完整說明文件
```

---

## 📄 授權條款 (License)

本專案採用 Apache 2.0 授權條款。評測資料集遵循原作者 Dan Hendrycks et al. (MMLU Benchmark) 規範。
