"""
MMLU Visual Report Generator
============================
Generates standalone HTML visual reports with embedded Chart.js interactive charts,
statistical tables, 4-domain breakdowns, and high-resolution static PNG charts
(Radar charts, Micro vs Macro bar charts, Grouped Domain bar charts, Individual Breakdowns)
saved directly to the `report/` or specified output directory.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from mmlu_benchmark.metrics import BenchmarkMetrics

logger = logging.getLogger("MMLU-ReportGenerator")

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


class ReportGenerator:
    """Generates visual HTML, Markdown, and Matplotlib PNG comparison reports."""

    def __init__(self, output_dir: str = "./report"):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, metrics: BenchmarkMetrics, filename: Optional[str] = None) -> str:
        safe_model_name = metrics.model_name.replace("/", "_").replace(":", "_")
        if filename is None:
            filename = f"mmlu_report_{safe_model_name}_{metrics.shots}shot.html"

        file_path = self.output_path / filename
        subjects_json = [
            {
                "subject": s.subject.replace("_", " ").title(),
                "category": getattr(s, "category", "Other"),
                "accuracy": round(s.accuracy * 100, 2),
                "ci_95": round(s.confidence_interval_95 * 100, 2),
                "total": s.total_samples,
                "correct": s.correct_samples,
                "avg_latency": round(s.avg_latency_ms, 1),
            }
            for s in metrics.subjects
        ]
        subject_labels = [s["subject"] for s in subjects_json]
        accuracy_data = [s["accuracy"] for s in subjects_json]

        # 4 Domain categories
        cat_summary = getattr(metrics, "category_summary", {})
        if not cat_summary and metrics.subjects:
            cat_map: Dict[str, List[float]] = {"STEM": [], "Humanities": [], "Social Sciences": [], "Other": []}
            for s in metrics.subjects:
                c = getattr(s, "category", "Other")
                if c in cat_map:
                    cat_map[c].append(s.accuracy * 100)
                else:
                    cat_map["Other"].append(s.accuracy * 100)
            cat_summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in cat_map.items()}
        else:
            cat_summary = {k: round(v * 100 if v <= 1.0 else v, 2) for k, v in cat_summary.items()}

        radar_labels = ["STEM", "Humanities", "Social Sciences", "Other"]
        radar_values = [cat_summary.get(c, 0.0) for c in radar_labels]

        html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MMLU 評測報告 - {metrics.model_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #10b981;
            --warning: #f59e0b;
            --purple: #8b5cf6;
            --bg: #f8fafc;
            --surface: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            padding: 32px 16px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: white;
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }}
        .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        .header h1 {{ font-size: 26px; font-weight: 800; margin-bottom: 8px; }}
        .header p {{ color: #94a3b8; font-size: 14px; }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--surface);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .stat-label {{ font-size: 13px; color: var(--text-muted); font-weight: 500; margin-bottom: 4px; }}
        .stat-val {{ font-size: 28px; font-weight: 800; color: var(--primary); }}
        .stat-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 860px) {{
            .charts-row {{ grid-template-columns: 1fr; }}
        }}

        .chart-card {{
            background: var(--surface);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border);
            margin-bottom: 24px;
        }}
        .chart-card h2 {{ font-size: 17px; font-weight: 700; margin-bottom: 16px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: #f1f5f9; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 12px; }}
        .domain-tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            background: #e2e8f0;
            color: #334155;
        }}
        .domain-tag.STEM {{ background: #dbeafe; color: #1d4ed8; }}
        .domain-tag.Humanities {{ background: #f3e8ff; color: #7e22ce; }}
        .domain-tag.Social-Sciences {{ background: #d1fae5; color: #047857; }}
        .domain-tag.Other {{ background: #fef3c7; color: #b45309; }}
        
        .progress-bar-bg {{ background: #e2e8f0; border-radius: 999px; height: 8px; width: 80px; display: inline-block; vertical-align: middle; margin-right: 8px; }}
        .progress-bar-fill {{ background: var(--primary); height: 100%; border-radius: 999px; }}
        .footer {{ text-align: center; font-size: 13px; color: var(--text-muted); margin-top: 32px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">MMLU EVALUATION REPORT</span>
            <h1>評測模型: {metrics.model_name}</h1>
            <p>設定: {metrics.shots}-Shot | 提供者: {metrics.provider} | 總耗時: {metrics.total_runtime_sec:.1f} 秒</p>
        </div>
        <div class="grid-stats">
            <div class="stat-card">
                <div class="stat-label">Micro Accuracy (整體準確率)</div>
                <div class="stat-val">{metrics.micro_accuracy * 100:.2f}%</div>
                <div class="stat-sub">{metrics.total_correct} / {metrics.total_samples} 題正確</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Macro Accuracy (跨學科平均)</div>
                <div class="stat-val">{metrics.macro_accuracy * 100:.2f}%</div>
                <div class="stat-sub">涵蓋 {len(metrics.subjects)} 個評測學科</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">四大領域平均 (STEM / 人文 / 社科 / 其他)</div>
                <div class="stat-val">{sum(radar_values)/max(1, len(radar_values)):.1f}%</div>
                <div class="stat-sub">STEM: {radar_values[0]:.1f}% | 人文: {radar_values[1]:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均推理延遲</div>
                <div class="stat-val">{sum(s.avg_latency_ms for s in metrics.subjects)/max(1, len(metrics.subjects)):.1f} ms</div>
                <div class="stat-sub">單題平均響應時間</div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-card" style="margin-bottom: 0;">
                <h2>🎯 四大領域能力雷達圖 (Radar Chart)</h2>
                <div style="height: 280px; position: relative;">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
            <div class="chart-card" style="margin-bottom: 0;">
                <h2>📊 四大領域表現長條圖 (Category Accuracy)</h2>
                <div style="height: 280px; position: relative;">
                    <canvas id="categoryBarChart"></canvas>
                </div>
            </div>
        </div>

        <div class="chart-card">
            <h2>📈 各學科詳細準確率分佈 (Accuracy by Subject %)</h2>
            <div style="height: 320px; position: relative;">
                <canvas id="accuracyChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h2>學科詳細數據列表</h2>
            <table>
                <thead>
                    <tr>
                        <th>學科名稱</th>
                        <th>領域分類</th>
                        <th>樣本總數</th>
                        <th>答對題數</th>
                        <th>準確率 (%)</th>
                        <th>95% 信賴區間</th>
                        <th>平均延遲</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join([f'''
                    <tr>
                        <td><strong>{s['subject']}</strong></td>
                        <td><span class="domain-tag {s['category'].replace(" ", "-")}">{s['category']}</span></td>
                        <td>{s['total']}</td>
                        <td>{s['correct']}</td>
                        <td>
                            <div class="progress-bar-bg"><div class="progress-bar-fill" style="width: {s['accuracy']}%;"></div></div>
                            <strong>{s['accuracy']}%</strong>
                        </td>
                        <td>±{s['ci_95']}%</td>
                        <td>{s['avg_latency']} ms</td>
                    </tr>
                    ''' for s in subjects_json])}
                </tbody>
            </table>
        </div>
        <div class="footer">
            Generated automatically by Pegatron ML Benchmark Pipeline | Report directory: <code>./report/</code>
        </div>
    </div>
    <script>
        // 1. Radar Chart
        new Chart(document.getElementById('radarChart').getContext('2d'), {{
            type: 'radar',
            data: {{
                labels: {json.dumps(radar_labels)},
                datasets: [{{
                    label: '{metrics.model_name}',
                    data: {json.dumps(radar_values)},
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    pointBackgroundColor: 'rgba(37, 99, 235, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{ stepSize: 20 }}
                    }}
                }}
            }}
        }});

        // 2. Category Bar Chart
        new Chart(document.getElementById('categoryBarChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(radar_labels)},
                datasets: [{{
                    label: '領域準確率 (%)',
                    data: {json.dumps(radar_values)},
                    backgroundColor: ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'],
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{ callback: function(val) {{ return val + '%'; }} }}
                    }}
                }}
            }}
        }});

        // 3. Subject Breakdown Bar Chart
        new Chart(document.getElementById('accuracyChart').getContext('2d'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(subject_labels)},
                datasets: [{{
                    label: '學科準確率 (%)',
                    data: {json.dumps(accuracy_data)},
                    backgroundColor: 'rgba(37, 99, 235, 0.85)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1.5,
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{ callback: function(val) {{ return val + '%'; }} }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Also generate static PNG plots if matplotlib is available
        self.generate_png_charts(metrics)
        return str(file_path)

    def generate_png_charts(
        self,
        metrics_or_multi_results: Union[BenchmarkMetrics, Dict[str, Any], List[BenchmarkMetrics]],
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Generates 4 high-resolution comparison and breakdown figures using matplotlib:
        1. Radar Chart (mmlu_comparison_radar.png)
        2. Overall Accuracy Bar Chart (mmlu_overall_accuracy_bar.png)
        3. Grouped Category Comparison Bar Chart (mmlu_category_comparison_bar.png)
        4. Individual Model Category Breakdown (mmlu_individual_model_breakdown.png)
        """
        if plt is None or np is None:
            logger.info("matplotlib/numpy not available. Skipping static PNG generation.")
            return []

        out_path = Path(output_dir) if output_dir else self.output_path
        out_path.mkdir(parents=True, exist_ok=True)

        # Standardize inputs into a unified dictionary structure
        model_dict: Dict[str, Dict[str, Any]] = {}
        if isinstance(metrics_or_multi_results, BenchmarkMetrics):
            m = metrics_or_multi_results
            model_dict[m.model_name] = {
                "micro_accuracy": m.micro_accuracy,
                "macro_accuracy": m.macro_accuracy,
                "category_summary": {
                    k: (v / 100.0 if v > 1.0 else v) for k, v in m.category_summary.items()
                },
            }
        elif isinstance(metrics_or_multi_results, list):
            for m in metrics_or_multi_results:
                model_dict[m.model_name] = {
                    "micro_accuracy": m.micro_accuracy,
                    "macro_accuracy": m.macro_accuracy,
                    "category_summary": {
                        k: (v / 100.0 if v > 1.0 else v) for k, v in m.category_summary.items()
                    },
                }
        elif isinstance(metrics_or_multi_results, dict):
            for k, v in metrics_or_multi_results.items():
                if isinstance(v, BenchmarkMetrics):
                    model_dict[k] = {
                        "micro_accuracy": v.micro_accuracy,
                        "macro_accuracy": v.macro_accuracy,
                        "category_summary": {
                            ck: (cv / 100.0 if cv > 1.0 else cv) for ck, cv in v.category_summary.items()
                        },
                    }
                elif isinstance(v, dict):
                    cat_s = v.get("category_summary", {})
                    model_dict[k] = {
                        "micro_accuracy": v.get("micro_accuracy", 0.0),
                        "macro_accuracy": v.get("macro_accuracy", 0.0),
                        "category_summary": {
                            ck: (cv / 100.0 if cv > 1.0 else cv) for ck, cv in cat_s.items()
                        },
                    }

        if not model_dict:
            return []

        categories = ["STEM", "Humanities", "Social Sciences", "Other"]
        model_keys = list(model_dict.keys())
        model_labels = [m.split("/")[-1] for m in model_keys]
        palette = ["#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b"]
        markers = ["o", "s", "^", "D", "v", "p", "*"]
        linestyles = ["-", "--", "-.", ":", "-", "--", "-."]
        exported_files = []

        # -------------------------------------------------------------
        # 1. Enhanced Radar Chart (Distinct colors, markers, alpha fills)
        # -------------------------------------------------------------
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8.0, 8.0), subplot_kw=dict(polar=True))
        for idx, (m_name, rep) in enumerate(model_dict.items()):
            label = m_name.split("/")[-1]
            color = palette[idx % len(palette)]
            marker = markers[idx % len(markers)]
            ls = linestyles[idx % len(linestyles)]
            vals = [rep["category_summary"].get(c, 0.0) * 100 for c in categories]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2.5, linestyle=ls, marker=marker, markersize=7.5, label=label, color=color)
            ax.fill(angles, vals, alpha=0.10, color=color)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(color="#cbd5e1", linestyle="--", linewidth=0.8)
        plt.title("MMLU Category Accuracy Across Models (Radar Chart)", size=14, y=1.10, fontweight="bold", pad=20)
        plt.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), frameon=True, facecolor="white", edgecolor="#e2e8f0", borderpad=1.0)
        radar_path = out_path / "mmlu_comparison_radar.png"
        plt.savefig(radar_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        exported_files.append(str(radar_path))

        # -------------------------------------------------------------
        # 2. Overall Accuracy Bar Chart (Micro vs Macro with Values)
        # -------------------------------------------------------------
        micros = [model_dict[m]["micro_accuracy"] * 100 for m in model_keys]
        macros = [model_dict[m]["macro_accuracy"] * 100 for m in model_keys]

        x = np.arange(len(model_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8.5, len(model_labels) * 2.8), 5.8))
        rects1 = ax.bar(x - width / 2, micros, width, label="Micro Accuracy (%)", color="#2563eb", edgecolor="white", linewidth=1.2)
        rects2 = ax.bar(x + width / 2, macros, width, label="Macro Accuracy (%)", color="#10b981", edgecolor="white", linewidth=1.2)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11, labelpad=10)
        ax.set_title("Overall MMLU Benchmark Accuracy Comparison", fontweight="bold", fontsize=14, pad=18)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontweight="bold", fontsize=11)
        ax.set_ylim(0, max(100, max(micros + macros + [10]) * 1.15))
        ax.legend(frameon=True, facecolor="white", borderpad=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        bar_path = out_path / "mmlu_overall_accuracy_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        exported_files.append(str(bar_path))

        # -------------------------------------------------------------
        # 3. Grouped Category Comparison Bar Chart
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        x_cat = np.arange(len(categories))
        num_models = len(model_keys)
        bar_w = 0.8 / max(1, num_models)

        for idx, m_key in enumerate(model_keys):
            cat_vals = [model_dict[m_key]["category_summary"].get(c, 0.0) * 100 for c in categories]
            pos = x_cat - 0.4 + (idx + 0.5) * bar_w
            rects = ax.bar(pos, cat_vals, bar_w, label=model_labels[idx], color=palette[idx % len(palette)], edgecolor="white", linewidth=1.2)
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_ylabel("Accuracy (%)", fontweight="bold", fontsize=11, labelpad=10)
        ax.set_title("MMLU 4 Domain Performance Comparison Across Models", fontweight="bold", fontsize=14, pad=18)
        ax.set_xticks(x_cat)
        ax.set_xticklabels(categories, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(title="Models", frameon=True, facecolor="white", borderpad=0.8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        cat_bar_path = out_path / "mmlu_category_comparison_bar.png"
        plt.savefig(cat_bar_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        exported_files.append(str(cat_bar_path))

        # -------------------------------------------------------------
        # 4. Individual Model Category Breakdown Subplots (Spaced Out)
        # -------------------------------------------------------------
        num_m = len(model_keys)
        fig, axes = plt.subplots(1, num_m, figsize=(max(5.5 * num_m, 6.5), 5.2), squeeze=False)
        for idx, (m_key, ax_sub) in enumerate(zip(model_keys, axes[0])):
            m_lbl = model_labels[idx]
            cat_scores = [model_dict[m_key]["category_summary"].get(c, 0.0) * 100 for c in categories]
            cat_colors = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"]
            bars = ax_sub.bar(categories, cat_scores, color=cat_colors, edgecolor="white", width=0.55, linewidth=1.2)

            for bar in bars:
                h = bar.get_height()
                ax_sub.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

            ax_sub.set_title(f"Model: {m_lbl}", fontweight="bold", fontsize=12, pad=12)
            ax_sub.set_ylim(0, 105)
            ax_sub.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=8)
            ax_sub.tick_params(axis="x", rotation=25)
            ax_sub.grid(axis="y", linestyle="--", alpha=0.4)

        plt.suptitle("Individual Model Performance Breakdown by Domain", fontsize=14, fontweight="bold", y=1.03)
        plt.subplots_adjust(wspace=0.38, top=0.86, bottom=0.18)
        breakdown_path = out_path / "mmlu_individual_model_breakdown.png"
        plt.savefig(breakdown_path, dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        exported_files.append(str(breakdown_path))

        logger.info(f"Visual charts exported to {out_path}: {[os.path.basename(f) for f in exported_files]}")
        return exported_files

