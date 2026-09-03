"""
Benchmark Metrics & Statistical Confidence Calculator
===================================================
Provides structured calculation of:
- Micro and Macro Accuracy
- 95% Normal-Approximation Confidence Intervals
- Markdown & JSON reporting
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SubjectResult:
    subject: str
    total_samples: int
    correct_samples: int
    accuracy: float
    avg_latency_ms: float
    category: str = "Other"
    shots_used: int = 5
    sample_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def confidence_interval_95(self) -> float:
        """Calculates standard 95% normal-approximation margin of error: 1.96 * sqrt(p*(1-p)/n)"""
        if self.total_samples == 0:
            return 0.0
        p = self.accuracy
        n = self.total_samples
        margin = 1.96 * math.sqrt(max(0, p * (1 - p)) / n)
        return round(margin, 4)


@dataclass
class BenchmarkMetrics:
    model_name: str
    provider: str
    shots: int
    total_samples: int
    total_correct: int
    macro_accuracy: float
    micro_accuracy: float
    total_runtime_sec: float
    subjects: List[SubjectResult] = field(default_factory=list)
    category_summary: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.category_summary and self.subjects:
            cat_map: Dict[str, List[float]] = {
                "STEM": [],
                "Humanities": [],
                "Social Sciences": [],
                "Other": [],
            }
            for s in self.subjects:
                cat = s.category if s.category in cat_map else "Other"
                cat_map[cat].append(s.accuracy)
            
            self.category_summary = {
                cat: round(sum(accs) / len(accs), 4) if accs else 0.0
                for cat, accs in cat_map.items()
            }

    @property
    def avg_latency_ms(self) -> float:
        """Returns average latency across all evaluated subject questions in milliseconds."""
        if not self.subjects:
            return 0.0
        latencies = [s.avg_latency_ms for s in self.subjects if s.avg_latency_ms > 0]
        return round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    @property
    def correct_count(self) -> int:
        return self.total_correct

    @property
    def total_questions(self) -> int:
        return self.total_samples

    @property
    def subject_results(self) -> List[SubjectResult]:
        return self.subjects

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "shots": self.shots,
            "total_samples": self.total_samples,
            "total_correct": self.total_correct,
            "macro_accuracy": round(self.macro_accuracy, 4),
            "micro_accuracy": round(self.micro_accuracy, 4),
            "total_runtime_sec": round(self.total_runtime_sec, 2),
            "category_summary": self.category_summary,
            "subjects": [
                {
                    "subject": s.subject,
                    "category": s.category,
                    "shots_used": s.shots_used,
                    "total": s.total_samples,
                    "correct": s.correct_samples,
                    "accuracy": round(s.accuracy, 4),
                    "ci_95": s.confidence_interval_95,
                    "avg_latency_ms": round(s.avg_latency_ms, 2),
                    "sample_details": s.sample_details,
                }
                for s in self.subjects
            ],
        }

    def to_markdown_summary(self) -> str:
        md = []
        md.append(f"# 📊 MMLU Benchmark Report: {self.model_name}")
        md.append(f"**Provider**: `{self.provider}` | **Setting**: `{self.shots}-Shot` | **Runtime**: `{self.total_runtime_sec:.1f}s`\n")
        md.append("### Summary Metrics")
        md.append(f"- **Micro Accuracy**: **{self.micro_accuracy * 100:.2f}%** ({self.total_correct}/{self.total_samples})")
        md.append(f"- **Macro Accuracy**: **{self.macro_accuracy * 100:.2f}%**")
        
        if self.category_summary:
            md.append("\n### 4-Domain Breakdown (STEM, Humanities, Social Sciences, Other)")
            md.append("| Domain | Accuracy (%) |")
            md.append("|:---|:---:|")
            for cat, acc in self.category_summary.items():
                md.append(f"| **{cat}** | **{acc * 100:.2f}%** |")

        md.append("\n### Subject Breakdown\n")
        md.append("| Subject | Domain | Samples | Correct | Accuracy (%) | 95% CI (±) | Avg Latency (ms) |")
        md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|")
        for s in self.subjects:
            md.append(
                f"| {s.subject} | {s.category} | {s.total_samples} | {s.correct_samples} | "
                f"**{s.accuracy * 100:.2f}%** | ±{s.confidence_interval_95 * 100:.2f}% | {s.avg_latency_ms:.1f}ms |"
            )
        return "\n".join(md)
