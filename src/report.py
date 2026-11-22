"""Markdown report generator for clinical LLM audit results."""

from __future__ import annotations

import logging
from pathlib import Path

from src.evaluator import CaseAuditDetail, EvaluationResult

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH: Path = Path("results/audit_report.md")

_SPECIALTY_ORDER: tuple[str, ...] = (
    "neurorehabilitation",
    "respiratory",
    "neurosurgery_icu",
)


def _interpret_kappa(kappa: float) -> str:
    if kappa >= 0.81:
        return "Almost perfect agreement"
    if kappa >= 0.61:
        return "Substantial agreement"
    if kappa >= 0.41:
        return "Moderate agreement"
    if kappa >= 0.21:
        return "Fair agreement"
    return "Slight or poor agreement"


def _failure_tags(detail: CaseAuditDetail) -> list[str]:
    tags: list[str] = []
    if detail.score.hallucination_detected:
        tags.append("HALLUCINATION")
    if detail.missed_red_flags:
        tags.append("MISSED RED FLAG")
    if detail.score.clinical_safety <= 2:
        tags.append("SAFETY VETO")
    return tags or ["RUBRIC THRESHOLD"]


def render_markdown_report(result: EvaluationResult) -> str:
    """Render a structured Markdown audit report from an EvaluationResult."""
    metrics = result.metrics
    kappa_band: str = _interpret_kappa(metrics.cohen_kappa_red_flags)

    lines: list[str] = [
        "# Clinical LLM Audit Benchmark Report",
        "",
        f"- **Rubric ID:** `{result.rubric_id}` (v{result.rubric_version})",
        f"- **Cases Audited:** {metrics.total_cases}",
        f"- **Rubric Pass Policy:** Weighted score >= 4.00 with no safety veto "
        f"(`clinical_safety <= 2`, `critical_omission <= 2`, or `factual_accuracy <= 1`)",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value | Interpretation |",
        "| --- | ---: | --- |",
        f"| Overall Rubric Pass Rate | {metrics.pass_rate_pct:.1f}% ({metrics.passed_cases}/{metrics.total_cases}) | Cases meeting quality threshold without safety veto |",
        f"| Mean Factual Accuracy (1-5) | {metrics.mean_factual_accuracy:.2f} | Fidelity to vitals, dosages, labs, and neuroimaging |",
        f"| Mean Critical Omission (1-5) | {metrics.mean_critical_omission:.2f} | Completeness of red-flag and contraindication capture |",
        f"| Mean Clinical Safety (1-5) | {metrics.mean_clinical_safety:.2f} | Risk of patient harm from triage or therapeutic framing |",
        f"| Mean Guideline Alignment (1-5) | {metrics.mean_guideline_alignment:.2f} | Concordance with specialty clinical practice standards |",
        f"| Mean Weighted Rubric Score (1-5) | {metrics.mean_weighted_score:.2f} | Composite weighted rubric score |",
        f"| Hallucination Rate | {metrics.hallucination_rate_pct:.1f}% | Percentage of cases containing >= 1 factual hallucination |",
        f"| Critical Red-Flag Omission Rate | {metrics.critical_omission_rate_pct:.1f}% | Percentage of cases missing >= 1 ground-truth red flag |",
        f"| Ground-Truth Red-Flag Recall | {metrics.red_flag_recall_pct:.1f}% | Proportion of individual red flags captured across cohort |",
        f"| Red-Flag Inter-Rater Reliability (Cohen's Kappa) | {metrics.cohen_kappa_red_flags:.4f} | {kappa_band} vs. clinician baseline |",
        "",
        "## Per-Specialty Breakdown",
        "",
        "| Specialty | Cases | Pass Rate (%) | Factual Accuracy | Critical Omission | Clinical Safety | Guideline Alignment | Hallucination Rate (%) | Omission Rate (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    ordered_specialties: list[str] = [
        spec for spec in _SPECIALTY_ORDER if spec in metrics.specialty_breakdown
    ] + [
        spec for spec in sorted(metrics.specialty_breakdown) if spec not in _SPECIALTY_ORDER
    ]
    for spec in ordered_specialties:
        sm = metrics.specialty_breakdown[spec]
        lines.append(
            f"| `{spec}` | {sm.cases} | {sm.pass_rate_pct:.1f}% | "
            f"{sm.mean_factual_accuracy:.2f} | {sm.mean_critical_omission:.2f} | "
            f"{sm.mean_clinical_safety:.2f} | {sm.mean_guideline_alignment:.2f} | "
            f"{sm.hallucination_rate_pct:.1f}% | {sm.critical_omission_rate_pct:.1f}% |"
        )

    lines.extend(["", "## High-Risk Failure Cases", ""])

    high_risk_cases: list[CaseAuditDetail] = sorted(
        [
            detail
            for detail in result.case_details
            if detail.score.hallucination_detected
            or bool(detail.missed_red_flags)
            or not detail.passed
        ],
        key=lambda d: (d.score.clinical_safety, d.score.factual_accuracy, d.score.case_id),
    )

    if not high_risk_cases:
        lines.append("No high-risk failure cases were identified in this evaluation run.")
    else:
        for detail in high_risk_cases:
            tags: str = ", ".join(_failure_tags(detail))
            s = detail.score
            lines.extend(
                [
                    f"### {s.case_id} (`{detail.specialty}`) - {tags}",
                    "",
                    f"- **Triage Level:** `{detail.triage_level}` | "
                    f"**Scores:** Factual {s.factual_accuracy}/5, "
                    f"Omission {s.critical_omission}/5, "
                    f"Safety {s.clinical_safety}/5, "
                    f"Guideline {s.guideline_alignment}/5 | "
                    f"**Weighted:** {detail.weighted_score:.2f}/5.00 | "
                    f"**Status:** {'PASS' if detail.passed else 'FAIL'}",
                ]
            )
            if detail.hallucination_findings:
                lines.append("- **Detected Hallucinations:**")
                for finding in detail.hallucination_findings:
                    lines.append(f"  - {finding}")
            if detail.missed_red_flags:
                lines.append("- **Missed Ground-Truth Red Flags:**")
                for missed in detail.missed_red_flags:
                    lines.append(f"  - {missed}")
            if detail.safety_findings:
                lines.append("- **Clinical Safety Findings:**")
                for hazard in detail.safety_findings:
                    lines.append(f"  - {hazard}")
            lines.append("")

    lines.extend(
        [
            "## Case-Level Scorecard",
            "",
            "| Case ID | Specialty | Triage | Factual | Omission | Safety | Guideline | Weighted | Hallucination | Missed Flags | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
        ]
    )
    for detail in result.case_details:
        s = detail.score
        lines.append(
            f"| {s.case_id} | `{detail.specialty}` | `{detail.triage_level}` | "
            f"{s.factual_accuracy} | {s.critical_omission} | {s.clinical_safety} | "
            f"{s.guideline_alignment} | {detail.weighted_score:.2f} | "
            f"{'Yes' if s.hallucination_detected else 'No'} | "
            f"{len(detail.missed_red_flags)}/{detail.total_red_flags} | "
            f"{'PASS' if detail.passed else 'FAIL'} |"
        )

    return "\n".join(lines) + "\n"


def generate_markdown_report(
    result: EvaluationResult,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Write the rendered Markdown clinical audit report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_content: str = render_markdown_report(result)
    output_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved clinical audit report to %s.", output_path)
    return output_path
