"""CLI entry point for the clinical-llm-audit-bench pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.evaluator import DEFAULT_RUBRIC_PATH, EvaluationResult, evaluate_outputs
from src.generator import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, generate_summaries
from src.report import generate_markdown_report

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR: Path = Path("results")


def run_pipeline(
    live: bool = False,
    notes_path: Path = DEFAULT_INPUT_PATH,
    outputs_path: Path = DEFAULT_OUTPUT_PATH,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    model: str | None = None,
) -> EvaluationResult:
    """Execute the end-to-end clinical audit pipeline: generate -> evaluate -> report."""
    results_dir.mkdir(parents=True, exist_ok=True)

    mode_label: str = "live" if live else "mock"
    logger.info("Starting clinical-llm-audit-bench pipeline in %s mode.", mode_label)

    generate_summaries(
        input_path=notes_path,
        output_path=outputs_path,
        mock=not live,
        model=model,
    )

    evaluation_result: EvaluationResult = evaluate_outputs(
        notes_path=notes_path,
        outputs_path=outputs_path,
        rubric_path=rubric_path,
        save_path=results_dir / "audit_results.json",
    )

    report_path: Path = generate_markdown_report(
        result=evaluation_result,
        output_path=results_dir / "audit_report.md",
    )

    metrics = evaluation_result.metrics
    logger.info(
        "Audit complete | Pass rate: %.1f%% (%d/%d) | Hallucination rate: %.1f%% | "
        "Omission rate: %.1f%% | Cohen's kappa: %.4f | Report: %s",
        metrics.pass_rate_pct,
        metrics.passed_cases,
        metrics.total_cases,
        metrics.hallucination_rate_pct,
        metrics.critical_omission_rate_pct,
        metrics.cohen_kappa_red_flags,
        report_path,
    )
    return evaluation_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end clinical LLM audit benchmark pipeline."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Query the configured OpenAI-compatible API instead of using mock LLM outputs.",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to synthetic clinical notes JSON file.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write/read generated LLM summary outputs JSON file.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC_PATH,
        help="Path to the YAML clinical scoring rubric.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where audit results and Markdown report will be saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional LLM model override when running with --live.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    run_pipeline(
        live=args.live,
        notes_path=args.notes,
        outputs_path=args.outputs,
        rubric_path=args.rubric,
        results_dir=args.results_dir,
        model=args.model,
    )


if __name__ == "__main__":
    main()
