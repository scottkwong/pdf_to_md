"""
Benchmark digital PDF text parser engines.

This module compares first-pass digital text parsing speed across supported
parser engines. It avoids model calls and can run against either a user-
provided PDF or generated deterministic fixture PDFs from the test suite.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List

from digital_text_parsers import (
    create_digital_text_parser,
    is_pymupdf_available,
)
from tests.create_test_pdf import ensure_default_fixture_pdfs


@dataclass
class ParserBenchmarkStats:
    """Per-parser benchmark statistics.

    Attributes:
        parser_name: Parser identifier.
        runs: Number of completed benchmark runs.
        durations: Durations for each run in seconds.
        mean_seconds: Mean duration in seconds.
        stdev_seconds: Sample standard deviation in seconds.
    """

    parser_name: str
    runs: int
    durations: List[float]
    mean_seconds: float
    stdev_seconds: float


@dataclass
class BenchmarkSummary:
    """Summary output for parser benchmark execution."""

    pdf_paths: List[str]
    stats: Dict[str, ParserBenchmarkStats]
    winner_parser: str


def _resolve_benchmark_inputs(pdf_path: str | None) -> List[str]:
    """Resolve benchmark input PDFs from override path or generated fixtures.

    Args:
        pdf_path: Optional user-provided PDF path.

    Returns:
        List of PDF paths to benchmark.

    Raises:
        ValueError: If provided path does not exist or is not a file.
    """
    if pdf_path:
        if not os.path.isfile(pdf_path):
            raise ValueError(
                f"Benchmark PDF path does not exist or is not a file: {pdf_path}"
            )
        return [pdf_path]

    return ensure_default_fixture_pdfs()


def _run_parser_once(parser_name: str, pdf_paths: List[str]) -> float:
    """Run one extraction pass for a parser engine across all PDFs.

    Args:
        parser_name: Parser identifier (`pypdf` or `pymupdf`).
        pdf_paths: Ordered list of PDF paths to process.

    Returns:
        Elapsed duration in seconds.
    """
    parser = create_digital_text_parser(parser_name).parser
    start = time.perf_counter()
    for path in pdf_paths:
        parser.extract_pages(path)
    return time.perf_counter() - start


def _available_parsers() -> List[str]:
    """Return benchmarkable parser identifiers for current environment."""
    backends = ["pypdf"]
    if is_pymupdf_available():
        backends.append("pymupdf")
    return backends


def benchmark_digital_text_parsers(
    runs: int = 10,
    pdf_path: str | None = None,
) -> BenchmarkSummary:
    """Benchmark digital text parser engines.

    Args:
        runs: Number of runs per parser.
        pdf_path: Optional user-provided PDF path override.

    Returns:
        Benchmark summary with per-parser statistics and winner.

    Raises:
        ValueError: If runs is invalid or benchmark input is invalid.
    """
    if runs < 1:
        raise ValueError("Benchmark runs must be >= 1.")

    paths = _resolve_benchmark_inputs(pdf_path=pdf_path)
    stats: Dict[str, ParserBenchmarkStats] = {}

    for parser_name in _available_parsers():
        durations: List[float] = []
        for _ in range(runs):
            durations.append(
                _run_parser_once(parser_name=parser_name, pdf_paths=paths)
            )

        mean_seconds = statistics.mean(durations)
        stdev_seconds = statistics.stdev(durations) if runs > 1 else 0.0
        stats[parser_name] = ParserBenchmarkStats(
            parser_name=parser_name,
            runs=runs,
            durations=durations,
            mean_seconds=mean_seconds,
            stdev_seconds=stdev_seconds,
        )

    winner_parser = min(stats.items(), key=lambda item: item[1].mean_seconds)[0]
    return BenchmarkSummary(
        pdf_paths=paths,
        stats=stats,
        winner_parser=winner_parser,
    )


def print_benchmark_report(summary: BenchmarkSummary) -> None:
    """Print benchmark summary report.

    Args:
        summary: Completed benchmark summary.
    """
    print("=" * 70)
    print("Digital Text Parser Benchmark (direct package extraction)")
    print("=" * 70)
    print("Input PDFs:")
    for path in summary.pdf_paths:
        print(f"  - {path}")
    print()
    print("Parser results:")
    for parser_name, parser_stats in sorted(summary.stats.items()):
        print(
            f"  {parser_name:10} "
            f"mean={parser_stats.mean_seconds:.6f}s "
            f"stdev={parser_stats.stdev_seconds:.6f}s "
            f"runs={parser_stats.runs}"
        )
    print()
    print(f"Winner (fastest mean): {summary.winner_parser}")
    print("=" * 70)
