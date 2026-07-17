"""
Tests for the benchmark HTML comparison report.

These run without a browser, an Ollama server, or any API keys; PDF page
rendering is exercised only through its graceful-fallback path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_models import (  # noqa: E402
    BenchmarkModelResult,
    BenchmarkModelSpec,
)
from benchmark_report import (  # noqa: E402
    _compare_key,
    _diff_spans,
    order_results,
    split_markdown_pages,
    write_benchmark_html,
)


def _result(label: str, markdown: str, cost: float) -> BenchmarkModelResult:
    """Build an ``ok`` benchmark result for report tests."""
    return BenchmarkModelResult(
        spec=BenchmarkModelSpec(label=label, kind="api", model=label),
        status="ok",
        elapsed_seconds=1.0,
        page_count=2,
        output_chars=len(markdown),
        cost_usd=cost,
        markdown=markdown,
    )


_TWO_PAGES = (
    "File: deck.pdf; Page: 1\n# Title\n\nBody one\n\n"
    "File: deck.pdf; Page: 2\nBody two\n"
)


def test_split_markdown_pages() -> None:
    """Assembled markdown splits into per-page bodies keyed by page number."""
    pages = split_markdown_pages(_TWO_PAGES)
    assert sorted(pages) == [1, 2]
    assert pages[1] == "# Title\n\nBody one"
    assert pages[2] == "Body two"


def test_split_markdown_pages_without_markers() -> None:
    """Markdown with no page markers collapses to a single page."""
    assert split_markdown_pages("just text") == {1: "just text"}
    assert split_markdown_pages("   ") == {}


def test_compare_key_ignores_markdown_formatting() -> None:
    """Formatting-only differences normalize to the same comparison key."""
    assert _compare_key("**Summarize:**") == _compare_key("Summarize:")
    assert _compare_key("| Text |") == _compare_key("Text")
    assert _compare_key("\n\t ") == " "


def test_diff_ignores_formatting_but_catches_content() -> None:
    """The diff highlights real word changes, not markdown syntax noise."""
    # Same words, different emphasis -> no highlighting.
    html, added, removed = _diff_spans("**Total** revenue", "Total revenue")
    assert added == 0 and removed == 0
    assert "<ins>" not in html and "<del>" not in html

    # A genuinely different word is highlighted both ways.
    html, added, removed = _diff_spans("Total revenue", "Total profit")
    assert added == 1 and removed == 1
    assert "<ins>" in html and "<del>" in html


def test_diff_escapes_html() -> None:
    """Model output containing HTML is escaped, not injected into the page."""
    html, _, _ = _diff_spans("safe", "<script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_order_results_reference_first_then_cheapest() -> None:
    """The reference leads; remaining models follow cheapest-first."""
    expensive = _result("expensive", _TWO_PAGES, 0.90)
    mid = _result("mid", _TWO_PAGES, 0.20)
    free = _result("free", _TWO_PAGES, 0.0)

    ordered = order_results([expensive, mid, free], reference_label="expensive")
    assert [r.spec.label for r in ordered] == ["expensive", "free", "mid"]

    # Defaults to the first successful result when no reference is named.
    assert order_results([mid, free])[0].spec.label == "mid"


def test_order_results_skips_failures() -> None:
    """Skipped and errored models are excluded from the comparison."""
    ok = _result("ok-model", _TWO_PAGES, 0.1)
    skipped = BenchmarkModelResult(
        spec=BenchmarkModelSpec(label="skipped", kind="local", model="x"),
        status="skipped",
        detail="no server",
    )
    assert [r.spec.label for r in order_results([ok, skipped])] == ["ok-model"]
    assert order_results([skipped]) == []


def test_write_benchmark_html(tmp_path) -> None:
    """A full report renders every page and diffs non-reference models."""
    out = str(tmp_path / "report.html")
    written = write_benchmark_html(
        results=[
            _result("best", _TWO_PAGES, 0.5),
            _result("other", _TWO_PAGES.replace("Body two", "Body three"), 0.0),
        ],
        pdf_path="nonexistent.pdf",  # Exercises the no-preview fallback.
        output_path=out,
        reference_label="best",
    )
    assert written == out

    html = open(out, encoding="utf-8").read()
    assert html.count('<section class="page">') == 2
    assert "best" in html and "other" in html
    # Missing PDF degrades to a placeholder rather than failing the report.
    assert "No page preview" in html
    # "two" -> "three" is a real content change and must be highlighted.
    assert "<ins>" in html and "<del>" in html


def test_write_benchmark_html_returns_none_without_results(tmp_path) -> None:
    """No successful models means no report is written."""
    assert write_benchmark_html([], "x.pdf", str(tmp_path / "r.html")) is None


def run_all_tests() -> bool:
    """Run report tests directly (used by run_tests.py-style runners)."""
    import tempfile
    import pathlib

    test_split_markdown_pages()
    test_split_markdown_pages_without_markers()
    test_compare_key_ignores_markdown_formatting()
    test_diff_ignores_formatting_but_catches_content()
    test_diff_escapes_html()
    test_order_results_reference_first_then_cheapest()
    test_order_results_skips_failures()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_benchmark_html(pathlib.Path(tmp))
        test_write_benchmark_html_returns_none_without_results(pathlib.Path(tmp))
    print("  ✓ benchmark report tests passed")
    return True


if __name__ == "__main__":
    run_all_tests()
