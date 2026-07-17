"""
Self-contained HTML comparison report for cross-provider benchmark runs.

``--benchmark`` extracts the same PDF with several models. This module turns
those results into a single scrollable HTML file that puts each rendered PDF
page next to every model's markdown for that page, word-diffed against a
reference model so quality differences are visible at a glance.

The report is one self-contained file: page images are embedded as data URIs
and all CSS is inline, so it can be opened straight from disk or moved around
without losing anything.
"""
from __future__ import annotations

import base64
import html
import io
import os
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    from benchmark_models import BenchmarkModelResult

# Page separator emitted by VisionExtractor, e.g. "File: deck.pdf; Page: 3".
_PAGE_MARKER_RE = re.compile(r"^File: .*?; Page: (\d+)\s*$", re.MULTILINE)

# Split into words while keeping the whitespace runs, so the diff aligns on
# words but reassembles with the original spacing intact.
_TOKEN_RE = re.compile(r"\S+|\s+")


def split_markdown_pages(markdown: str) -> Dict[int, str]:
    """Split assembled markdown into per-page text keyed by page number.

    Args:
        markdown: Full document markdown containing ``File: ...; Page: N``
            separators.

    Returns:
        Mapping of 1-based page number to that page's markdown body. Content
        before the first marker is ignored; an unmarked document maps to
        ``{1: markdown}``.
    """
    matches = list(_PAGE_MARKER_RE.finditer(markdown))
    if not matches:
        return {1: markdown.strip()} if markdown.strip() else {}

    pages: Dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        pages[int(match.group(1))] = markdown[start:end].strip()
    return pages


def render_pdf_page_images(
    pdf_path: str,
    dpi: int = 150,
    max_width: int = 1600,
) -> Dict[int, str]:
    """Render each PDF page to a base64 JPEG data URI for embedding.

    Rendered generously: the page column is drag-resizable, so an image that
    only suited the default width would blur as soon as it was widened.

    Args:
        pdf_path: PDF to render.
        dpi: Render resolution before downscaling.
        max_width: Downscale pages wider than this, to bound the HTML size.

    Returns:
        Mapping of 1-based page number to a ``data:image/jpeg;base64,...`` URI.
        Returns an empty mapping if the PDF cannot be rendered, so a report is
        still produced without page previews.
    """
    try:
        from pdf2image import convert_from_path  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - optional at report time
        return {}

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except Exception:  # pragma: no cover - poppler/render failure
        return {}

    encoded: Dict[int, str] = {}
    for index, image in enumerate(images, start=1):
        if image.width > max_width:
            height = int(image.height * (max_width / image.width))
            image = image.resize((max_width, height))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=72)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        encoded[index] = f"data:image/jpeg;base64,{data}"
    return encoded


def _compare_key(token: str) -> str:
    """Normalize a token so the diff tracks content, not markdown formatting.

    Diffing raw markdown makes every model light up red and green over
    cosmetic differences (``**Summarize:**`` vs ``Summarize:``, table pipes,
    bullet characters), which buries the differences that matter. Comparing on
    a stripped, lowercased key means only genuinely different words highlight,
    while the original markdown is still what gets displayed.

    Args:
        token: A word or whitespace run from the source markdown.

    Returns:
        A normalization key: a single space for whitespace, or the token's
        alphanumeric content lowercased (possibly empty for pure syntax).
    """
    if not token.strip():
        return " "
    return re.sub(r"[^0-9a-z]", "", token.lower())


def _diff_spans(reference: str, candidate: str) -> Tuple[str, int, int]:
    """Word-diff ``candidate`` against ``reference`` as highlighted HTML.

    Matching runs on normalized keys (see ``_compare_key``) so formatting
    noise does not register as a difference, while the rendered output keeps
    the model's original markdown.

    Args:
        reference: Baseline text (the reference model's page).
        candidate: Text being compared.

    Returns:
        Tuple of (HTML string, added word count, removed word count). Text only
        in the candidate is wrapped in ``<ins>``; text only in the reference is
        shown struck through in ``<del>`` so omissions stay visible.
    """
    ref_tokens = _TOKEN_RE.findall(reference)
    cand_tokens = _TOKEN_RE.findall(candidate)
    matcher = SequenceMatcher(
        None,
        [_compare_key(t) for t in ref_tokens],
        [_compare_key(t) for t in cand_tokens],
        autojunk=False,
    )

    parts: List[str] = []
    added = removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_chunk = html.escape("".join(ref_tokens[i1:i2]))
        cand_chunk = html.escape("".join(cand_tokens[j1:j2]))
        if tag == "equal":
            parts.append(cand_chunk)
            continue
        if tag in ("replace", "delete") and ref_chunk.strip():
            parts.append(f"<del>{ref_chunk}</del>")
            removed += sum(1 for t in ref_tokens[i1:i2] if _compare_key(t).strip())
        if tag in ("replace", "insert") and cand_chunk.strip():
            parts.append(f"<ins>{cand_chunk}</ins>")
            added += sum(1 for t in cand_tokens[j1:j2] if _compare_key(t).strip())
    return "".join(parts), added, removed


def order_results(
    results: Sequence["BenchmarkModelResult"],
    reference_label: Optional[str] = None,
) -> List["BenchmarkModelResult"]:
    """Order results for display: reference model first, then by cost.

    The reference is the yardstick every other column is diffed against, so it
    leads. Remaining models follow cheapest-first.

    Args:
        results: Benchmark results to order.
        reference_label: Label of the model to lead with. Defaults to the first
            successful result.

    Returns:
        Ordered list containing only successful (``status == "ok"``) results.
    """
    usable = [r for r in results if r.status == "ok"]
    if not usable:
        return []

    reference = None
    if reference_label:
        reference = next(
            (r for r in usable if r.spec.label == reference_label), None
        )
    if reference is None:
        reference = usable[0]

    rest = sorted(
        (r for r in usable if r is not reference), key=lambda r: r.cost_usd
    )
    return [reference] + rest


_CSS = """
:root { color-scheme: light dark; --bg:#ffffff; --fg:#1a1a1a; --muted:#666;
  --line:#e2e2e2; --panel:#f7f7f8; --ins-bg:#d7f5dd; --ins-fg:#0b5b21;
  --del-bg:#ffdce0; --del-fg:#8a1c26; }
@media (prefers-color-scheme: dark) { :root { --bg:#0f1115; --fg:#e6e6e6;
  --muted:#9aa0a6; --line:#2a2e35; --panel:#161a20; --ins-bg:#123a1e;
  --ins-fg:#7ee79b; --del-bg:#3d151b; --del-fg:#ff9aa5; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.55
  -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
header { padding:20px 24px; border-bottom:1px solid var(--line); }
h1 { margin:0 0 4px; font-size:19px; }
.sub { color:var(--muted); font-size:13px; }
.legend { margin-top:10px; font-size:12px; color:var(--muted); }
.legend ins, .legend del { padding:1px 5px; border-radius:3px; }
table.summary { border-collapse:collapse; margin-top:14px; font-size:13px; }
table.summary th, table.summary td { border:1px solid var(--line);
  padding:5px 10px; text-align:left; white-space:nowrap; }
table.summary th { background:var(--panel); }
.ref-badge { background:var(--ins-bg); color:var(--ins-fg); border-radius:3px;
  padding:1px 6px; font-size:11px; margin-left:6px; }
section.page { border-bottom:1px solid var(--line); padding:22px 24px; }
.page-title { font-size:13px; color:var(--muted); margin-bottom:12px;
  text-transform:uppercase; letter-spacing:.06em; }
.row { display:flex; align-items:flex-start; }
/* Page column width is a single global, so dragging any page resizes them all. */
.pdf { flex:0 0 var(--pdf-w, 420px); position:sticky; top:16px; min-width:0; }
.pdf img { width:100%; height:auto; display:block; border:1px solid var(--line);
  border-radius:6px; background:#fff; }
.pdf .nopreview { color:var(--muted); font-size:12px; padding:20px;
  border:1px dashed var(--line); border-radius:6px; text-align:center; }
.dragger { flex:0 0 11px; align-self:stretch; cursor:col-resize;
  position:relative; }
.dragger::before { content:""; position:absolute; top:0; bottom:0; left:5px;
  width:1px; background:var(--line); }
.dragger:hover::before, .dragger.active::before { background:#6b8afd; width:2px;
  left:4px; }
body.dragging { cursor:col-resize; user-select:none; }
.cols { flex:1 1 auto; display:flex; gap:14px; overflow-x:auto;
  padding-bottom:6px; min-width:0; }
.col { flex:1 0 340px; min-width:340px; border:1px solid var(--line);
  border-radius:6px; background:var(--panel); }
.col h3 { margin:0; padding:8px 11px; font-size:12px; border-bottom:1px solid
  var(--line); display:flex; justify-content:space-between; gap:8px;
  align-items:center; }
.col .stat { color:var(--muted); font-weight:400; font-size:11px; }
.col pre { margin:0; padding:11px; white-space:pre-wrap; word-wrap:break-word;
  font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; max-height:70vh;
  overflow:auto; }
ins { background:var(--ins-bg); color:var(--ins-fg); text-decoration:none;
  border-radius:2px; }
del { background:var(--del-bg); color:var(--del-fg); border-radius:2px; }
@media (max-width:1000px) { .row { flex-direction:column; }
  .pdf { position:static; flex-basis:auto; width:100%; max-width:520px; }
  .dragger { display:none; } }
"""

# Drag any page column's handle to resize the page preview across every page.
# Width is written to one CSS variable on :root, so all sections stay in step.
_JS = """
(function () {
  var MIN = 180, root = document.documentElement, active = null;

  function widthFrom(event, dragger) {
    var left = dragger.parentElement.getBoundingClientRect().left;
    var max = Math.max(MIN, window.innerWidth - 320);
    return Math.min(Math.max(event.clientX - left, MIN), max);
  }

  document.addEventListener('mousedown', function (event) {
    var dragger = event.target.closest('.dragger');
    if (!dragger) return;
    active = dragger;
    dragger.classList.add('active');
    document.body.classList.add('dragging');
    event.preventDefault();
  });

  document.addEventListener('mousemove', function (event) {
    if (!active) return;
    root.style.setProperty('--pdf-w', widthFrom(event, active) + 'px');
  });

  document.addEventListener('mouseup', function () {
    if (!active) return;
    active.classList.remove('active');
    document.body.classList.remove('dragging');
    active = null;
  });

  // Double-click the handle to fit the page column to its natural image width.
  document.addEventListener('dblclick', function (event) {
    if (!event.target.closest('.dragger')) return;
    var img = document.querySelector('.pdf img');
    if (img && img.naturalWidth) {
      root.style.setProperty('--pdf-w', Math.min(
        img.naturalWidth, window.innerWidth - 320) + 'px');
    }
  });
})();
"""


def _summary_table(ordered: List["BenchmarkModelResult"]) -> str:
    """Render the top-of-report summary table.

    Args:
        ordered: Display-ordered results, reference first.

    Returns:
        HTML for the summary table.
    """
    rows = []
    for index, result in enumerate(ordered):
        badge = '<span class="ref-badge">reference</span>' if index == 0 else ""
        rows.append(
            f"<tr><td>{html.escape(result.spec.label)}{badge}</td>"
            f"<td>{result.elapsed_seconds:.1f}s</td>"
            f"<td>{result.page_count}</td>"
            f"<td>{result.output_chars:,}</td>"
            f"<td>${result.cost_usd:.4f}</td></tr>"
        )
    return (
        '<table class="summary"><tr><th>Model</th><th>Time</th><th>Pages</th>'
        "<th>Chars</th><th>Cost</th></tr>" + "".join(rows) + "</table>"
    )


def write_benchmark_html(
    results: Sequence["BenchmarkModelResult"],
    pdf_path: str,
    output_path: str,
    reference_label: Optional[str] = None,
) -> Optional[str]:
    """Write a scrollable page-by-page model comparison to a single HTML file.

    Each page section shows the rendered PDF page beside every model's markdown
    for that page. Non-reference columns are word-diffed against the reference:
    ``<ins>`` (green) is text only that model produced, ``<del>`` (red) is
    reference text it missed.

    Args:
        results: Benchmark results to compare.
        pdf_path: The benchmarked PDF, rendered for page previews.
        output_path: Destination ``.html`` path.
        reference_label: Model label to use as the diff baseline. Defaults to
            the first successful result.

    Returns:
        The written path, or None if no successful results were available.
    """
    ordered = order_results(results, reference_label)
    if not ordered:
        return None

    per_model_pages = {r.spec.label: split_markdown_pages(r.markdown) for r in ordered}
    reference = ordered[0]
    ref_pages = per_model_pages[reference.spec.label]

    page_numbers = sorted(
        {num for pages in per_model_pages.values() for num in pages}
    )
    images = render_pdf_page_images(pdf_path)

    sections: List[str] = []
    for page_num in page_numbers:
        if images.get(page_num):
            preview = f'<img src="{images[page_num]}" alt="Page {page_num}">'
        else:
            preview = '<div class="nopreview">No page preview</div>'

        columns: List[str] = []
        for index, result in enumerate(ordered):
            text = per_model_pages[result.spec.label].get(page_num, "")
            if index == 0:
                body = html.escape(text)
                stat = f"{len(text.split()):,} words · baseline"
            else:
                body, added, removed = _diff_spans(
                    ref_pages.get(page_num, ""), text
                )
                stat = f"+{added} / -{removed} words vs reference"
            badge = (
                '<span class="ref-badge">reference</span>' if index == 0 else ""
            )
            columns.append(
                f'<div class="col"><h3><span>{html.escape(result.spec.label)}'
                f'{badge}</span><span class="stat">{stat}</span></h3>'
                f"<pre>{body or '<em>(empty)</em>'}</pre></div>"
            )

        sections.append(
            f'<section class="page"><div class="page-title">Page {page_num}'
            f'</div><div class="row"><div class="pdf">{preview}</div>'
            '<div class="dragger" title="Drag to resize the page; '
            'double-click to fit"></div>'
            f'<div class="cols">{"".join(columns)}</div></div></section>'
        )

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Model comparison — {html.escape(os.path.basename(pdf_path))}"
        f"</title><style>{_CSS}</style></head><body>"
        "<header><h1>Benchmark comparison — "
        f"{html.escape(os.path.basename(pdf_path))}</h1>"
        f'<div class="sub">{len(ordered)} models · {len(page_numbers)} pages · '
        f"diffed against <strong>{html.escape(reference.spec.label)}</strong>"
        "</div>"
        '<div class="legend"><ins>green</ins> = only this model produced it · '
        "<del>red</del> = reference text this model missed · "
        "drag the divider beside a page to resize it (double-click to fit)"
        "</div>"
        f"{_summary_table(ordered)}</header>{''.join(sections)}"
        f"<script>{_JS}</script></body></html>"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(doc)
    return output_path
