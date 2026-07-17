"""
Cross-provider model benchmark for PDF-to-Markdown extraction.

Runs the same PDF through the top vision model from each of several providers
and reports wall-clock time, output size, token usage, and cost side by side.
By default it compares:

- OpenAI:    ``gpt-5.5``            (cloud API, requires OPENAI_API_KEY)
- Anthropic: ``claude-opus-4.6``    (cloud API, requires ANTHROPIC_API_KEY)
- Local:     ``qwen2.5vl:7b``        (offline via a local Ollama server)

Models run concurrently (one worker per model), and each model's own
``VisionExtractor`` processes its pages in parallel too, so the benchmark does
as much work as possible at once. Progress for every model is streamed to the
console with a per-model ``[label]`` tag. The local model may need to download
its weights on first run; that pull is streamed as progress as well.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

from local_ocr import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_OLLAMA_GENERATE_URL,
    LocalOllamaProvider,
    ensure_ollama_model,
    is_ollama_running,
)

# Thread-safe console output. Benchmarks run models concurrently, so all
# progress prints go through this lock to keep lines from interleaving.
_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    """Print a benchmark progress line under a shared lock."""
    with _PRINT_LOCK:
        print(message, flush=True)


@dataclass
class BenchmarkModelSpec:
    """A single model to benchmark.

    Attributes:
        label: Human-readable name shown in progress lines and the report.
        kind: ``"api"`` for cloud providers or ``"local"`` for Ollama.
        model: models.json key (api) or Ollama model tag (local).
        prefer_openrouter: Route cloud calls via OpenRouter when available.
        ollama_url: Ollama generate endpoint for local models.
        num_ctx: Context window in tokens for local models.
        num_predict: Output token budget per page for local models.
    """

    label: str
    kind: str
    model: str
    prefer_openrouter: bool = False
    ollama_url: str = DEFAULT_OLLAMA_GENERATE_URL
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT


@dataclass
class BenchmarkModelResult:
    """Outcome of benchmarking one model.

    Attributes:
        spec: The spec that was run.
        status: ``"ok"``, ``"skipped"``, or ``"error"``.
        detail: Reason for a skip or error (empty when ``ok``).
        elapsed_seconds: Wall-clock extraction time.
        page_count: Number of pages processed.
        output_chars: Length of the produced markdown.
        input_tokens: Total input tokens (0 for local).
        output_tokens: Total output tokens (0 for local).
        cost_usd: Total cost in USD (0 for local).
        output_path: Where the markdown was saved, if applicable.
        markdown: The produced markdown, retained so the HTML comparison
            report can diff models page by page.
    """

    spec: BenchmarkModelSpec
    status: str
    detail: str = ""
    elapsed_seconds: float = 0.0
    page_count: int = 0
    output_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    output_path: Optional[str] = None
    markdown: str = ""


def default_benchmark_specs(
    local_model: str = DEFAULT_LOCAL_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_GENERATE_URL,
    prefer_openrouter: bool = False,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> List[BenchmarkModelSpec]:
    """Return the default top-model-per-provider benchmark set.

    Args:
        local_model: Ollama model tag for the local entry.
        ollama_url: Ollama generate endpoint for the local entry.
        prefer_openrouter: Whether cloud entries should prefer OpenRouter.
        num_ctx: Context window in tokens for the local entry.
        num_predict: Output token budget per page for the local entry.

    Returns:
        List of specs for OpenAI, Anthropic, and Local top models.
    """
    return [
        BenchmarkModelSpec(
            label="OpenAI (gpt-5.5)",
            kind="api",
            model="gpt-5.5",
            prefer_openrouter=prefer_openrouter,
        ),
        BenchmarkModelSpec(
            label="Anthropic (claude-opus-4.6)",
            kind="api",
            model="claude-opus-4.6",
            prefer_openrouter=prefer_openrouter,
        ),
        BenchmarkModelSpec(
            label=f"Local ({local_model})",
            kind="local",
            model=local_model,
            ollama_url=ollama_url,
            num_ctx=num_ctx,
            num_predict=num_predict,
        ),
    ]


def parse_benchmark_specs(
    spec_arg: str,
    local_model: str = DEFAULT_LOCAL_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_GENERATE_URL,
    prefer_openrouter: bool = False,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> List[BenchmarkModelSpec]:
    """Parse a comma-separated ``--benchmark-models`` override into specs.

    Each entry is either ``local`` / ``local:<model>`` for an Ollama model, or
    a models.json key for a cloud model.

    Args:
        spec_arg: Comma-separated list of model identifiers.
        local_model: Default Ollama model when an entry is bare ``local``.
        ollama_url: Ollama generate endpoint for local entries.
        prefer_openrouter: Whether cloud entries should prefer OpenRouter.
        num_ctx: Context window in tokens for local entries.
        num_predict: Output token budget per page for local entries.

    Returns:
        List of parsed specs, preserving input order.

    Raises:
        ValueError: If the argument is empty.
    """
    entries = [item.strip() for item in spec_arg.split(",") if item.strip()]
    if not entries:
        raise ValueError("--benchmark-models must list at least one model.")

    specs: List[BenchmarkModelSpec] = []
    for entry in entries:
        lowered = entry.lower()
        if lowered == "local":
            specs.append(
                BenchmarkModelSpec(
                    label=f"Local ({local_model})",
                    kind="local",
                    model=local_model,
                    ollama_url=ollama_url,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
            )
        elif lowered.startswith("local:"):
            model = entry.split(":", 1)[1].strip() or local_model
            specs.append(
                BenchmarkModelSpec(
                    label=f"Local ({model})",
                    kind="local",
                    model=model,
                    ollama_url=ollama_url,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
            )
        else:
            specs.append(
                BenchmarkModelSpec(
                    label=entry,
                    kind="api",
                    model=entry,
                    prefer_openrouter=prefer_openrouter,
                )
            )
    return specs


def _resolve_benchmark_pdf(pdf_path: Optional[str]) -> str:
    """Resolve the PDF to benchmark, generating a fixture if none is given.

    Args:
        pdf_path: Optional user-provided PDF path.

    Returns:
        Path to a PDF to benchmark.

    Raises:
        ValueError: If a provided path does not exist or is not a file.
    """
    if pdf_path:
        if not os.path.isfile(pdf_path):
            raise ValueError(
                f"Benchmark PDF path does not exist or is not a file: {pdf_path}"
            )
        return pdf_path

    from tests.create_test_pdf import ensure_default_fixture_pdfs

    return ensure_default_fixture_pdfs()[0]


def _resolve_api_routing(spec: BenchmarkModelSpec):
    """Determine whether a cloud spec can run and how it should be routed.

    Mirrors what ``create_provider`` will actually do, so the precheck never
    disagrees with the build step (which would otherwise fall through to the
    interactive fallback menu and hang a benchmark thread).

    Args:
        spec: Cloud model spec to resolve.

    Returns:
        Tuple ``(skip_reason, prefer_openrouter)`` where ``skip_reason`` is
        None when the spec is runnable.
    """
    from llm_providers import get_available_providers, load_models_config

    config = load_models_config()
    if spec.model not in config:
        return f"'{spec.model}' is not defined in models.json", False

    provider_name = config[spec.model].get("provider", "unknown")
    has_openrouter_id = bool(config[spec.model].get("openrouter_id"))
    available = get_available_providers()

    openrouter_ok = available.get("openrouter") and has_openrouter_id
    direct_ok = available.get(provider_name)

    if spec.prefer_openrouter and openrouter_ok:
        return None, True
    if direct_ok:
        return None, False
    if openrouter_ok:
        # OpenRouter is the only usable path (e.g. only OPENROUTER_API_KEY set).
        return None, True

    key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY/GEMINI_API_KEY",
    }.get(provider_name, f"{provider_name} API key")
    return f"no {key_env} or OPENROUTER_API_KEY set", False


def _precheck_api_spec(spec: BenchmarkModelSpec) -> Optional[str]:
    """Return a skip reason for a cloud spec, or None if it can run.

    Args:
        spec: Cloud model spec to validate.

    Returns:
        A human-readable skip reason, or None when the spec is runnable.
    """
    reason, _ = _resolve_api_routing(spec)
    return reason


def _precheck_local_spec(spec: BenchmarkModelSpec) -> Optional[str]:
    """Return a skip reason for a local spec, or None if it can run.

    Args:
        spec: Local model spec to validate.

    Returns:
        A human-readable skip reason, or None when the spec is runnable.
    """
    if not is_ollama_running(spec.ollama_url):
        return f"no Ollama server reachable at {spec.ollama_url}"
    return None


def _build_extractor(spec: BenchmarkModelSpec, max_parallel_pages: int):
    """Build a configured VisionExtractor for a benchmark spec.

    Args:
        spec: Model spec to build an extractor for.
        max_parallel_pages: Page-level parallelism for the extractor.

    Returns:
        A ready-to-run VisionExtractor instance.
    """
    from extractors import VisionExtractor

    if spec.kind == "local":
        _log(f"[{spec.label}] ensuring model is downloaded...")
        ensure_ollama_model(spec.model, spec.ollama_url, verbose=True)
        provider = LocalOllamaProvider(
            model_name=spec.model,
            base_url=spec.ollama_url,
            num_ctx=spec.num_ctx,
            num_predict=spec.num_predict,
        )
        model_id = spec.model
    else:
        from llm_providers import create_provider

        _, prefer_openrouter = _resolve_api_routing(spec)
        provider, model_id = create_provider(spec.model, prefer_openrouter)

    return VisionExtractor(
        provider=provider,
        model_id=model_id,
        mode="vt",
        max_parallel_pages=max_parallel_pages,
    )


def _run_single_benchmark(
    spec: BenchmarkModelSpec,
    pdf_path: str,
    output_dir: Optional[str],
    max_parallel_pages: int,
) -> BenchmarkModelResult:
    """Run one model's extraction and capture its stats.

    Args:
        spec: Model spec to run.
        pdf_path: Path to the PDF being benchmarked.
        output_dir: Directory to save per-model markdown (and page images).
        max_parallel_pages: Page-level parallelism for the extractor.

    Returns:
        A populated BenchmarkModelResult (status ``ok``, ``skipped``, or
        ``error``).
    """
    skip_reason = (
        _precheck_local_spec(spec)
        if spec.kind == "local"
        else _precheck_api_spec(spec)
    )
    if skip_reason:
        _log(f"[{spec.label}] SKIPPED — {skip_reason}")
        return BenchmarkModelResult(
            spec=spec, status="skipped", detail=skip_reason
        )

    try:
        extractor = _build_extractor(spec, max_parallel_pages)
    except Exception as error:
        _log(f"[{spec.label}] ERROR building extractor — {error}")
        return BenchmarkModelResult(
            spec=spec, status="error", detail=str(error)
        )

    # Each model renders the PDF into its own isolated cache directory. Models
    # run concurrently against the same PDF, so a shared image cache would race
    # (one model reading a folder another is still writing).
    cache_dir = tempfile.mkdtemp(prefix="pdf_to_md_bench_")

    _log(f"[{spec.label}] starting extraction of {os.path.basename(pdf_path)}...")
    start = time.perf_counter()
    try:
        result = extractor.extract(pdf_path, cache_dir, verbose=False)
    except Exception as error:
        elapsed = time.perf_counter() - start
        _log(f"[{spec.label}] ERROR after {elapsed:.1f}s — {error}")
        return BenchmarkModelResult(
            spec=spec,
            status="error",
            detail=str(error),
            elapsed_seconds=elapsed,
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    elapsed = time.perf_counter() - start

    meta = result.metadata or {}
    output_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.basename(pdf_path).rsplit(".", 1)[0]
        safe_label = "".join(
            ch if ch.isalnum() else "_" for ch in spec.label
        ).strip("_")
        output_path = os.path.join(output_dir, f"{stem}.{safe_label}.md")
        with open(output_path, "w") as handle:
            handle.write(result.markdown)

    _log(
        f"[{spec.label}] DONE in {elapsed:.1f}s — "
        f"{result.page_count} pages, {len(result.markdown):,} chars, "
        f"cost ${meta.get('total_cost_usd', 0.0):.4f}"
    )

    return BenchmarkModelResult(
        spec=spec,
        status="ok",
        elapsed_seconds=elapsed,
        page_count=result.page_count,
        output_chars=len(result.markdown),
        input_tokens=meta.get("total_input_tokens", 0),
        output_tokens=meta.get("total_output_tokens", 0),
        cost_usd=meta.get("total_cost_usd", 0.0),
        output_path=output_path,
        markdown=result.markdown,
    )


def benchmark_models(
    specs: List[BenchmarkModelSpec],
    pdf_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_parallel_pages: int = 10,
) -> List[BenchmarkModelResult]:
    """Benchmark several models against the same PDF.

    Cloud models run concurrently, since their latency is the remote API's.
    Local models run one at a time: they share a single GPU, so overlapping
    them measures contention rather than the model, inflating every local time
    (observed: 13x) while telling you nothing useful.

    Args:
        specs: Models to benchmark.
        pdf_path: PDF to run; a fixture is generated when None.
        output_dir: Optional directory to save each model's markdown output.
        max_parallel_pages: Page-level parallelism within each model.

    Returns:
        Results in the same order as ``specs``.

    Raises:
        ValueError: If ``specs`` is empty or the PDF path is invalid.
    """
    if not specs:
        raise ValueError("At least one model spec is required to benchmark.")

    resolved_pdf = _resolve_benchmark_pdf(pdf_path)
    local_count = sum(1 for s in specs if s.kind == "local")

    _log("=" * 70)
    _log("Cross-Provider Model Benchmark")
    _log("=" * 70)
    _log(f"Input PDF: {resolved_pdf}")
    _log(f"Models ({len(specs)}):")
    for spec in specs:
        _log(f"  - {spec.label} [{spec.kind}]")
    if local_count > 1:
        _log(
            f"Note: {local_count} local models share one GPU and will run "
            "one at a time so their timings stay comparable."
        )
    _log("=" * 70)

    results: List[Optional[BenchmarkModelResult]] = [None] * len(specs)

    def run(index: int) -> None:
        """Benchmark one spec and record its result by original position."""
        results[index] = _run_single_benchmark(
            specs[index], resolved_pdf, output_dir, max_parallel_pages
        )

    api_indices = [i for i, s in enumerate(specs) if s.kind != "local"]
    local_indices = [i for i, s in enumerate(specs) if s.kind == "local"]

    # Cloud models overlap freely; the local queue is drained on one worker so
    # only one model occupies the GPU at a time.
    with ThreadPoolExecutor(max_workers=max(1, len(api_indices) + 1)) as executor:
        futures = [executor.submit(run, i) for i in api_indices]
        if local_indices:

            def run_locals_in_sequence() -> None:
                """Run every local model back to back on a single worker."""
                for index in local_indices:
                    run(index)

            futures.append(executor.submit(run_locals_in_sequence))
        for future in as_completed(futures):
            future.result()

    return [r for r in results if r is not None]


def print_model_benchmark_report(results: List[BenchmarkModelResult]) -> None:
    """Print a side-by-side comparison table for benchmark results.

    Args:
        results: Benchmark results to summarize.
    """
    print()
    print("=" * 88)
    print("Model Benchmark Results")
    print("=" * 88)
    header = (
        f"{'Model':<32} {'Status':<8} {'Time(s)':>8} {'Pages':>6} "
        f"{'Chars':>9} {'Cost($)':>9}"
    )
    print(header)
    print("-" * 88)

    for result in results:
        if result.status == "ok":
            print(
                f"{result.spec.label:<32} {'ok':<8} "
                f"{result.elapsed_seconds:>8.1f} {result.page_count:>6} "
                f"{result.output_chars:>9,} {result.cost_usd:>9.4f}"
            )
        else:
            print(
                f"{result.spec.label:<32} {result.status:<8} "
                f"{'-':>8} {'-':>6} {'-':>9} {'-':>9}"
            )

    # Notes for anything that did not complete.
    notes = [r for r in results if r.status != "ok"]
    if notes:
        print("-" * 88)
        for result in notes:
            print(f"  {result.spec.label}: {result.status} — {result.detail}")

    completed = [r for r in results if r.status == "ok"]
    if completed:
        fastest = min(completed, key=lambda r: r.elapsed_seconds)
        cheapest = min(completed, key=lambda r: r.cost_usd)
        print("-" * 88)
        print(
            f"Fastest: {fastest.spec.label} "
            f"({fastest.elapsed_seconds:.1f}s)"
        )
        print(
            f"Cheapest: {cheapest.spec.label} "
            f"(${cheapest.cost_usd:.4f})"
        )
        saved = [r for r in completed if r.output_path]
        if saved:
            print("Saved markdown outputs:")
            for result in saved:
                print(f"  - {result.spec.label}: {result.output_path}")
    print("=" * 88)
