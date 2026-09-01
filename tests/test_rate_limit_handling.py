"""
Tests for rate-limit handling: 429-aware retry, the recovery sweep, and
provider-declared concurrency.

Covers issue #14, where a 199-page Fireworks run dropped 46 pages as [ERROR]
placeholders because the ~10s retry envelope expired inside a per-minute quota
window.
"""
import os
import sys
from types import SimpleNamespace
from unittest import mock

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors import (  # noqa: E402
    RECOVERY_CONCURRENCY,
    VisionExtractor,
    _stop_for_exception,
    _wait_for_exception,
    is_rate_limit_error,
    is_retryable_error,
    retry_after_seconds,
)


class RateLimitError(Exception):
    """
    Stand-in for the SDKs' rate-limit exception.

    openai, anthropic, and google-genai each define their own class; detection
    keys off the shared class name and a 429 status, so this double is faithful
    without importing any of them.
    """

    def __init__(self, message: str = "rate limited", retry_after=None):
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(status_code=429, headers=headers)


class _Outcome:
    """Minimal tenacity outcome carrying a raised exception."""

    def __init__(self, exc: BaseException):
        self._exc = exc

    def exception(self) -> BaseException:
        return self._exc

    def failed(self) -> bool:
        return True


def _state(exc: BaseException, attempt_number: int = 1, elapsed: float = 0.0):
    """Build a stand-in tenacity retry state for the policy callables."""
    return SimpleNamespace(
        outcome=_Outcome(exc),
        attempt_number=attempt_number,
        seconds_since_start=elapsed,
        idle_for=0.0,
    )


# --- classification -------------------------------------------------------


def test_detects_rate_limit_by_class_name_and_status() -> None:
    """429s are recognized across SDKs without importing any of them."""
    assert is_rate_limit_error(RateLimitError())
    assert is_rate_limit_error(SimpleNamespace(status_code=429))  # type: ignore[arg-type]

    plain = Exception("boom")
    plain.response = SimpleNamespace(status_code=429)
    assert is_rate_limit_error(plain)


def test_non_rate_limit_errors_are_not_misread() -> None:
    """A bad request or local bug must not be treated as throttling."""
    assert not is_rate_limit_error(ValueError("bad page"))

    bad_request = Exception("bad request")
    bad_request.response = SimpleNamespace(status_code=400)
    assert not is_rate_limit_error(bad_request)


def test_retry_after_header_parsed_when_sane() -> None:
    """A numeric Retry-After is honored; junk and absurd values are ignored."""
    assert retry_after_seconds(RateLimitError(retry_after="30")) == 30.0
    assert retry_after_seconds(RateLimitError()) is None
    assert retry_after_seconds(RateLimitError(retry_after="soon")) is None
    assert retry_after_seconds(RateLimitError(retry_after="-5")) is None
    assert retry_after_seconds(RateLimitError(retry_after="99999")) is None
    assert retry_after_seconds(ValueError("no response")) is None


def test_retryable_covers_throttling_and_transport_only() -> None:
    """Transient faults are worth retrying; deterministic ones are not."""
    assert is_retryable_error(RateLimitError())
    assert is_retryable_error(type("APIConnectionError", (Exception,), {})())

    server_error = Exception("upstream died")
    server_error.response = SimpleNamespace(status_code=503)
    assert is_retryable_error(server_error)

    assert not is_retryable_error(ValueError("malformed response"))
    assert not is_retryable_error(AttributeError("'str' object has no usage"))


# --- retry envelope -------------------------------------------------------


def test_rate_limited_call_outlives_the_transient_attempt_cap() -> None:
    """The core issue: a 429 must not be abandoned after three quick tries."""
    throttled = RateLimitError()

    # A deterministic error still gives up at three attempts.
    assert _stop_for_exception(_state(ValueError("nope"), attempt_number=3))

    # A 429 keeps going well past that, into the next quota window.
    assert not _stop_for_exception(_state(throttled, attempt_number=3))
    assert not _stop_for_exception(_state(throttled, attempt_number=8))


def test_rate_limit_retry_is_bounded() -> None:
    """Retrying is generous but not unbounded."""
    throttled = RateLimitError()
    assert _stop_for_exception(_state(throttled, attempt_number=2, elapsed=301))
    assert _stop_for_exception(_state(throttled, attempt_number=12))


def test_retry_after_hint_sets_the_floor_on_the_wait() -> None:
    """When the provider says how long to wait, we wait at least that long."""
    waited = _wait_for_exception(_state(RateLimitError(retry_after="30")))
    assert waited >= 30


def test_transient_wait_stays_short() -> None:
    """Non-throttling failures keep the original tight backoff."""
    for attempt in range(1, 10):
        assert _wait_for_exception(_state(ValueError("x"), attempt)) <= 5


# --- provider-declared concurrency ---------------------------------------


class _Provider:
    """Provider double declaring a safe concurrency."""

    def __init__(self, limit=None):
        if limit is not None:
            self.MAX_SAFE_CONCURRENCY = limit

    def process_vision(self, **kwargs):  # pragma: no cover - never called
        raise AssertionError("not used")


def test_concurrency_defaults_to_provider_limit() -> None:
    """Fireworks-class providers cap their own fan-out."""
    assert VisionExtractor(_Provider(3), "m").max_parallel_pages == 3


def test_concurrency_falls_back_when_provider_declares_nothing() -> None:
    """Providers predating the attribute keep the historical default."""
    assert VisionExtractor(_Provider(), "m").max_parallel_pages == 10


def test_explicit_parallelism_overrides_provider_limit() -> None:
    """An explicit --parallel always wins."""
    assert VisionExtractor(_Provider(3), "m", max_parallel_pages=8).max_parallel_pages == 8


def test_non_integer_declaration_is_ignored() -> None:
    """A test double answering every attribute must not set concurrency."""
    assert VisionExtractor(mock.Mock(), "m").max_parallel_pages == 10


def test_fireworks_declares_reduced_concurrency() -> None:
    """The shipped Fireworks provider carries the lowered limit."""
    from llm_providers import BaseProvider, FireworksProvider

    assert FireworksProvider.MAX_SAFE_CONCURRENCY == 3
    assert BaseProvider.MAX_SAFE_CONCURRENCY == 10
    assert FireworksProvider.MAX_SAFE_CONCURRENCY <= RECOVERY_CONCURRENCY


# --- recovery sweep -------------------------------------------------------


def _extractor_with_pages(page_count: int, provider=None) -> VisionExtractor:
    """Build an extractor whose page data is stubbed to `page_count` pages."""
    extractor = VisionExtractor(
        provider=provider or _Provider(),
        model_id="test-model",
        mode="v",
        max_parallel_pages=10,
    )
    images = [Image.new("RGB", (8, 8), color="white") for _ in range(page_count)]
    prior_texts = [None] * page_count
    extractor._load_page_data = mock.Mock(return_value=(images, prior_texts))
    return extractor


def test_sweep_recovers_pages_that_were_throttled(tmp_path) -> None:
    """A page 429'd on the first pass is recovered instead of placeheld."""
    extractor = _extractor_with_pages(4)
    attempts: dict[int, int] = {}

    def flaky(page_index, image, prior_text, pdf_file_name, total_pages):
        attempts[page_index] = attempts.get(page_index, 0) + 1
        if page_index in (1, 2) and attempts[page_index] == 1:
            raise RateLimitError()
        return page_index, f"page {page_index}"

    extractor._process_single_page = flaky
    result = extractor.extract("deck.pdf", str(tmp_path), verbose=False)

    assert attempts[1] == 2 and attempts[2] == 2
    assert "[ERROR" not in result.markdown
    for index in range(4):
        assert f"page {index}" in result.markdown


def test_sweep_is_skipped_for_deterministic_failures(tmp_path) -> None:
    """A page that fails for a real reason is not re-sent at double cost."""
    extractor = _extractor_with_pages(3)
    attempts: dict[int, int] = {}

    def broken(page_index, image, prior_text, pdf_file_name, total_pages):
        attempts[page_index] = attempts.get(page_index, 0) + 1
        raise ValueError("malformed response")

    extractor._process_single_page = broken
    result = extractor.extract("deck.pdf", str(tmp_path), verbose=False)

    assert all(count == 1 for count in attempts.values())
    assert "[ERROR: Page 1 failed to process]" in result.markdown


def test_sweep_keeps_permanent_failures_alongside_unrecovered(tmp_path) -> None:
    """Mixed failures: the throttled page retries, the broken one does not."""
    extractor = _extractor_with_pages(3)
    attempts: dict[int, int] = {}

    def mixed(page_index, image, prior_text, pdf_file_name, total_pages):
        attempts[page_index] = attempts.get(page_index, 0) + 1
        if page_index == 0:
            raise ValueError("malformed response")
        if page_index == 1 and attempts[page_index] == 1:
            raise RateLimitError()
        return page_index, f"page {page_index}"

    extractor._process_single_page = mixed
    result = extractor.extract("deck.pdf", str(tmp_path), verbose=False)

    assert attempts[0] == 1          # deterministic, not retried
    assert attempts[1] == 2          # throttled, retried and recovered
    assert "[ERROR: Page 1 failed to process]" in result.markdown
    assert "page 1" in result.markdown


def test_sweep_runs_at_reduced_concurrency(tmp_path) -> None:
    """The retry asks for less throughput than the pass that was throttled."""
    extractor = _extractor_with_pages(6)
    seen_workers = []
    original = extractor._process_pages_parallel

    def record(*args, **kwargs):
        seen_workers.append(kwargs.get("max_workers"))
        return original(*args, **kwargs)

    attempts: dict[int, int] = {}

    def flaky(page_index, image, prior_text, pdf_file_name, total_pages):
        attempts[page_index] = attempts.get(page_index, 0) + 1
        if attempts[page_index] == 1:
            raise RateLimitError()
        return page_index, f"page {page_index}"

    extractor._process_single_page = flaky
    extractor._process_pages_parallel = record
    extractor.extract("deck.pdf", str(tmp_path), verbose=False)

    assert seen_workers[0] is None                      # first pass: full fan-out
    assert seen_workers[1] == RECOVERY_CONCURRENCY      # sweep: throttled back
