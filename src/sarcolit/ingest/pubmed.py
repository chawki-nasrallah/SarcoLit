"""Fetch PubMed records via NCBI E-utilities for the SarcoLit corpus.

Step 3 of ``v0.1-corpus``. This module does exactly one thing: reliably pull the
raw PubMed XML for the locked search query onto disk. Parsing, cleaning, and
dedup are Step 4 and run *against these saved files*, so re-parsing never
re-hits the network — and a flaky run can be resumed instead of restarted.

Flow (the standard E-utilities two-step):

1. ``esearch`` with ``usehistory=y`` runs the query once and parks the result
   set on NCBI's history server, handing back a ``WebEnv`` + ``query_key`` and
   the total ``count``. We never download the full PMID list ourselves.
2. ``efetch`` pages through that parked result set in batches of ``batch_size``,
   writing each page's raw XML straight to ``data/raw/pubmed/batch_NNNN.xml``.

Run it:  ``uv run --env-file .env python -m sarcolit.ingest.pubmed``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm import tqdm

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "sarcolit"

# The locked v0.1 search strategy. Single source of truth — kept in sync with
# docs/releases/v0.1-corpus.md. Whitespace/newlines are fine: E-utilities
# tolerates them in the `term` parameter.
DEFAULT_QUERY = """
(
    (
        "Sarcopenia/diagnosis"[Majr]
        OR "Sarcopenia/classification"[Majr]
        OR "Sarcopenia/physiopathology"[Majr]
        OR "Sarcopenia/pathology"[Majr]
    )
    OR
    (
        ("Muscular Atrophy/diagnosis"[Majr] OR "Muscular Atrophy/physiopathology"[Majr])
        AND ("Aging"[MeSH] OR "Aged"[MeSH])
    )
    OR
    (
        "Electromyography"[MeSH]
        AND (
            "Sarcopenia"[MeSH]
            OR "Muscular Atrophy"[MeSH]
            OR "Muscle Weakness"[MeSH]
            OR "Aging"[Majr]
        )
    )
)
AND humans[MeSH]
AND english[la]
AND hasabstract
AND 2000:2026[dp]
NOT ("Mice"[MeSH] OR "Rats"[MeSH] OR "Disease Models, Animal"[MeSH])
"""


@dataclass
class FetchConfig:
    """Everything the fetch needs. Defaults are production values for v0.1."""

    query: str = DEFAULT_QUERY
    out_dir: Path = Path("data/raw/pubmed")
    batch_size: int = 200  # NCBI recommends <=500 for efetch; 200 keeps files small.
    email: str | None = None
    api_key: str | None = None
    tool: str = TOOL_NAME

    @classmethod
    def from_env(cls, **overrides: object) -> FetchConfig:
        """Build a config, reading NCBI credentials from the environment.

        ``NCBI_EMAIL`` and ``NCBI_API_KEY`` come from ``.env`` (loaded via
        ``uv run --env-file .env``). Both are optional to NCBI, but an email is
        polite etiquette and the key lifts the rate limit from 3 to 10 req/s.
        """
        env = {
            "email": os.environ.get("NCBI_EMAIL") or None,
            "api_key": os.environ.get("NCBI_API_KEY") or None,
        }
        return cls(**{**env, **overrides})  # type: ignore[arg-type]


@dataclass
class SearchResult:
    """The handle to a result set parked on NCBI's history server."""

    count: int
    webenv: str
    query_key: str


class _Throttle:
    """Minimum-interval pacing so we stay under NCBI's per-second cap.

    NCBI allows 3 req/s without an API key and 10 req/s with one. We block
    just long enough before each request to honour the chosen rate.
    """

    def __init__(self, max_per_second: float) -> None:
        self._min_interval = 1.0 / max_per_second
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


def _is_transient(exc: BaseException) -> bool:
    """Retry network blips and server-side errors, but not a bad query (4xx)."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(
    client: httpx.Client, throttle: _Throttle, url: str, params: dict[str, object]
) -> httpx.Response:
    """One throttled, retried GET against an E-utilities endpoint."""
    throttle.wait()
    resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp


def _base_params(cfg: FetchConfig) -> dict[str, object]:
    params: dict[str, object] = {"db": "pubmed", "tool": cfg.tool}
    if cfg.email:
        params["email"] = cfg.email
    if cfg.api_key:
        params["api_key"] = cfg.api_key
    return params


def esearch(client: httpx.Client, throttle: _Throttle, cfg: FetchConfig) -> SearchResult:
    """Run the query once and park the results on the history server."""
    params = _base_params(cfg) | {
        "term": cfg.query,
        "usehistory": "y",
        "retmax": 0,  # we only want the handle + count, not the PMID list
        "retmode": "json",
    }
    resp = _get(client, throttle, f"{EUTILS_BASE}/esearch.fcgi", params)
    result = resp.json()["esearchresult"]
    return SearchResult(
        count=int(result["count"]),
        webenv=result["webenv"],
        query_key=result["querykey"],
    )


def _efetch_batch(
    client: httpx.Client,
    throttle: _Throttle,
    cfg: FetchConfig,
    search: SearchResult,
    retstart: int,
) -> str:
    """Fetch one page of records from the parked result set as raw XML."""
    params = _base_params(cfg) | {
        "WebEnv": search.webenv,
        "query_key": search.query_key,
        "retstart": retstart,
        "retmax": cfg.batch_size,
        "retmode": "xml",
    }
    resp = _get(client, throttle, f"{EUTILS_BASE}/efetch.fcgi", params)
    return resp.text


def _query_hash(query: str) -> str:
    """Stable short hash of the (whitespace-normalised) query, for the manifest."""
    normalised = " ".join(query.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def fetch(cfg: FetchConfig | None = None) -> Path:
    """Fetch the full corpus to disk and write a reproducibility manifest.

    Resumable: any ``batch_NNNN.xml`` already on disk is skipped, so re-running
    after a network failure only fetches the missing pages. Returns the path to
    the manifest.
    """
    cfg = cfg or FetchConfig.from_env()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    throttle = _Throttle(10.0 if cfg.api_key else 3.0)

    if not cfg.email:
        print(
            "warning: NCBI_EMAIL not set — proceeding without it. NCBI asks for "
            "an email so they can contact you before throttling. Set it in .env.",
            file=sys.stderr,
        )

    with httpx.Client(timeout=60.0, headers={"User-Agent": f"{cfg.tool}/0.1"}) as client:
        search = esearch(client, throttle, cfg)
        print(f"esearch: {search.count} records match the locked query.")

        starts = list(range(0, search.count, cfg.batch_size))
        for i, retstart in enumerate(tqdm(starts, desc="efetch", unit="batch")):
            batch_path = cfg.out_dir / f"batch_{i:04d}.xml"
            if batch_path.exists():
                continue  # resume: already fetched on a previous run
            xml = _efetch_batch(client, throttle, cfg, search, retstart)
            batch_path.write_text(xml, encoding="utf-8")

    manifest_path = cfg.out_dir / "manifest.json"
    manifest = {
        "query": " ".join(cfg.query.split()),
        "query_hash": _query_hash(cfg.query),
        "count": search.count,
        "batch_size": cfg.batch_size,
        "n_batches": len(starts),
        "fetched_at": datetime.now(UTC).isoformat(),
        "eutils_tool": cfg.tool,
        "had_api_key": bool(cfg.api_key),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"done: {len(starts)} batches in {cfg.out_dir}/ — manifest at {manifest_path}")
    return manifest_path


def main() -> None:
    fetch()


if __name__ == "__main__":
    main()
