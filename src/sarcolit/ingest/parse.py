"""Parse, clean, and deduplicate the raw PubMed XML into a versioned corpus.

Step 4 of ``v0.1-corpus``. Runs entirely against the files Step 3
(:mod:`sarcolit.ingest.pubmed`) wrote to ``data/raw/pubmed/`` — no network. It
walks every ``batch_NNNN.xml``, extracts the eight fields the release note pins
(PMID, title, abstract, journal, year, authors, MeSH terms, DOI), drops
duplicates and records with no usable abstract, and writes a deterministic
JSONL corpus plus a dataset card to ``data/sarcolit-corpus-v0.1/``.

Why JSONL: zero extra dependencies, human-readable / git-diffable, and still
loads natively with ``datasets.load_dataset("json", ...)`` when v0.2 needs it.

Why these handling choices (driven by what the real XML looks like):

* Structured abstracts (``<AbstractText Label="METHODS">`` ...) are flattened
  into one plain-text paragraph — the section labels are dropped.
* Title/abstract text is pulled with ``itertext()`` so inline markup
  (``<i>``, ``<sub>``) collapses to its text instead of being lost.
* Year falls back from ``<Year>`` to a 4-digit scan of ``<MedlineDate>``.
* DOI is looked for in ``<ELocationID>`` first, then ``<ArticleIdList>``.

Run it:  ``uv run python -m sarcolit.ingest.parse``
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

RAW_DIR = Path("data/raw/pubmed")
CORPUS_DIR = Path("data/sarcolit-corpus-v0.1")
CORPUS_VERSION = "v0.1"


@dataclass
class Record:
    """One cleaned PubMed paper — exactly the fields the release note retains."""

    pmid: str
    title: str
    abstract: str
    journal: str
    year: int | None
    authors: list[str]
    mesh_terms: list[str]
    doi: str | None


def _clean(text: str) -> str:
    """Collapse runs of whitespace/newlines into single spaces and strip."""
    return " ".join(text.split())


def _itertext(el: ET.Element | None) -> str:
    """All text inside an element, flattening any inline markup, cleaned."""
    if el is None:
        return ""
    return _clean("".join(el.itertext()))


def _extract_abstract(article: ET.Element) -> str:
    """Flatten a (possibly multi-section) abstract into one plain paragraph."""
    abstract = article.find("Abstract")
    if abstract is None:
        return ""
    segments = [_itertext(seg) for seg in abstract.findall("AbstractText")]
    return _clean(" ".join(seg for seg in segments if seg))


def _extract_year(pubdate: ET.Element | None) -> int | None:
    """Prefer <Year>; fall back to the first 4-digit run in <MedlineDate>."""
    if pubdate is None:
        return None
    year = pubdate.findtext("Year")
    if year and year.isdigit():
        return int(year)
    medline = pubdate.findtext("MedlineDate")
    if medline:
        match = re.search(r"\d{4}", medline)
        if match:
            return int(match.group())
    return None


def _extract_authors(author_list: ET.Element | None) -> list[str]:
    """Format each author as 'ForeName LastName' (or a collective/group name)."""
    if author_list is None:
        return []
    authors: list[str] = []
    for author in author_list.findall("Author"):
        collective = author.findtext("CollectiveName")
        if collective:
            authors.append(_clean(collective))
            continue
        last = author.findtext("LastName")
        fore = author.findtext("ForeName")
        if last and fore:
            authors.append(_clean(f"{fore} {last}"))
        elif last:
            authors.append(_clean(last))
    return authors


def _extract_mesh(citation: ET.Element) -> list[str]:
    """Pull the MeSH descriptor names (the main heading of each term)."""
    mesh_list = citation.find("MeshHeadingList")
    if mesh_list is None:
        return []
    terms: list[str] = []
    for heading in mesh_list.findall("MeshHeading"):
        descriptor = heading.find("DescriptorName")
        if descriptor is not None and descriptor.text:
            terms.append(_clean(descriptor.text))
    return terms


def _extract_doi(article: ET.Element, pubmed_data: ET.Element | None) -> str | None:
    """DOI from <ELocationID EIdType='doi'>, falling back to <ArticleIdList>."""
    for eloc in article.findall("ELocationID"):
        if eloc.get("EIdType") == "doi" and eloc.text:
            return eloc.text.strip()
    if pubmed_data is not None:
        for article_id in pubmed_data.iter("ArticleId"):
            if article_id.get("IdType") == "doi" and article_id.text:
                return article_id.text.strip()
    return None


def parse_article(article: ET.Element) -> Record | None:
    """Turn one <PubmedArticle> element into a Record, or None if unusable."""
    citation = article.find("MedlineCitation")
    if citation is None:
        return None
    pmid = citation.findtext("PMID")
    art = citation.find("Article")
    if not pmid or art is None:
        return None

    journal = art.find("Journal")
    pubdate = journal.find("JournalIssue/PubDate") if journal is not None else None

    return Record(
        pmid=pmid.strip(),
        title=_itertext(art.find("ArticleTitle")),
        abstract=_extract_abstract(art),
        journal=_itertext(journal.find("Title")) if journal is not None else "",
        year=_extract_year(pubdate),
        authors=_extract_authors(art.find("AuthorList")),
        mesh_terms=_extract_mesh(citation),
        doi=_extract_doi(art, article.find("PubmedData")),
    )


def iter_records(raw_dir: Path = RAW_DIR) -> Iterator[Record]:
    """Yield a Record for every <PubmedArticle> across all batch files.

    Batches are read in sorted filename order so the raw traversal is itself
    deterministic (the final corpus is additionally sorted by PMID).
    """
    for path in sorted(raw_dir.glob("batch_*.xml")):
        root = ET.parse(path).getroot()
        for article in root.findall("PubmedArticle"):
            record = parse_article(article)
            if record is not None:
                yield record


def dedup(records: Iterator[Record] | list[Record]) -> list[Record]:
    """Drop records sharing a PMID, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[Record] = []
    for record in records:
        if record.pmid in seen:
            continue
        seen.add(record.pmid)
        unique.append(record)
    return unique


def build_corpus(raw_dir: Path = RAW_DIR, out_dir: Path = CORPUS_DIR) -> dict[str, object]:
    """Parse → dedup → drop empty-abstract → write JSONL + dataset card.

    Returns the stats dict (also written into the dataset card). The corpus is
    sorted by integer PMID so the output bytes are identical run-to-run.
    """
    parsed = list(iter_records(raw_dir))
    deduped = dedup(parsed)
    usable = [r for r in deduped if r.abstract and r.title]
    usable.sort(key=lambda r: int(r.pmid))

    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for record in usable:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    stats = _compute_stats(usable, n_parsed=len(parsed), n_deduped=len(deduped))
    _write_card(out_dir, stats)
    return stats


def _compute_stats(records: list[Record], n_parsed: int, n_deduped: int) -> dict[str, object]:
    """Corpus summary for the dataset card / release note."""
    years = Counter(r.year for r in records if r.year is not None)
    journals = Counter(r.journal for r in records if r.journal)
    mesh = Counter(term for r in records for term in r.mesh_terms)
    return {
        "n_parsed": n_parsed,
        "n_after_dedup": n_deduped,
        "n_with_abstract": len(records),
        "n_duplicates_removed": n_parsed - n_deduped,
        "year_distribution": dict(sorted(years.items())),
        "top_journals": journals.most_common(10),
        "top_mesh": mesh.most_common(20),
    }


def _write_card(out_dir: Path, stats: dict[str, object]) -> None:
    """Write a human-readable dataset card alongside the corpus."""
    lines = [
        f"# SarcoLit corpus {CORPUS_VERSION}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "PubMed abstracts (sarcopenia + EMG-of-aging-muscle). Abstracts only, "
        "redistributable under NLM Terms. See `docs/releases/v0.1-corpus.md`.",
        "",
        "## Counts",
        "",
        f"- Records parsed: {stats['n_parsed']}",
        f"- Duplicates removed: {stats['n_duplicates_removed']}",
        f"- After dedup: {stats['n_after_dedup']}",
        f"- With usable abstract + title (final): {stats['n_with_abstract']}",
        "",
        "## Top 10 journals",
        "",
        *[f"- {name} ({n})" for name, n in stats["top_journals"]],  # type: ignore[union-attr]
        "",
        "## Top 20 MeSH terms",
        "",
        *[f"- {term} ({n})" for term, n in stats["top_mesh"]],  # type: ignore[union-attr]
        "",
    ]
    (out_dir / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    stats = build_corpus()
    print(
        f"corpus {CORPUS_VERSION}: {stats['n_with_abstract']} records "
        f"({stats['n_duplicates_removed']} duplicates removed) "
        f"-> {CORPUS_DIR / 'corpus.jsonl'}"
    )


if __name__ == "__main__":
    main()
