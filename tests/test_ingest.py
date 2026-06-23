"""Unit tests for the PubMed fetcher's pure logic (no network)."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import pytest

from sarcolit.ingest import parse, pubmed


def test_base_params_includes_credentials_when_set() -> None:
    cfg = pubmed.FetchConfig(email="a@b.com", api_key="key123")
    params = pubmed._base_params(cfg)
    assert params["db"] == "pubmed"
    assert params["email"] == "a@b.com"
    assert params["api_key"] == "key123"


def test_base_params_omits_credentials_when_absent() -> None:
    cfg = pubmed.FetchConfig(email=None, api_key=None)
    params = pubmed._base_params(cfg)
    assert "email" not in params
    assert "api_key" not in params


def test_from_env_reads_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCBI_EMAIL", "env@b.com")
    monkeypatch.setenv("NCBI_API_KEY", "envkey")
    cfg = pubmed.FetchConfig.from_env()
    assert cfg.email == "env@b.com"
    assert cfg.api_key == "envkey"


def test_from_env_blank_vars_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCBI_EMAIL", "")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    cfg = pubmed.FetchConfig.from_env()
    assert cfg.email is None
    assert cfg.api_key is None


def test_query_hash_is_stable_under_whitespace() -> None:
    a = pubmed._query_hash("foo   bar\n  baz")
    b = pubmed._query_hash("foo bar baz")
    assert a == b
    assert len(a) == 12


def test_is_transient_classification() -> None:
    request = httpx.Request("GET", "https://example.com")
    too_many = httpx.HTTPStatusError(
        "429", request=request, response=httpx.Response(429, request=request)
    )
    server_err = httpx.HTTPStatusError(
        "503", request=request, response=httpx.Response(503, request=request)
    )
    bad_query = httpx.HTTPStatusError(
        "400", request=request, response=httpx.Response(400, request=request)
    )
    assert pubmed._is_transient(too_many) is True
    assert pubmed._is_transient(server_err) is True
    assert pubmed._is_transient(bad_query) is False
    assert pubmed._is_transient(httpx.ConnectError("boom")) is True
    assert pubmed._is_transient(ValueError("unrelated")) is False


def test_throttle_enforces_minimum_interval() -> None:
    # 50 ms interval, asserted loosely at 30 ms: comfortably above Windows'
    # ~15.6 ms monotonic-clock granularity so the test isn't flaky there.
    throttle = pubmed._Throttle(max_per_second=20.0)  # 50 ms min interval
    throttle.wait()
    start = time.monotonic()
    throttle.wait()
    assert time.monotonic() - start >= 0.03


# --- parse.py: cleaning/extraction (no network) -----------------------------

# A minimal but representative <PubmedArticleSet> exercising the messy bits:
# structured abstract, inline markup, MedlineDate-only year, DOI in two places,
# a collective author, and a duplicate + an abstract-less record.
SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">222</PMID>
      <Article>
        <Journal>
          <Title>Journal of Aging</Title>
          <JournalIssue><PubDate><Year>2010</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Muscle <i>quality</i> in aging</ArticleTitle>
        <ELocationID EIdType="doi" ValidYN="Y">10.1/abc</ELocationID>
        <Abstract>
          <AbstractText Label="AIM">To assess muscle.</AbstractText>
          <AbstractText Label="METHODS">We measured 30 adults.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Sarcopenia</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Aging</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData></PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">111</PMID>
      <Article>
        <Journal>
          <Title>EMG Reports</Title>
          <JournalIssue><PubDate><MedlineDate>2001 Jan-Feb</MedlineDate></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Surface EMG of elderly muscle</ArticleTitle>
        <Abstract><AbstractText>A single-section abstract.</AbstractText></Abstract>
        <AuthorList>
          <Author><CollectiveName>EWGSOP Group</CollectiveName></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">111</ArticleId>
        <ArticleId IdType="doi">10.2/def</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">111</PMID>
      <Article>
        <Journal><Title>EMG Reports</Title></Journal>
        <ArticleTitle>Duplicate of 111</ArticleTitle>
        <Abstract><AbstractText>Should be dropped as a dup.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">333</PMID>
      <Article>
        <Journal><Title>No Abstract Journal</Title></Journal>
        <ArticleTitle>Record with no abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _articles() -> list[ET.Element]:
    return ET.fromstring(SAMPLE_XML).findall("PubmedArticle")


def test_parse_article_flattens_structured_abstract_and_markup() -> None:
    rec = parse.parse_article(_articles()[0])
    assert rec is not None
    assert rec.pmid == "222"
    assert rec.title == "Muscle quality in aging"  # inline <i> flattened
    assert rec.abstract == "To assess muscle. We measured 30 adults."  # labels gone
    assert rec.journal == "Journal of Aging"
    assert rec.year == 2010
    assert rec.authors == ["Jane Smith"]
    assert rec.mesh_terms == ["Sarcopenia", "Aging"]
    assert rec.doi == "10.1/abc"


def test_parse_article_fallbacks_medlinedate_doi_and_collective() -> None:
    rec = parse.parse_article(_articles()[1])
    assert rec is not None
    assert rec.year == 2001  # extracted from "2001 Jan-Feb"
    assert rec.doi == "10.2/def"  # found in ArticleIdList, not ELocationID
    assert rec.authors == ["EWGSOP Group"]  # collective name used


def test_dedup_keeps_first_by_pmid() -> None:
    parsed = [parse.parse_article(a) for a in _articles()]
    parsed = [r for r in parsed if r is not None]
    deduped = parse.dedup(parsed)
    pmids = [r.pmid for r in deduped]
    assert pmids.count("111") == 1
    assert deduped[1].title == "Surface EMG of elderly muscle"  # first 111 kept


def test_build_corpus_is_deterministic_and_filters(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "batch_0000.xml").write_text(SAMPLE_XML, encoding="utf-8")
    out = tmp_path / "corpus"

    stats = parse.build_corpus(raw_dir=raw, out_dir=out)
    first = (out / "corpus.jsonl").read_bytes()

    # No abstract (333) and the duplicate (second 111) are excluded.
    assert stats["n_with_abstract"] == 2
    assert stats["n_duplicates_removed"] == 1

    parse.build_corpus(raw_dir=raw, out_dir=out)  # re-run
    assert (out / "corpus.jsonl").read_bytes() == first  # byte-identical

    # Sorted by integer PMID: 111 before 222.
    lines = first.decode("utf-8").splitlines()
    assert '"pmid": "111"' in lines[0]
    assert '"pmid": "222"' in lines[1]
