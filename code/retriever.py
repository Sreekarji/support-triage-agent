"""
retriever.py — TF-IDF retriever over the HackerRank / Claude / Visa support corpus.

Loads every .md file from data/{hackerrank,claude,visa}/, chunks them into
passages by markdown headings and paragraphs, builds a TF-IDF index, and
exposes a deterministic search(query, company, top_k) function.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    """A single chunk of text from the support corpus."""
    text: str
    source_file: str      # relative path inside data/
    domain: str           # "hackerrank" | "claude" | "visa"
    title: str = ""       # first heading or cleaned filename


@dataclass
class SearchResult:
    """A ranked search result."""
    text: str
    source_file: str
    domain: str
    title: str
    score: float


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """TF-IDF retriever with optional per-domain filtering."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Load all .md files, chunk them, and build the TF-IDF index.

        Parameters
        ----------
        data_dir : str or None
            Absolute path to the ``data/`` directory.  Defaults to
            ``<repo_root>/data`` relative to this file.
        """
        if data_dir is None:
            data_dir = str(Path(__file__).resolve().parent.parent / "data")

        self.data_dir = Path(data_dir)
        self.passages: List[Passage] = []

        # Deterministic vectoriser — no randomness involved
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = None

        self._load_all_documents()
        self._build_index()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    _DOMAINS = ("hackerrank", "claude", "visa")

    def _load_all_documents(self) -> None:
        """Recursively load every .md file under each domain directory."""
        for domain in self._DOMAINS:
            domain_dir = self.data_dir / domain
            if not domain_dir.exists():
                print(f"[retriever] WARNING: {domain_dir} not found, skipping.")
                continue

            for md_file in sorted(domain_dir.rglob("*.md")):
                try:
                    text = md_file.read_text(encoding="utf-8")
                except Exception as exc:
                    print(f"[retriever] WARNING: cannot read {md_file}: {exc}")
                    continue

                if not text.strip():
                    continue

                title = self._extract_title(text, md_file)
                rel_path = str(md_file.relative_to(self.data_dir))
                chunks = self._chunk_document(text)

                for chunk in chunks:
                    chunk = chunk.strip()
                    if len(chunk) < 30:          # skip trivially short chunks
                        continue
                    self.passages.append(
                        Passage(text=chunk, source_file=rel_path,
                                domain=domain, title=title)
                    )

        # --- summary ---
        unique_files = len({p.source_file for p in self.passages})
        print(f"[retriever] Loaded {len(self.passages)} passages "
              f"from {unique_files} files across {len(self._DOMAINS)} domains")
        for domain in self._DOMAINS:
            d_passages = [p for p in self.passages if p.domain == domain]
            d_files = len({p.source_file for p in d_passages})
            print(f"  * {domain:12s}: {d_files:>4d} files, {len(d_passages):>5d} passages")

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(text: str, filepath: Path) -> str:
        """Return the first ``# Heading`` or a cleaned-up filename."""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return stripped.lstrip("# ").strip()
        # Fallback: filename without numeric prefix
        name = re.sub(r"^\d+-", "", filepath.stem)
        return name.replace("-", " ").strip().title()

    @staticmethod
    def _chunk_document(text: str, max_chunk_size: int = 1500) -> List[str]:
        """
        Split a markdown document into passages.

        Strategy:
        1. Split on markdown headings (``#``, ``##``, ``###``).
        2. If a section exceeds *max_chunk_size*, split further on
           paragraph boundaries (double-newline).
        """
        sections = re.split(r"\n(?=#{1,3}\s)", text)
        chunks: List[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= max_chunk_size:
                chunks.append(section)
            else:
                # Sub-split on paragraphs
                paragraphs = re.split(r"\n\s*\n", section)
                buf = ""
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if buf and len(buf) + len(para) + 2 > max_chunk_size:
                        chunks.append(buf)
                        buf = para
                    else:
                        buf = f"{buf}\n\n{para}" if buf else para
                if buf:
                    chunks.append(buf)

        if not chunks and text.strip():
            chunks = [text.strip()]

        return chunks

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Fit the TF-IDF vectoriser on all passages."""
        if not self.passages:
            print("[retriever] WARNING: no passages to index.")
            return

        texts = [p.text for p in self.passages]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[retriever] TF-IDF index built: "
              f"{self.tfidf_matrix.shape[0]} docs x {self.tfidf_matrix.shape[1]} features")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    _COMPANY_ALIASES = {
        "hackerrank": "hackerrank",
        "hacker rank": "hackerrank",
        "hr": "hackerrank",
        "claude": "claude",
        "anthropic": "claude",
        "visa": "visa",
    }

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Return the *top_k* most relevant passages for *query*.

        Parameters
        ----------
        query : str
            Free-text search query.
        company : str or None
            If given, restrict results to this domain.
            Accepts common aliases (``"HackerRank"``, ``"Anthropic"``, …).
        top_k : int
            Maximum number of results.

        Returns
        -------
        list[SearchResult]
            Sorted by descending cosine similarity.  Results with
            score ≤ 0 are omitted.
        """
        if self.tfidf_matrix is None or not self.passages:
            return []

        # --- normalise company filter ---
        domain_filter: Optional[str] = None
        if company and company.strip().lower() not in ("none", ""):
            domain_filter = self._COMPANY_ALIASES.get(
                company.strip().lower(), company.strip().lower()
            )

        # --- query vector ---
        query_vec = self.vectorizer.transform([query])

        # --- cosine similarities ---
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # --- domain mask ---
        if domain_filter:
            mask = np.array([p.domain == domain_filter for p in self.passages],
                            dtype=np.float64)
            sims = sims * mask

        # --- deterministic top-k (argsort is stable) ---
        top_idx = np.argsort(-sims)[:top_k]

        results: List[SearchResult] = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0:
                break                    # remaining will also be ≤ 0
            p = self.passages[idx]
            results.append(SearchResult(
                text=p.text,
                source_file=p.source_file,
                domain=p.domain,
                title=p.title,
                score=score,
            ))

        return results


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Initialising retriever ...")
    print("=" * 60)
    retriever = Retriever()

    test_cases = [
        ("How do I add extra time for candidates on a test?", "hackerrank"),
        ("How do I delete a conversation with private info?", "claude"),
        ("lost or stolen Visa card India", "visa"),
        ("Who is the actor in Iron Man?", None),
    ]

    for query, company in test_cases:
        label = company or "ANY"
        print(f"\n{'-' * 60}")
        print(f"Query   : {query}")
        print(f"Company : {label}")
        print(f"{'-' * 60}")
        results = retriever.search(query, company=company, top_k=3)
        if not results:
            print("  (no results)")
        for i, r in enumerate(results, 1):
            preview = r.text[:200].replace("\n", " ")
            print(f"\n  [{i}] score={r.score:.4f}  domain={r.domain}")
            print(f"      file ={r.source_file}")
            print(f"      title={r.title}")
            print(f"      text ={preview}...")
