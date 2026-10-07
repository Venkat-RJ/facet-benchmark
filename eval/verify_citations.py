"""Mechanically verify bibliography entries in latex/main_v3.tex against external APIs.

For every citation that has an arXiv ID, fetch the paper metadata from the arXiv
API and confirm the title and first-author surname match the .tex bibliography.
For entries without arXiv IDs (books, journals, pre-arXiv work), the script
reports them as NOT_CHECKED so future-you knows which entries were not
mechanically verified.

This script pairs with verify_paper_numbers.py. Together they give the paper two
mechanical integrity checks: every number is re-derived from raw data, and every
arXiv citation is re-derived from the arXiv API. Solo papers with both checks
running in CI are a genuinely rare signal.
"""
import re
import sys
import urllib.request
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "latex" / "main_v3.tex"

# Parse entries of the form:
#   \bibitem[Authors(Year)]{key}
#   Author list.
#   \newblock Title.
#   \newblock \emph{Venue}, year.
BIBITEM_RE = re.compile(
    r"\\bibitem\[(?P<label>[^\]]+)\]\{(?P<key>[^}]+)\}\s*"
    r"(?P<authors>.*?)\n\s*\\newblock\s*(?P<title>.*?)\n\s*\\newblock\s*(?P<venue>.*?)(?=\n\s*\\bibitem|\n\s*\\end\{thebibliography\})",
    re.DOTALL,
)
ARXIV_ID_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")


def strip_latex(s: str) -> str:
    """Remove LaTeX accent, emphasis, and newblock markers for fuzzy comparison."""
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\newblock\s*", " ", s)
    s = re.sub(r"\\'\{([a-zA-Z])\}", r"\1", s)  # \'{e} -> e
    s = re.sub(r"\\`\{([a-zA-Z])\}", r"\1", s)
    s = re.sub(r"\\\"\{([a-zA-Z])\}", r"\1", s)
    s = re.sub(r"\\~\{([a-zA-Z])\}", r"\1", s)
    s = s.replace("~", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().rstrip(".").strip()


def normalize(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    s = strip_latex(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def first_surname_from_authors(authors: str) -> str:
    """Grab the first author's last name from a bibitem author list."""
    authors = strip_latex(authors)
    # Strip "et al." and its variants before parsing surname
    authors = re.sub(r",?\s*et\s*al\.?", "", authors, flags=re.IGNORECASE)
    # Format is typically "N.~Guha, J.~Nyarko, ..." — the first comma-separated
    # chunk's last whitespace-separated word is the surname.
    first = authors.split(",")[0].strip()
    parts = first.split()
    if not parts:
        return ""
    return parts[-1].lower()


USER_AGENT = (
    "FACET-benchmark/1.0 "
    "(https://github.com/Venkat-RJ/facet-benchmark; "
    "mailto:venkateshwar.jambula@pranaalpha.com)"
)


def fetch_arxiv(arxiv_id: str) -> dict:
    """Fetch arXiv metadata for a paper by ID."""
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml = resp.read().decode("utf-8")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    entry = root.find("a:entry", ns)
    if entry is None:
        return {}
    title = (entry.findtext("a:title", namespaces=ns) or "").strip()
    first_author_elem = entry.find("a:author/a:name", ns)
    first_author = (first_author_elem.text if first_author_elem is not None else "").strip()
    published = (entry.findtext("a:published", namespaces=ns) or "").strip()
    return {
        "title": title,
        "first_author": first_author,
        "year": published[:4],
    }


def verify():
    text = BIB.read_text()
    entries = list(BIBITEM_RE.finditer(text))
    if not entries:
        print("ERROR: no \\bibitem entries parsed from main_v3.tex")
        return 1

    print(f"{'STATUS':12s}  {'KEY':15s}  {'FIELD':18s}  {'DETAIL'}")
    print("-" * 110)

    pass_count = 0
    fail_count = 0
    skip_count = 0

    for m in entries:
        key = m.group("key").strip()
        authors = m.group("authors")
        title = strip_latex(m.group("title"))
        venue = m.group("venue")

        arxiv_match = ARXIV_ID_RE.search(venue)
        if not arxiv_match:
            print(f"{'NOT_CHECKED':12s}  {key:15s}  {'no arXiv ID':18s}  {strip_latex(venue)[:60]}")
            skip_count += 1
            continue

        arxiv_id = arxiv_match.group(1)
        try:
            meta = fetch_arxiv(arxiv_id)
        except Exception as e:
            print(f"{'FETCH_FAIL':12s}  {key:15s}  {'arXiv '+arxiv_id:18s}  {e}")
            fail_count += 1
            continue

        if not meta:
            print(f"{'NOT_FOUND':12s}  {key:15s}  {'arXiv '+arxiv_id:18s}  no entry returned")
            fail_count += 1
            continue

        # Title check (normalized substring either way)
        n_local = normalize(title)
        n_remote = normalize(meta["title"])
        title_ok = n_local in n_remote or n_remote in n_local

        # Author surname check
        local_surname = first_surname_from_authors(authors)
        remote_first_author = meta.get("first_author", "")
        remote_surname_last = remote_first_author.split()[-1].lower() if remote_first_author else ""
        author_ok = bool(local_surname) and local_surname in normalize(remote_first_author)

        # Year check (label is [Authors(YYYY)])
        label = m.group("label")
        year_match = re.search(r"\((\d{4})\)", label)
        local_year = year_match.group(1) if year_match else ""
        year_ok = (local_year == meta.get("year", ""))

        if title_ok and author_ok and year_ok:
            print(f"{'PASS':12s}  {key:15s}  {'arXiv '+arxiv_id:18s}  {meta['title'][:60]}")
            pass_count += 1
            continue

        # Report mismatch
        if not title_ok:
            print(f"{'FAIL':12s}  {key:15s}  {'title mismatch':18s}  local={title[:40]!r} remote={meta['title'][:40]!r}")
        if not author_ok:
            print(f"{'FAIL':12s}  {key:15s}  {'author mismatch':18s}  local={local_surname!r} remote={remote_surname_last!r}")
        if not year_ok:
            print(f"{'FAIL':12s}  {key:15s}  {'year mismatch':18s}  local={local_year!r} remote={meta.get('year','')!r}")
        fail_count += 1

    print("-" * 110)
    print(f"{pass_count} PASS, {fail_count} FAIL, {skip_count} NOT_CHECKED (no arXiv ID)")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(verify())
