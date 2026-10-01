"""Download the openly licensed knowledge base and evaluation data.

    python scripts/download_data.py                  # everything
    python scripts/download_data.py --medlineplus    # just one source

Sources (see docs/DATA_SOURCES.md for licences):
  medlineplus  US NLM MedlinePlus health topics XML (public domain)    -> data/medical_documents/medlineplus/
  who          WHO fact sheets (CC BY-NC-SA 3.0 IGO)                    -> data/medical_documents/who/
  medquad      MedQuAD question/answer pairs for evaluation (CC BY 4.0) -> data/raw/medquad/

Any other PDFs / text you are licensed to use (e.g. WHO guideline PDFs, which
WHO's IRIS site does not allow to be fetched by scripts) can simply be dropped
into data/medical_documents/<source>/ before running scripts/ingest.py.
"""
from __future__ import annotations

import argparse
import html
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "medical_documents"
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "MedRAG-plus/0.1 (academic research project)"}


def get(url: str, **kwargs) -> requests.Response:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- MedlinePlus

def download_medlineplus() -> None:
    index = get("https://medlineplus.gov/xml.html").text
    files = sorted(set(re.findall(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", index)))
    if not files:
        raise RuntimeError("No MedlinePlus topic XML link found on https://medlineplus.gov/xml.html")
    latest = files[-1]
    out_dir = DOCS / "medlineplus"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / latest.rsplit("/", 1)[1]
    if target.exists():
        print(f"MedlinePlus: {target.name} already present")
        return
    print(f"MedlinePlus: downloading {latest}")
    target.write_bytes(get(latest).content)
    for old in out_dir.glob("mplus_topics_*.xml"):
        if old != target:
            old.unlink()  # keep exactly one snapshot so chunks are not indexed twice
    print(f"MedlinePlus: saved {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------- WHO fact sheets

def html_to_markdown(fragment: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", fragment)
    text = re.sub(r"(?i)<h[1-4][^>]*>(.*?)</h[1-4]>", lambda m: f"\n\n## {re.sub(r'<[^>]+>', '', m.group(1)).strip()}\n\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</p>|<br\s*/?>|</li>|</ul>|</ol>", "\n", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def download_who(limit: int | None = None) -> None:
    base = "https://www.who.int"
    index = get(f"{base}/news-room/fact-sheets").text
    slugs = sorted(set(re.findall(r'href="/news-room/fact-sheets/detail/([^"#?]+)"', index)))
    if limit:
        slugs = slugs[:limit]
    out_dir = DOCS / "who"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"WHO: {len(slugs)} fact sheets")
    saved = 0
    for i, slug in enumerate(slugs, 1):
        safe = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
        target = out_dir / f"{safe}.md"
        if target.exists():
            continue
        url = f"{base}/news-room/fact-sheets/detail/{slug}"
        try:
            page = get(url).text
        except requests.RequestException as exc:
            print(f"  ! {slug}: {exc}")
            continue
        title_match = re.search(r"<title>\s*([^<]+?)\s*</title>", page)
        start = page.find("sf-detail-body-wrapper")
        if start < 0:
            continue
        end = page.find("related-item", start)
        body = html_to_markdown(page[page.find(">", start) + 1:end if end > 0 else None])
        body = re.split(r"\n## (References|Related)\b", body)[0].strip()
        if len(body) < 200:
            continue
        title = html.unescape(title_match.group(1)) if title_match else slug
        target.write_text(f"---\ntitle: {title}\nurl: {url}\n---\n\n{body}\n", encoding="utf-8")
        saved += 1
        if i % 25 == 0:
            print(f"  {i}/{len(slugs)}")
        time.sleep(0.5)  # be polite
    print(f"WHO: saved {saved} new fact sheets to {out_dir.relative_to(ROOT)}")


# ---------------------------------------------------------------- MedQuAD

def download_medquad() -> None:
    out_dir = RAW / "medquad"
    if out_dir.exists() and any(out_dir.rglob("*.xml")):
        print("MedQuAD: already present")
        return
    print("MedQuAD: downloading GitHub archive")
    archive = zipfile.ZipFile(io.BytesIO(get("https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip").content))
    out_dir.mkdir(parents=True, exist_ok=True)
    for member in archive.namelist():
        if member.endswith(".xml") or member.endswith(("README.md", "LICENSE.txt", "LICENSE")):
            rel = Path(*Path(member).parts[1:])
            if rel.parts:
                dest = out_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(member))
    print(f"MedQuAD: {sum(1 for _ in out_dir.rglob('*.xml'))} XML files in {out_dir.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medlineplus", action="store_true")
    parser.add_argument("--who", action="store_true")
    parser.add_argument("--medquad", action="store_true")
    parser.add_argument("--who-limit", type=int, default=None, help="only fetch the first N WHO fact sheets")
    args = parser.parse_args()
    run_all = not (args.medlineplus or args.who or args.medquad)
    failures = 0
    for enabled, fn in ((args.medlineplus, download_medlineplus), (args.who, lambda: download_who(args.who_limit)),
                        (args.medquad, download_medquad)):
        if run_all or enabled:
            try:
                fn()
            except Exception as exc:
                failures += 1
                print(f"FAILED: {exc}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
