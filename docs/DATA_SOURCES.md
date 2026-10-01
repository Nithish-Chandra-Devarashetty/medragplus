# Knowledge sources and licences

MedRAG+ replaces the proprietary textbook corpus of the base paper (Merck Manual,
Harrison's, Oxford Handbook, Gale Encyclopedia) with openly licensed sources.
In the base paper those books were used only as the retrieval knowledge base;
no model was trained on them, so swapping the corpus changes no code.

| Source | Folder | Licence | Fetched by |
|---|---|---|---|
| MedlinePlus Health Topics (US National Library of Medicine), ~1,000 English topics | `data/medical_documents/medlineplus/` | Public domain (US Government work) | `scripts/download_data.py --medlineplus` |
| WHO fact sheets, ~240 topics | `data/medical_documents/who/` | CC BY-NC-SA 3.0 IGO (non-commercial, attribution) | `scripts/download_data.py --who` |
| MedQuAD (MedlinePlus subset used for evaluation only, **not** indexed) | `data/raw/medquad/` | CC BY 4.0 | `scripts/download_data.py --medquad` |

## Adding more sources

Drop files into `data/medical_documents/<source>/` and run `python scripts/ingest.py`.
Supported formats: PDF (page numbers are kept), `.txt`, `.md` (optionally with
`title:` / `url:` front matter; split per `##` section), `.html`, MedlinePlus XML
and JATS `.nxml` (e.g. StatPearls).

Recommended additions:

* **StatPearls** (NCBI Bookshelf, CC BY-NC-ND 4.0): about 9,000 peer-reviewed clinical
  articles, the closest free equivalent to Merck/Harrison's. Available as a bulk
  NXML download from the NCBI FTP site. Large (several GB), so add it after the
  MVP works.
* **WHO guideline PDFs** (e.g. *Emergency Triage Assessment and Treatment*). The
  WHO IRIS repository blocks scripted downloads, so download these manually.
* The original textbooks, **only** if your institution licenses them for this use.

## Citing in the paper

State that the knowledge base differs from the base paper ("we replace the
proprietary textbook corpus with openly licensed sources: MedlinePlus and WHO"),
and do not compare answer-quality numbers as if both systems used the same corpus.
