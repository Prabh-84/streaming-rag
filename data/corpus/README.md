# Corpus directory

Production corpora go here, one subdirectory per `corpus_id`. Nothing in this directory is
generated; `scripts/ingest_corpus.py` reads it and writes indexes to `data/processed/`.

```text
data/corpus/
  <corpus_id>/            # [A-Za-z0-9][A-Za-z0-9_-]{0,63}, e.g. "default"
    slots.yaml            # required — domain slot schema (REQ-CORPUS-01)
    *.md | *.markdown | *.txt   # documents; subdirectories allowed
```

- Markdown `#` headings split a document into sections `§1`, `§2`, ...; text before the first
  heading is `§0`. Citations take the form `[doc_id §section]`, where `doc_id` is the file's path
  inside the corpus directory without its extension.
- Documents must be UTF-8. Other file types are skipped and reported.
- `slots.yaml` shape:

  ```yaml
  corpus_id: default          # optional; must match the directory name if present
  description: ...            # optional
  slots:
    - name: location          # lowercase identifier, unique
      patterns: ["..."]       # one or more non-empty strings
      value_type: text        # numeric | categorical | date | text (default text)
  ```

Ingest with `python scripts/ingest_corpus.py --corpus-id <corpus_id>` (or `--all`). Re-running on an
unchanged corpus is a no-op; `--force` rebuilds.

The synthetic corpora under `tests/fixtures/corpora/` are test fixtures only — never copy them here.
