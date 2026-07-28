# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Stage 5 (EMBED) of the six-stage ingestion skeleton — `../ARCHITECTURE.md` blocks 5 (TEI), 6 (Qdrant/ledger) and 11 are authoritative. Standalone stage module: `settings.py`, `sources.py`, `db.py` are **deliberately duplicated** from the other stage packages — fix infra bugs in every copy.

## Commands

```bash
uv sync --frozen --extra dev                  # lock-exact venv
.venv/bin/python -m pytest                    # offline tests (mock TEI + fake BM25, in-memory Qdrant)
python -m rag_embed --batch-id <id>    # needs ../postgres, ../qdrant, ../tei up
```

## Invariants

- **Library-first, thin `__main__`**: stage logic in `rag_embed.embed.run_embed_batch()`; `__main__.py` is a logic-free argparse wrapper.
- **This stage owns `ingest_docs` writes** (the block-11 crash-safety rule: ledger rows appear only when vectors actually landed). The write is update-or-insert, not an upsert: `manifest_id` is the PK *and* migrates from the old live sighting to the new one (`_write_ledger`). `skipped_unchanged` docs get a ledger write too (fresh signals + unchanged text_sha) or delta re-triggers forever.
- **Deterministic point identity**: integer bit-pack `(manifest_id << 20) | node_seq` (`qdrant.point_id`). Doc updates go through `replace_doc_points` (delete-by-filter on `manifest_id`, then upsert) — never bare upserts, or shrunken re-chunks leave stale tail points.
- **Infra failures fail the batch** (TEI/Qdrant errors propagate for the task retry); per-doc handling is stage 4's job. A batch still `planned` refuses to run ("did parse_chunk run?").
- **Hybrid dense + sparse**: dense from TEI (`tei.TeiClient`), sparse from in-process BM25 (`sparse.Bm25Embedder`, fastembed `Qdrant/bm25`, German). Points carry named vectors `{"dense_bge_m3", "sparse_bm25"}`; the collection sets `Modifier.IDF`, so Qdrant applies IDF at query time. **Cross-stage contract**: the retrieval stage must embed queries with the same BM25 model + `language="german"` and the **same fastembed version** — its stemmer + token hash define the sparse id space, so a drift silently breaks matching. BGE-M3's own sparse head stays deferred (no self-hostable server serves it; block-7 decision + rejected alternatives in `../ARCHITECTURE.md`).
- Qdrant/TEI/BM25 details live behind `qdrant.QdrantWriter` / `tei.TeiClient` / `sparse.Bm25Embedder`; tests use qdrant-client's `:memory:` mode, httpx MockTransport, and a fake BM25 model injected into `Bm25Embedder` — keep it that way (offline; the real BM25 model downloads its stemmer/stopwords from HF on first use).
- Schema mirrors in `db.py` (`ingest_docs` write side, manifest/batches/staged_nodes read side) — change together with `../postgres/sql/*.sql`.
