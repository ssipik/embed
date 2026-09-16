# rag-embed

Stage 5 (EMBED) of the RAG ingestion pipeline — see [`../ARCHITECTURE.md`](../ARCHITECTURE.md) (blocks 5, 6, 11). For one run (all parsed documents in the run's batches, one Nomad job): embed the staged nodes via the self-hosted **TEI** service (BGE-M3), upsert points into **Qdrant**, and do the ledger bookkeeping that makes ingestion count as complete.

- **`staged` docs** → embed (dense via TEI + sparse via BM25) → replace the doc's Qdrant points (deterministic integer point ids; delete-by-doc-filter first, so a shrunken re-chunk leaves no stale points) → upsert the `ingest_docs` ledger row.
- **`skipped_unchanged` docs** (layer-2 drop in stage 4) → ledger row only: vectors are current, but the fresh sighting's change signals must land or delta would re-trigger `changed` forever.
- **This stage owns ledger writes** — a doc becomes "known" only once its vectors actually landed (crash-safety). `rag_delta.mark_ingested()` was the dev stand-in for exactly this and is superseded when running the full chain.
- Hybrid vectors: **dense** (BGE-M3 via TEI) + **sparse** (BM25 via fastembed, in-process, German analyzer) stored as named Qdrant vectors with server-side IDF — the sparse leg of block-7 hybrid retrieval. BGE-M3's own sparse head isn't served by any self-hostable server, so BM25 fills the sparse leg (see the stage-5 decision in `../ARCHITECTURE.md`).

## Usage

```bash
uv sync --frozen --extra dev        # lock-exact venv from uv.lock
.venv/bin/python -m pytest          # offline tests (mock TEI + fake BM25, in-memory Qdrant)

# Embed one run (needs ../postgres, ../qdrant and ../tei up)
python -m rag_embed --run-id <run id plan was given>

# Containerized
docker compose run --rm --build pipeline python -m rag_embed --run-id <run id>
```

Config env vars: `RAG_TEI_URL` (default `http://localhost:8080`), `RAG_QDRANT_URL` (default `http://localhost:6333`), `RAG_QDRANT_COLLECTION` (`rag_chunks`), `RAG_EMBED_BATCH_SIZE` (32), `RAG_SPARSE_AVG_LEN` (256), `RAG_PG_DSN`; concurrency: `RAG_TEI_REPLICAS` (rendered by the Nomad job), `RAG_EMBED_CLIENTS_PER_REPLICA` (2), `RAG_EMBED_CONCURRENCY` (override). The BM25 analyzer languages are not env-configurable — they name the collection's sparse fields, so they live in `qdrant.SPARSE_VECTORS`.

## Design notes

- **Parallel over TEI replicas, not over batches**: documents are packed into groups of ~32 chunks (one full TEI request) and processed by `clients_per_replica × healthy TEI replicas` worker threads. More clients than that only wait (measured, `../evaluation/tests/bench_tei_embed_concurrency.py`).
- **Infra failures propagate** (TEI/Qdrant unreachable → the run fails; the retry skips documents already `embedded`); only parse-stage problems are per-doc. The DAG health-checks TEI before dispatching (block 5).
- Chunk text lives permanently in Qdrant point payloads next to the vectors (block 6: Postgres staging is transient — finalize clears it).
- Standalone module by convention: infrastructure (`settings.py`, `sources.py`, `db.py`) duplicated from the other stages, dedup deferred.
