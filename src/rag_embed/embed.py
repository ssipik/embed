from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client import models
from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from rag_embed.db import (
    get_engine,
    ingest_batch_docs,
    ingest_batches,
    ingest_docs,
    ingest_manifest,
    staged_nodes,
)
from rag_embed.qdrant import DENSE_VECTOR, SPARSE_VECTORS, QdrantWriter, point_id
from rag_embed.settings import get_settings
from rag_embed.sparse import Bm25Embedders
from rag_embed.tei import TeiClient

logger = logging.getLogger(__name__)


@dataclass
class EmbedBatchReport:
    batch_id: int
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "docs_embedded": 0, "docs_marked_only": 0, "points": 0
        }
    )


def build_points(
    nodes: list[dict],
    dense_vectors: list[list[float]],
    sparse_vectors: dict[str, list[models.SparseVector]],
) -> list[models.PointStruct]:
    """Staged nodes + their dense and sparse vectors → Qdrant points.

    sparse_vectors is {sparse vector name: one vector per node} — one entry per
    BM25 analyzer (qdrant.SPARSE_VECTORS). Each point carries all of them as
    named vectors alongside the dense one, for hybrid retrieval. Point id is
    deterministic (manifest_id and node_seq) — re-runs of the same sighting
    overwrite, never duplicate. Payload = chunk text + the chunk metadata; the
    chunk text lives here permanently, since Postgres staging is deleted by
    finalize.
    """
    # one column per sparse field; strict zip keeps a short column loud rather
    # than silently truncating to the shortest
    columns = list(sparse_vectors.values())
    points = [
        models.PointStruct(
            id=point_id(n["manifest_id"], n["node_seq"]),
            vector={
                DENSE_VECTOR: dvec,
                **dict(zip(sparse_vectors, svecs, strict=True)),
            },
            payload={"text": n["text"], "node_seq": n["node_seq"], **n["metadata"]},
        )
        for n, dvec, *svecs in zip(nodes, dense_vectors, *columns, strict=True)
    ]
    return points


def ledger_row(doc: dict) -> dict:
    """The ingest_docs values for one successfully ingested doc.

    manifest_id is this batch's sighting — the one whose chunks are now in
    Qdrant. Everything else about the document (url, title, doc_version) is
    read through that sighting, so the ledger stores only state.
    """
    return {
        "manifest_id": doc["manifest_id"],
        "text_sha": doc.get("text_sha"),
        "parser_version": doc.get("parser_version"),
        "status": "active",
    }


def _write_ledger(engine: Engine, previous_manifest_id: int | None, row: dict) -> None:
    """Write one document's ledger row.

    manifest_id is the ledger's primary key AND the value that has to move
    from the old sighting to the new one, which ON CONFLICT cannot express —
    hence update-or-insert rather than an upsert.
    """
    with engine.begin() as conn:
        if previous_manifest_id is None:
            conn.execute(insert(ingest_docs).values(**row))
        else:
            conn.execute(
                update(ingest_docs)
                .where(ingest_docs.c.manifest_id == previous_manifest_id)
                .values(**row)
            )


def run_embed_batch(
    batch_id: int,
    engine: Engine | None = None,
    embedder: TeiClient | None = None,
    sparse_embedder: Bm25Embedders | None = None,
    writer: QdrantWriter | None = None,
) -> EmbedBatchReport:
    """The embed stage for one parsed batch (the mapped Airflow task).

    'staged' docs: embed their staged nodes via TEI, replace the doc's
    points in Qdrant, mark the ledger. 'skipped_unchanged' docs: mark the
    ledger only — their vectors are current, but the fresh sighting's
    change signals must land or delta re-triggers 'changed' forever.
    Infra failures (TEI/Qdrant down) propagate and fail the batch task —
    only parse-stage problems are per-doc. Ends by setting the batch to
    'done' and returns counts only (the stage contract).
    """
    settings = get_settings()
    engine = engine or get_engine()
    own_embedder = embedder is None
    embedder = embedder or TeiClient(
        settings.tei_url, settings.http_timeout, settings.embed_batch_size
    )
    own_sparse = sparse_embedder is None
    sparse_embedder = sparse_embedder or Bm25Embedders.for_fields(
        SPARSE_VECTORS,
        settings.sparse_model_id,
        settings.sparse_avg_len,
        settings.embed_batch_size,
    )
    own_writer = writer is None
    writer = writer or QdrantWriter(settings.qdrant_url, settings.qdrant_collection)

    with engine.connect() as conn:
        batch = conn.execute(
            select(ingest_batches).where(ingest_batches.c.batch_id == batch_id)
        ).mappings().first()
        if batch is None:
            raise ValueError(f"unknown batch {batch_id!r} — did plan run?")
        batch_docs = [
            dict(r)
            for r in conn.execute(
                select(ingest_batch_docs).where(ingest_batch_docs.c.batch_id == batch_id)
            ).mappings()
        ]
        # No batch-status column: a batch is unparsed iff a doc is still
        # 'pending' (parse moves every doc to staged/skipped_unchanged/error).
        if any(d["status"] == "pending" for d in batch_docs):
            raise ValueError(
                f"batch {batch_id!r} not parsed yet — did parse_chunk run?"
            )
        nodes = [
            dict(r)
            for r in conn.execute(
                select(staged_nodes)
                .where(staged_nodes.c.batch_id == batch_id)
                .order_by(staged_nodes.c.manifest_id, staged_nodes.c.node_seq)
            ).mappings()
        ]
        batch_docs_ids = {d["manifest_id"] for d in batch_docs}
        # embed reads no manifest fields: chunk metadata comes from
        # staged_nodes, and ledger_row stores only state. The one manifest fact
        # it needs is the live sighting to delete, below.
        # The live sighting for each batch doc, keyed by the batch's
        # manifest_id — the id whose points must be deleted before the new
        # ones land. The ledger is per document, keyed by the older live
        # sighting, so bridge batch sighting → document → live sighting →
        # ledger, filtering on the batch manifest_id (a PK lookup) rather than
        # a row-value IN on (source_system, doc_url).
        bm = ingest_manifest.alias("bm")  # this batch's sightings
        lm = ingest_manifest.alias("lm")  # the ledger's live sightings
        ledger_docs_ids = {
            r["manifest_id"]: r["previous_manifest_id"]
            for r in conn.execute(
                select(
                    bm.c.manifest_id,
                    ingest_docs.c.manifest_id.label("previous_manifest_id"),
                )
                .select_from(bm)
                .join(
                    lm,
                    (lm.c.source_system == bm.c.source_system)
                    & (lm.c.doc_url == bm.c.doc_url),
                )
                .join(ingest_docs, ingest_docs.c.manifest_id == lm.c.manifest_id)
                .where(bm.c.manifest_id.in_(batch_docs_ids))
            ).mappings()
        }

    nodes_by_doc: dict[int, list[dict]] = {}
    for n in nodes:
        nodes_by_doc.setdefault(n["manifest_id"], []).append(n)

    report = EmbedBatchReport(batch_id=batch_id)
    try:
        for doc in batch_docs:
            logger.info("computing embeddings for doc: %s ----------------", doc['manifest_id'])
            manifest_id = doc["manifest_id"]
            # The sighting whose points are live now — None on first ingestion
            ledger_doc_id = ledger_docs_ids.get(manifest_id)
            if doc["status"] == "staged":
                doc_nodes = nodes_by_doc.get(manifest_id, [])
                texts = [n["text"] for n in doc_nodes]
                # Compute dense (fixed dimension) and sparse (dimensionless) vectors
                dense = embedder.embed(texts)
                sparse = sparse_embedder.embed(texts)
                points = build_points(doc_nodes, dense, sparse)
                # Update Qdrant entry
                if points:
                    writer.ensure_collection(dense_size=len(dense[0]))
                writer.replace_doc_points(ledger_doc_id, points)
                report.counts["docs_embedded"] += 1
                report.counts["points"] += len(points)
            elif doc["status"] == "skipped_unchanged":
                # Vectors are unchanged; the document has a new manifest_id
                if ledger_doc_id is not None and ledger_doc_id != manifest_id:
                    writer.repoint_doc_points(ledger_doc_id, manifest_id)
                report.counts["docs_marked_only"] += 1
            else:
                continue  # pending/error (parse problems) or already embedded

            # Update ledger entry
            _write_ledger(engine, ledger_doc_id, ledger_row(doc))
            # Update status of processing batch doc
            if doc["status"] == "staged":
                with engine.begin() as conn:
                    conn.execute(
                        update(ingest_batch_docs)
                        .where(
                            ingest_batch_docs.c.batch_id == batch_id,
                            ingest_batch_docs.c.manifest_id == manifest_id,
                        )
                        .values(status="embedded")
                    )
    finally:
        if own_embedder:
            embedder.close()
        if own_sparse:
            sparse_embedder.close()
        if own_writer:
            writer.close()

    # No batch-status write: a batch is 'done' (fully embedded) iff none of its
    # docs remain 'pending' or 'staged'; finalize derives that from per-doc
    # status. Already-embedded docs are recognised by their status on re-run.
    logger.info("embed %s: %s", batch_id, report.counts)
    return report
