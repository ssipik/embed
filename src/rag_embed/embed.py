from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from math import ceil
from typing import Callable

from qdrant_client import models
from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection, Engine

from rag_embed.db import (
    get_engine,
    ingest_batch_docs,
    ingest_batches,
    ingest_docs,
    ingest_manifest,
    staged_nodes,
)
from rag_embed.qdrant import DENSE_VECTOR, SPARSE_LEGS, QdrantWriter, point_id
from rag_embed.settings import Settings, get_settings
from rag_embed.sparse import Bm25Embedders
from rag_embed.tei import TeiClient

logger = logging.getLogger(__name__)

# Statuses parse leaves on a document that embed acts on.
WORK_STATUSES = ("staged", "skipped_unchanged", "skipped_empty")
# Cap on documents in one group; only reached by skipped docs and one-chunk documents.
MAX_DOCS_PER_GROUP = 100


def _zero_counts() -> dict[str, int]:
    return {"docs_embedded": 0, "docs_marked_only": 0, "docs_empty": 0, "points": 0}


@dataclass
class EmbedRunReport:
    run_id: str
    concurrency: int = 1
    groups: int = 0
    counts: dict[str, int] = field(default_factory=_zero_counts)

    def add(self, counts: dict[str, int]) -> None:
        for key, value in counts.items():
            self.counts[key] += value


def build_points(
    nodes: list[dict],
    dense_vectors: list[list[float]],
    sparse_vectors: dict[str, list[models.SparseVector]],
) -> list[models.PointStruct]:
    """Staged nodes + their dense and sparse vectors → Qdrant points.

    sparse_vectors is {sparse vector name: one vector per node} — one entry per
    BM25 analyzer (qdrant.SPARSE_LEGS). Each point carries all of them as
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


def ledger_row(doc: dict, status: str = "active") -> dict:
    """The ingest_docs values for one ingested doc.

    manifest_id is this batch's sighting — the one whose chunks are now in
    Qdrant. Everything else about the document (url, title, doc_version) is
    read through that sighting, so the ledger stores only state.

    status is 'active', or 'empty' for a document that parsed to no text and
    therefore has no vectors at all.
    """
    return {
        "manifest_id": doc["manifest_id"],
        "text_sha": doc.get("text_sha"),
        "parser_version": doc.get("parser_version"),
        "status": status,
    }


def _write_ledger(conn: Connection, previous_manifest_id: int | None, row: dict) -> None:
    """Write one document's ledger row.

    manifest_id is the ledger's primary key AND the value that has to move
    from the old sighting to the new one, which ON CONFLICT cannot express —
    hence update-or-insert rather than an upsert.
    """
    if previous_manifest_id is None:
        conn.execute(insert(ingest_docs).values(**row))
    else:
        conn.execute(
            update(ingest_docs)
            .where(ingest_docs.c.manifest_id == previous_manifest_id)
            .values(**row)
        )


def select_work(rows: list[dict]) -> list[dict]:
    """The run's documents embed still has to process, one row per document.

    rows are all of the run's ingest_batch_docs rows. Plan is append-only, so a
    re-planned document has a row in every batch it was planned in; the row of
    the newest batch is its current state:
    - 'pending': raise, parse has not run for it
    - staged / skipped_unchanged / skipped_empty: work
    - embedded / error: nothing to do
    """
    latest: dict[int, dict] = {}
    for row in rows:
        current = latest.get(row["manifest_id"])
        if current is None or row["batch_id"] > current["batch_id"]:
            latest[row["manifest_id"]] = row
    pending = sorted(mid for mid, row in latest.items() if row["status"] == "pending")
    if pending:
        raise ValueError(
            f"{len(pending)} document(s) not parsed yet (e.g. manifest_id "
            f"{pending[0]}) — did parse_chunk run?"
        )
    return [row for _, row in sorted(latest.items()) if row["status"] in WORK_STATUSES]


def pack_groups(
    docs: list[dict], max_chunks: int, max_docs: int = MAX_DOCS_PER_GROUP
) -> list[list[dict]]:
    """Split the work into groups that one worker processes together.

    - staged docs fill a group up to max_chunks chunks (one full TEI request)
    - a doc with max_chunks chunks or more gets a group of its own
    - skipped docs have no chunks and are grouped separately
    - no group holds more than max_docs documents
    """
    groups: list[list[dict]] = []
    current: list[dict] = []
    chunks = 0
    for doc in docs:
        if doc["status"] != "staged":
            continue
        # an unknown count is treated as a full group, so it can't overfill one
        n = doc.get("node_count") or max_chunks
        if current and (chunks + n > max_chunks or len(current) >= max_docs):
            groups.append(current)
            current, chunks = [], 0
        current.append(doc)
        chunks += n
    if current:
        groups.append(current)
    skipped = [doc for doc in docs if doc["status"] != "staged"]
    groups += [skipped[i : i + max_docs] for i in range(0, len(skipped), max_docs)]
    return groups


def group_chunk_target(docs: list[dict], max_chunks: int, concurrency: int) -> int:
    """Chunks per group, so that every worker gets one.

    pack_groups otherwise fills each group to max_chunks (one full TEI
    request), so a run holding fewer than max_chunks x concurrency chunks
    produces fewer groups than workers and leaves the extra TEI replicas the
    workers were sized to idle. The target only ever lowers the group size:
    - a run with no staged docs, or a single worker: max_chunks, unchanged
    - a run big enough to fill every worker: max_chunks, unchanged
    - a smaller run: its chunks divided over the workers
    """
    if concurrency <= 1:
        return max_chunks
    total = sum(
        doc.get("node_count") or max_chunks for doc in docs if doc["status"] == "staged"
    )
    if total <= 0:
        return max_chunks
    return max(1, min(max_chunks, ceil(total / concurrency)))


@dataclass
class Clients:
    """One worker's connections to TEI, BM25 and Qdrant — never shared between threads."""

    embedder: TeiClient
    sparse: Bm25Embedders
    writer: QdrantWriter

    def close(self) -> None:
        self.embedder.close()
        self.sparse.close()
        self.writer.close()


def make_clients(settings: Settings) -> Clients:
    return Clients(
        embedder=TeiClient(settings.tei_url, settings.tei_timeout, settings.embed_batch_size),
        sparse=Bm25Embedders.for_legs(
            SPARSE_LEGS, settings.sparse_avg_len, settings.embed_batch_size
        ),
        writer=QdrantWriter(
            settings.qdrant_url, settings.qdrant_collection, settings.http_timeout
        ),
    )


class CollectionGuard:
    """Creates the Qdrant collection once, even when several workers get there together."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = False

    def __call__(self, writer: QdrantWriter, dense_size: int) -> None:
        if self._done:
            return
        with self._lock:
            if not self._done:
                writer.ensure_collection(dense_size=dense_size)
                self._done = True


@dataclass
class GroupResult:
    # (the ledger's previous sighting, the new ledger row) per document
    ledger: list[tuple[int | None, dict]]
    # staged docs whose batch rows become 'embedded'
    embedded_ids: list[int]
    counts: dict[str, int]


def process_group(
    group: list[dict], nodes: list[dict], clients: Clients, ensure_collection: CollectionGuard
) -> GroupResult:
    """Vectors and Qdrant writes for one group; write_group_state records the result in Postgres.

    nodes are the staged_nodes of the group's staged docs, ordered by
    (manifest_id, node_seq). Per document status:
    - 'staged': TEI + BM25 vectors, replace the document's points, ledger 'active'
    - 'skipped_unchanged': vectors are current; move the points to the new
      sighting, ledger 'active' (or delta re-triggers 'changed' forever)
    - 'skipped_empty': no text, so delete the points, ledger 'empty'
    """
    counts = _zero_counts()
    staged = [d for d in group if d["status"] == "staged"]
    points: list[models.PointStruct] = []
    if nodes:
        texts = [n["text"] for n in nodes]
        dense = clients.embedder.embed(texts)
        sparse = clients.sparse.embed(texts)
        points = build_points(nodes, dense, sparse)
        ensure_collection(clients.writer, len(dense[0]))
    if staged:
        clients.writer.replace_docs_points(
            [d["previous_manifest_id"] for d in staged if d["previous_manifest_id"] is not None],
            points,
        )
    ledger = [(d["previous_manifest_id"], ledger_row(d)) for d in staged]
    counts["docs_embedded"] = len(staged)
    counts["points"] = len(points)

    for doc in group:
        if doc["status"] != "skipped_unchanged":
            continue
        previous = doc["previous_manifest_id"]
        if previous is not None and previous != doc["manifest_id"]:
            clients.writer.repoint_doc_points(previous, doc["manifest_id"])
        ledger.append((previous, ledger_row(doc)))
        counts["docs_marked_only"] += 1

    empty = [d for d in group if d["status"] == "skipped_empty"]
    clients.writer.delete_docs_points(
        [d["previous_manifest_id"] for d in empty if d["previous_manifest_id"] is not None]
    )
    ledger += [(d["previous_manifest_id"], ledger_row(d, "empty")) for d in empty]
    counts["docs_empty"] = len(empty)

    return GroupResult(ledger, [d["manifest_id"] for d in staged], counts)


def _run_batch_ids(run_id: str):
    return select(ingest_batches.c.batch_id).where(ingest_batches.c.run_id == run_id)


def load_work(engine: Engine, run_id: str) -> list[dict]:
    """select_work over the run's batch docs, plus each document's live sighting in the ledger.

    A run without batches is not an error: it returns no work.
    """
    run_batches = _run_batch_ids(run_id)
    bd = ingest_batch_docs
    with engine.connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                select(
                    bd.c.manifest_id,
                    bd.c.batch_id,
                    bd.c.status,
                    bd.c.text_sha,
                    bd.c.parser_version,
                    bd.c.node_count,
                ).where(bd.c.batch_id.in_(run_batches))
            ).mappings()
        ]
        if not rows:
            # plan legitimately produces no batches when nothing changed
            return []
        work = select_work(rows)
        # The ledger is per document, keyed by its live sighting, so bridge
        # batch sighting → document (source_system, doc_url) → live sighting →
        # ledger. A subquery rather than an IN list: a backfill run can hold
        # more documents than Postgres allows bind parameters.
        bm = ingest_manifest.alias("bm")  # this run's sightings
        lm = ingest_manifest.alias("lm")  # the ledger's live sightings
        previous = {
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
                .where(bm.c.manifest_id.in_(select(bd.c.manifest_id).where(bd.c.batch_id.in_(run_batches))))
            ).mappings()
        }
    for doc in work:
        doc["previous_manifest_id"] = previous.get(doc["manifest_id"])
    return work


def load_nodes(engine: Engine, manifest_ids: list[int]) -> list[dict]:
    if not manifest_ids:
        return []
    with engine.connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                select(staged_nodes)
                .where(staged_nodes.c.manifest_id.in_(manifest_ids))
                .order_by(staged_nodes.c.manifest_id, staged_nodes.c.node_seq)
            ).mappings()
        ]


def write_group_state(engine: Engine, run_id: str, result: GroupResult) -> None:
    """Ledger rows and 'embedded' marks for one group, in one transaction.

    Marks every 'staged' row of the run for the document, not only the newest
    batch's: finalize treats a batch with any 'staged' row as unfinished.
    """
    with engine.begin() as conn:
        for previous, row in result.ledger:
            _write_ledger(conn, previous, row)
        if result.embedded_ids:
            conn.execute(
                update(ingest_batch_docs)
                .where(
                    ingest_batch_docs.c.manifest_id.in_(result.embedded_ids),
                    ingest_batch_docs.c.status == "staged",
                    ingest_batch_docs.c.batch_id.in_(_run_batch_ids(run_id)),
                )
                .values(status="embedded")
            )


def run_parallel(
    groups: list[list[dict]],
    concurrency: int,
    new_clients: Callable[[], Clients],
    handle: Callable[[Clients, list[dict]], None],
) -> None:
    """Process groups with `concurrency` worker threads, each with its own clients.

    The first error stops the workers from taking more groups; groups already
    running finish, then that error is raised.
    """
    work: queue.SimpleQueue[list[dict]] = queue.SimpleQueue()
    for group in groups:
        work.put(group)
    stop = threading.Event()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker() -> None:
        clients = None
        try:
            clients = new_clients()
            while not stop.is_set():
                try:
                    group = work.get_nowait()
                except queue.Empty:
                    return
                handle(clients, group)
        except BaseException as exc:
            with errors_lock:
                errors.append(exc)
            stop.set()
        finally:
            if clients is not None:
                clients.close()

    threads = [
        threading.Thread(target=worker, name=f"embed-{i}")
        for i in range(min(concurrency, len(groups)))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def run_embed(
    run_id: str,
    engine: Engine | None = None,
    new_clients: Callable[[], Clients] | None = None,
    concurrency: int | None = None,
) -> EmbedRunReport:
    """The embed stage for one run: every parsed document in the run's batches.

    Documents are packed into groups (pack_groups) and processed by
    settings.embed_concurrency worker threads, which follows the number of
    healthy TEI replicas (settings.py). Postgres is written only after a group's
    points landed, so a crash redoes that group. Infra failures (TEI, Qdrant,
    Postgres) fail the run; on the retry, documents already 'embedded' are
    skipped. Returns counts only (the stage contract).
    """
    settings = get_settings()
    concurrency = concurrency or settings.embed_concurrency
    # one connection per worker at most
    engine = engine or get_engine(pool_size=max(5, concurrency))
    new_clients = new_clients or (lambda: make_clients(settings))

    work = load_work(engine, run_id)
    chunks_per_group = group_chunk_target(work, settings.embed_batch_size, concurrency)
    groups = pack_groups(work, chunks_per_group)
    report = EmbedRunReport(run_id=run_id, concurrency=concurrency, groups=len(groups))
    logger.info(
        "embed %s: %d docs in %d groups of <=%d chunks; tei replicas=%s -> concurrency=%d",
        run_id, len(work), len(groups), chunks_per_group, settings.tei_replicas, concurrency,
    )

    guard = CollectionGuard()
    report_lock = threading.Lock()

    def handle(clients: Clients, group: list[dict]) -> None:
        staged_ids = [d["manifest_id"] for d in group if d["status"] == "staged"]
        result = process_group(group, load_nodes(engine, staged_ids), clients, guard)
        write_group_state(engine, run_id, result)
        with report_lock:
            report.add(result.counts)
        logger.info("embed %s: group of %d docs done %s", run_id, len(group), result.counts)

    run_parallel(groups, concurrency, new_clients, handle)
    logger.info("embed %s: %s", run_id, report.counts)
    return report
