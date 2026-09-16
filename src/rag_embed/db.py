from __future__ import annotations

from typing import Iterable, Sequence

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Engine

from rag_embed.settings import get_settings

metadata = MetaData()

# Mirrors ../postgres/sql/001_discover.sql (authoritative). Read side: the
# manifest sighting supplies the change signals copied to the ledger at the
# moment of successful ingestion.
ingest_manifest = Table(
    "ingest_manifest",
    metadata,
    # GENERATED ALWAYS AS IDENTITY in the DDL: never supplied on insert.
    Column("manifest_id", BigInteger, primary_key=True),
    Column("run_id", Text, nullable=False),
    Column("source_system", Text, nullable=False),
    Column("doc_url", Text, nullable=False),
    Column("source_id", Text, nullable=False),
    Column("title", Text),
    Column("published_at", DateTime(timezone=True)),
    Column("doc_version", Text),
    Column("size_bytes", BigInteger),
    Column("extra", JSONB, nullable=False),
    Column("execution_ts", DateTime(timezone=True), nullable=False),
    # the natural key; discover upserts on it so a re-run of one run updates
    # its sighting instead of minting a second manifest_id
    UniqueConstraint("run_id", "source_system", "doc_url"),
)

# Mirrors ../postgres/sql/002_delta.sql. WRITE side here — this stage owns
# ledger writes: a doc becomes "known" only once its vectors actually
# landed in Qdrant (crash-safety, orchestration). rag_delta.mark_ingested() was
# the dev stand-in for exactly this.
ingest_docs = Table(
    "ingest_docs",
    metadata,
    # the sighting whose points are live in Qdrant
    Column("manifest_id", BigInteger, primary_key=True),
    Column("text_sha", Text),
    Column("parser_version", Text),
    Column("status", Text, nullable=False),
)

# Mirrors ../postgres/sql/003_plan.sql — batch state, updated in place here.
ingest_batches = Table(
    "ingest_batches",
    metadata,
    # GENERATED ALWAYS AS IDENTITY in the DDL: plan gets it via RETURNING.
    Column("batch_id", BigInteger, primary_key=True),
    # the run the batch was PLANNED in, not its docs' sighting runs
    Column("run_id", Text, nullable=False),
    Column("doc_count", Integer, nullable=False),
    Column("planned_cost", BigInteger, nullable=False),
)

ingest_batch_docs = Table(
    "ingest_batch_docs",
    metadata,
    # PK is (batch_id, manifest_id): plan is append-only, so a re-listed doc
    # gets a new row on the new batch and earlier batches keep theirs. Every
    # writer must filter on BOTH (see 003_plan.sql).
    Column("manifest_id", BigInteger, primary_key=True),
    Column("batch_id", BigInteger, primary_key=True),
    Column("cost", BigInteger, nullable=False),
    Column("status", Text, nullable=False),
    Column("error", Text),
    Column("content_sha", Text),
    Column("text_sha", Text),
    Column("parser_version", Text),
    Column("node_count", Integer),
    # what the parser could read: pages, chars_per_page, image_coverage,
    # low_text (the parse_chunk stage writes it; see 004_parse_chunk.sql).
    # none_as_null: a Python None must land as SQL NULL, not JSON 'null'.
    Column("parse_quality", JSONB(none_as_null=True)),
)

# Mirrors ../postgres/sql/004_parse_chunk.sql. Read side: the staged chunks
# this stage embeds; finalize deletes them after the run.
staged_nodes = Table(
    "staged_nodes",
    metadata,
    Column("manifest_id", BigInteger, primary_key=True),
    Column("node_seq", Integer, primary_key=True),
    Column("batch_id", BigInteger, nullable=False),
    Column("text", Text, nullable=False),
    Column("metadata", JSONB, nullable=False),
)


def get_engine(dsn: str | None = None, pool_size: int = 5) -> Engine:
    """pool_size: connections kept open; embed sizes it to its worker count."""
    return create_engine(dsn or get_settings().pg_dsn, pool_size=pool_size)


def upsert_rows(
    engine: Engine,
    table: Table,
    rows: Sequence[dict],
    key: Iterable[str],
    preserve: Iterable[str] = (),
) -> int:
    """Idempotent batch upsert; `preserve` columns are write-once (excluded
    from the conflict update — e.g. first_ingested_run_id)."""
    key = list(key)
    skip = set(key) | set(preserve)
    deduped = list({tuple(r[k] for k in key): r for r in rows}.values())
    if not deduped:
        return 0
    stmt = insert(table)
    stmt = stmt.on_conflict_do_update(
        index_elements=key,
        # only columns actually supplied: an identity column (manifest_id,
        # batch_id) is absent from the rows and must not be overwritten
        set_={
            c.name: stmt.excluded[c.name]
            for c in table.columns
            if c.name not in skip and c.name in deduped[0]
        },
    )
    with engine.begin() as conn:
        conn.execute(stmt, deduped)
    return len(deduped)
