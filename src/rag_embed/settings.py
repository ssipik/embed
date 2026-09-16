from __future__ import annotations

import os
from dataclasses import dataclass

from rag_config import load_params, load_settings

DEFAULT_PG_DSN = "postgresql+psycopg://rag:rag@localhost:5433/rag_ingestion"
DEFAULT_USER_AGENT = "rag-pipeline-ingest/0.1"
# Dev defaults for the two standing services this stage talks to (TEI and Qdrant);
# on the cluster these become the Nomad service addresses.
DEFAULT_TEI_URL = "http://localhost:8080"
DEFAULT_QDRANT_URL = "http://localhost:6333"
# The collection name and batch size come from rag-config's settings.yaml, the
# sparse model per leg and avg_len from its parameters.yaml — see qdrant.py,
# which joins the two into SPARSE_LEGS. Shared with the retrieval stage, which is
# the point: both stages must embed the same way, and both must pin the same
# fastembed version (its stemmer + token hash define the sparse id space) — that
# pin stays in each pyproject.

# Workers that send to TEI at the same time, per healthy TEI replica. Measured
# 2026-09-15 (evaluation/tests/bench_tei_embed_concurrency.py): one client leaves
# replicas partly idle behind traefik's uneven spread, three per replica only wait.
DEFAULT_CLIENTS_PER_REPLICA = 2


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


@dataclass(frozen=True)
class Settings:
    pg_dsn: str
    http_timeout: float
    tei_timeout: float
    user_agent: str
    tei_url: str
    qdrant_url: str
    qdrant_collection: str
    embed_batch_size: int
    sparse_avg_len: float
    # healthy TEI replicas, rendered by the Nomad job's template at start; None off the cluster
    tei_replicas: int | None
    clients_per_replica: int
    # manual override of the computed worker count
    embed_concurrency_override: int | None

    @property
    def embed_concurrency(self) -> int:
        """Worker threads for one embed run.

        - RAG_EMBED_CONCURRENCY when set
        - otherwise clients_per_replica × healthy TEI replicas
        - one replica assumed when the count is unknown (local dev) or 0
        """
        if self.embed_concurrency_override is not None:
            return max(1, self.embed_concurrency_override)
        return self.clients_per_replica * max(1, self.tei_replicas or 1)


def get_settings() -> Settings:
    shared = load_settings()
    params = load_params().embed
    return Settings(
        pg_dsn=os.environ.get("RAG_PG_DSN", DEFAULT_PG_DSN),
        http_timeout=float(os.environ.get("RAG_HTTP_TIMEOUT", "30")),
        # Its own knob because one /embed call carries embed_batch_size chunks
        # and TEI on CPU runs them 8 at a time: measured 13-16 s for a batch of
        # long chunks, so the 30 s general timeout aborts a request TEI is still
        # working on. Raising RAG_HTTP_TIMEOUT instead would slow down every
        # other call's failure detection.
        tei_timeout=float(os.environ.get("RAG_TEI_TIMEOUT", "300")),
        user_agent=os.environ.get("RAG_HTTP_USER_AGENT", DEFAULT_USER_AGENT),
        tei_url=os.environ.get("RAG_TEI_URL", DEFAULT_TEI_URL),
        qdrant_url=os.environ.get("RAG_QDRANT_URL", DEFAULT_QDRANT_URL),
        qdrant_collection=os.environ.get(
            "RAG_QDRANT_COLLECTION", shared.qdrant_collection
        ),
        embed_batch_size=int(
            os.environ.get("RAG_EMBED_BATCH_SIZE", shared.embed_batch_size)
        ),
        sparse_avg_len=float(
            os.environ.get("RAG_SPARSE_AVG_LEN", params.sparse_avg_len)
        ),
        tei_replicas=_optional_int("RAG_TEI_REPLICAS"),
        clients_per_replica=int(
            os.environ.get("RAG_EMBED_CLIENTS_PER_REPLICA", DEFAULT_CLIENTS_PER_REPLICA)
        ),
        embed_concurrency_override=_optional_int("RAG_EMBED_CONCURRENCY"),
    )
