from __future__ import annotations

import os
from dataclasses import dataclass

from rag_config import load_params

DEFAULT_PG_DSN = "postgresql+psycopg://rag:rag@localhost:5433/rag_ingestion"
DEFAULT_USER_AGENT = "rag-pipeline-ingest/0.1"
# Dev defaults for the two standing services this stage talks to (TEI and Qdrant);
# on the cluster these become the Nomad service addresses.
DEFAULT_TEI_URL = "http://localhost:8080"
DEFAULT_QDRANT_URL = "http://localhost:6333"
# The collection name, batch size and the sparse (BM25) settings come from
# rag-config's parameters.yaml. The sparse model id and avg_len are shared with
# the retrieval stage there, which is the point: both stages must embed the same
# way, and both must pin the same fastembed version (its stemmer + token hash
# define the sparse id space) — that pin stays in each pyproject.


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
    sparse_model_id: str
    sparse_avg_len: float


def get_settings() -> Settings:
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
            "RAG_QDRANT_COLLECTION", params.qdrant_collection
        ),
        embed_batch_size=int(
            os.environ.get("RAG_EMBED_BATCH_SIZE", params.batch_size)
        ),
        sparse_model_id=os.environ.get("RAG_SPARSE_MODEL_ID", params.sparse_model_id),
        sparse_avg_len=float(
            os.environ.get("RAG_SPARSE_AVG_LEN", params.sparse_avg_len)
        ),
    )
