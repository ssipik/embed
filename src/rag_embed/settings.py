from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_PG_DSN = "postgresql+psycopg://rag:rag@localhost:5433/rag_ingestion"
DEFAULT_USER_AGENT = "rag-pipeline-ingest/0.1"
# Dev defaults for the two standing services this stage talks to (TEI and Qdrant);
# on the cluster these become the Nomad service addresses.
DEFAULT_TEI_URL = "http://localhost:8080"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "rag_chunks"
DEFAULT_EMBED_BATCH_SIZE = 32  # chunks per TEI request; server does the real batching
# Sparse (BM25) producer — fastembed, in-process. The model id is a cross-stage contract:
# the retrieval stage must embed queries the same way, and both stages must pin the same
# fastembed version (its stemmer + token hash define the sparse id space). avg_len is
# BM25's length-normalization reference in tokens (not a vector size); tune to the chunk
# size once. The analyzer languages are not settings — they name the collection's sparse
# fields, so they live with the rest of that contract in qdrant.SPARSE_VECTORS.
DEFAULT_SPARSE_MODEL_ID = "Qdrant/bm25"
DEFAULT_SPARSE_AVG_LEN = 256.0


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
            "RAG_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION
        ),
        embed_batch_size=int(
            os.environ.get("RAG_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE)
        ),
        sparse_model_id=os.environ.get("RAG_SPARSE_MODEL_ID", DEFAULT_SPARSE_MODEL_ID),
        sparse_avg_len=float(
            os.environ.get("RAG_SPARSE_AVG_LEN", DEFAULT_SPARSE_AVG_LEN)
        ),
    )
