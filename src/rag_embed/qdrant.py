"""Thin writer for the Qdrant vector store.

Points are keyed by manifest_id — the sighting whose chunks are live. Across
the active corpus that is unique per document, because ingest_docs.manifest_id
names exactly one sighting per document.

Re-embedding therefore has two ids in play: the ledger's current manifest_id,
whose points must be deleted, and the new sighting's, under which the fresh
points are written. Deleting on the new id would be a no-op and would leave
the old chunks behind as orphans.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from rag_config import load_settings, sparse_legs

__all__ = [
    "QdrantWriter",
    "point_id",
    "NODE_SEQ_BITS",
    "DENSE_VECTOR",
    "SPARSE_VECTOR_DE",
    "SPARSE_VECTOR_EN",
    "SPARSE_VECTORS",
    "SPARSE_LEGS",
    "PAYLOAD_INDEXES",
    "doc_filter",
    "docs_filter",
]

# chunk slots per sighting: 2**20 = 1,048,576
NODE_SEQ_BITS = 20

# Named-vector ids from rag-config's settings.yaml, and the model embedding each
# leg from its parameters.yaml — this stage writes them, the retrieval stage
# queries them, and they used to be two copies of the same constants that
# nothing checked. sparse_legs() joins the two halves and refuses to run if they
# disagree; the reasoning behind each lives in rag_config.params.
_settings = load_settings()
SPARSE_LEGS = sparse_legs(settings=_settings)
DENSE_VECTOR = _settings.dense_vector
# {Qdrant field: its BM25 analyzer} — the shape the collection is built from.
SPARSE_VECTORS: dict[str, str] = {
    leg.vector: language for language, leg in SPARSE_LEGS.items()
}
SPARSE_VECTOR_DE = SPARSE_LEGS["german"].vector
SPARSE_VECTOR_EN = SPARSE_LEGS["english"].vector

# Payload fields that carry an index, and its type. Unindexed payload still
# filters, but by full scan — these are the fields something filters on hot:
#  - manifest_id: this stage's delete-by-doc filter
#  - source_url: the retrieval stage's fetch_document, keyed on document
#    identity rather than the mutable manifest_id
#  - the rest: the agent's search filters (block 8)
# language is deliberately absent — it is the source's configured constant, not
# detected, so it is not offered as a filter (../masterplan/ARCHITECTURE.md
# block 7).
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "manifest_id": models.PayloadSchemaType.INTEGER,
    "source_system": models.PayloadSchemaType.KEYWORD,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "source_url": models.PayloadSchemaType.KEYWORD,
    "published_at": models.PayloadSchemaType.DATETIME,
}


def point_id(manifest_id: int, node_seq: int) -> int:
    """Deterministic unsigned-int point id, so re-embedding one sighting
    overwrites its own points rather than duplicating them."""
    if not 0 <= node_seq < (1 << NODE_SEQ_BITS):
        raise ValueError(
            f"node_seq {node_seq} outside the {1 << NODE_SEQ_BITS} chunk slots "
            f"per document — ids would collide with the next manifest_id"
        )
    return (manifest_id << NODE_SEQ_BITS) | node_seq


def doc_filter(manifest_id: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="manifest_id", match=models.MatchValue(value=manifest_id)
            )
        ]
    )


def docs_filter(manifest_ids: list[int]) -> models.Filter:
    """Points of any of these sightings — one delete for a whole group."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="manifest_id", match=models.MatchAny(any=list(manifest_ids))
            )
        ]
    )


class QdrantWriter:
    def __init__(self, url: str, collection: str, client: QdrantClient | None = None):
        self.collection = collection
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self, dense_size: int) -> None:
        """Create the collection (+ PAYLOAD_INDEXES) on first contact; later
        calls are no-ops — so adding an index here only reaches an existing
        collection after it is dropped and re-ingested.

        Named vectors: dense (BGE-M3 via TEI, cosine) + one sparse field per
        BM25 analyzer (SPARSE_VECTORS). Sparse fields have no size — they are
        dimensionless; Modifier.IDF makes Qdrant apply inverse-document-frequency
        at query time over the stored term freqs, per field.
        """
        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dense_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                for name in SPARSE_VECTORS
            },
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self._client.create_payload_index(self.collection, field, schema)

    def replace_doc_points(
        self,
        previous_manifest_id: int | None,
        points: list[models.PointStruct],
    ) -> None:
        """Drop the document's live points, then write the new sighting's.

        previous_manifest_id is the ledger's — None for a first ingestion,
        where there is nothing to delete.
        """
        previous = [] if previous_manifest_id is None else [previous_manifest_id]
        self.replace_docs_points(previous, points)

    def replace_docs_points(
        self, previous_manifest_ids: list[int], points: list[models.PointStruct]
    ) -> None:
        """replace_doc_points for a group of documents: one delete, one upsert.

        previous_manifest_ids are the ledger's live sightings of the group's
        documents (first ingestions have none).
        """
        self.delete_docs_points(previous_manifest_ids)
        if points:
            self._client.upsert(self.collection, points=points, wait=True)

    def delete_doc_points(self, manifest_id: int) -> None:
        self.delete_docs_points([manifest_id])

    def delete_docs_points(self, manifest_ids: list[int]) -> None:
        # a group of only empty docs never calls ensure_collection, so there
        # may be no collection yet — and then nothing to delete
        if not manifest_ids or not self._client.collection_exists(self.collection):
            return
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=docs_filter(manifest_ids)),
            wait=True,
        )

    def repoint_doc_points(self, previous_manifest_id: int, manifest_id: int) -> None:
        """Move a document's existing points onto a newer sighting.

        Needed for skipped_unchanged: the text did not change, so the vectors
        stay, but the ledger must advance to the new sighting or delta keeps
        reading the old doc_version and re-triggers 'changed' forever. The
        payload has to follow, otherwise a later delete would filter on a
        manifest_id no point carries.
        """
        if not self._client.collection_exists(self.collection):
            return
        self._client.set_payload(
            self.collection,
            payload={"manifest_id": manifest_id},
            points=models.FilterSelector(
                filter=doc_filter(previous_manifest_id)
            ).filter,
            wait=True,
        )

    def close(self) -> None:
        self._client.close()
