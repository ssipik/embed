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

__all__ = [
    "QdrantWriter",
    "point_id",
    "NODE_SEQ_BITS",
    "DENSE_VECTOR",
    "SPARSE_VECTOR_DE",
    "SPARSE_VECTOR_EN",
    "SPARSE_VECTORS",
]

# chunk slots per sighting: 2**20 = 1,048,576
NODE_SEQ_BITS = 20

# Named-vector ids — model-specific so a second model can be added without
# ambiguity (e.g. dense_e5, sparse_splade). Cross-stage contract: the retrieval
# stage must query these exact names.
DENSE_VECTOR = "dense_bge_m3"
SPARSE_VECTOR_DE = "sparse_bm25_de"
SPARSE_VECTOR_EN = "sparse_bm25_en"

# Sparse field → BM25 analyzer language. Every chunk gets both, whatever language
# it is in: the analyzer defines the sparse id space, so picking one per document
# would make matching depend on an ingest-time language guess. Each field keeps its
# own document-frequency stats, hence its own IDF.
SPARSE_VECTORS: dict[str, str] = {
    SPARSE_VECTOR_DE: "german",
    SPARSE_VECTOR_EN: "english",
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


class QdrantWriter:
    def __init__(self, url: str, collection: str, client: QdrantClient | None = None):
        self.collection = collection
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self, dense_size: int) -> None:
        """Create the collection (+ the payload index deletion filtering
        needs) on first contact; later calls are no-ops.

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
        self._client.create_payload_index(
            self.collection, "manifest_id", models.PayloadSchemaType.INTEGER
        )

    def replace_doc_points(
        self,
        previous_manifest_id: int | None,
        points: list[models.PointStruct],
    ) -> None:
        """Drop the document's live points, then write the new sighting's.

        previous_manifest_id is the ledger's — None for a first ingestion,
        where there is nothing to delete.
        """
        if previous_manifest_id is not None:
            self.delete_doc_points(previous_manifest_id)
        if points:
            self._client.upsert(self.collection, points=points, wait=True)

    def delete_doc_points(self, manifest_id: int) -> None:
        self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=doc_filter(manifest_id)),
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
