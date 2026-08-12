import httpx
import pytest
from qdrant_client import QdrantClient, models

from rag_embed.embed import build_points, ledger_row
from rag_embed.qdrant import (
    DENSE_VECTOR,
    NODE_SEQ_BITS,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR_DE,
    SPARSE_VECTOR_EN,
    SPARSE_VECTORS,
    QdrantWriter,
    point_id,
)
from rag_embed.tei import TeiClient


def node(mid=1, seq=0, text="hello world"):
    """A staged_nodes row: keyed by the sighting it came from."""
    return {
        "manifest_id": mid,
        "node_seq": seq,
        "text": text,
        "metadata": {
            "manifest_id": mid,
            "source_url": f"https://x/d{mid}",
            "language": "en",
        },
    }


def sparse_vecs(n, fields=SPARSE_VECTORS):
    """n placeholder BM25 sparse vectors (one term each) per sparse field.

    Distinct indices per field so a test can tell the two spaces apart.
    """
    return {
        name: [models.SparseVector(indices=[i + 1], values=[1.0]) for _ in range(n)]
        for i, name in enumerate(fields)
    }


class TestPointIdentity:
    def test_deterministic(self):
        assert point_id(1, 0) == point_id(1, 0)

    def test_distinct_per_chunk_and_sighting(self):
        assert len({point_id(1, 0), point_id(1, 1), point_id(2, 0)}) == 3

    def test_no_overlap_between_adjacent_sightings(self):
        # the last slot of one sighting must not equal the first of the next
        last = point_id(1, (1 << NODE_SEQ_BITS) - 1)
        first_of_next = point_id(2, 0)
        assert last < first_of_next

    def test_node_seq_overflow_raises(self):
        # silently wrapping would collide with the next sighting's ids
        with pytest.raises(ValueError):
            point_id(1, 1 << NODE_SEQ_BITS)


class TestBuildPoints:
    def test_payload_carries_text_and_metadata(self):
        points = build_points([node()], [[0.1, 0.2]], sparse_vecs(1))
        assert points[0].payload["text"] == "hello world"
        assert points[0].payload["language"] == "en"
        assert points[0].payload["node_seq"] == 0

    def test_named_dense_and_sparse_vectors(self):
        points = build_points([node()], [[0.1, 0.2]], sparse_vecs(1))
        assert points[0].vector[DENSE_VECTOR] == [0.1, 0.2]
        assert points[0].vector[SPARSE_VECTOR_DE].indices == [1]
        assert points[0].vector[SPARSE_VECTOR_EN].indices == [2]

    def test_vector_count_mismatch_fails_loudly(self):
        with pytest.raises(ValueError):
            build_points([node()], [], sparse_vecs(0))
        with pytest.raises(ValueError):
            build_points([node()], [[0.1, 0.2]], sparse_vecs(0))

    def test_one_short_sparse_field_fails_loudly(self):
        # a per-field length bug must not silently truncate the batch
        vecs = sparse_vecs(2)
        vecs[SPARSE_VECTOR_EN] = vecs[SPARSE_VECTOR_EN][:1]
        with pytest.raises(ValueError):
            build_points([node(seq=0), node(seq=1)], [[0.1, 0.2]] * 2, vecs)


class TestTeiClient:
    def make(self, handler, batch_size=2):
        transport = httpx.MockTransport(handler)
        return TeiClient(
            "http://tei", 5, batch_size, client=httpx.Client(transport=transport)
        )

    def test_batches_preserve_order(self):
        calls = []

        def handler(request):
            inputs = __import__("json").loads(request.content)["inputs"]
            calls.append(len(inputs))
            return httpx.Response(200, json=[[float(len(t))] for t in inputs])

        tei = self.make(handler, batch_size=2)
        vectors = tei.embed(["a", "bb", "ccc"])
        assert calls == [2, 1]  # sub-batched
        assert vectors == [[1.0], [2.0], [3.0]]

    def test_http_error_propagates(self):
        tei = self.make(lambda request: httpx.Response(503))
        with pytest.raises(httpx.HTTPStatusError):
            tei.embed(["a"])


class TestQdrantWriter:
    """Against qdrant-client's in-memory local mode — real filter/upsert
    semantics, no server."""

    def make(self):
        return QdrantWriter("unused", "test_chunks", client=QdrantClient(":memory:"))

    def test_replace_is_idempotent_and_drops_stale_tail(self):
        w = self.make()
        w.ensure_collection(2)
        five = build_points(
            [node(seq=i) for i in range(5)], [[0.1, 0.2]] * 5, sparse_vecs(5)
        )
        w.replace_doc_points(None, five)  # first ingestion: nothing to delete
        two = build_points(
            [node(seq=i) for i in range(2)], [[0.3, 0.4]] * 2, sparse_vecs(2)
        )
        w.replace_doc_points(1, two)  # re-chunk shrank, same sighting
        assert w._client.count("test_chunks").count == 2

    def test_reembed_leaves_nothing_under_the_previous_sighting(self):
        # the point of deleting on the OLD manifest_id: a new sighting mints
        # new point ids, so the old chunks would otherwise linger as orphans
        w = self.make()
        w.ensure_collection(2)
        w.replace_doc_points(None, build_points([node(mid=1)], [[0.1, 0.2]], sparse_vecs(1)))
        w.replace_doc_points(1, build_points([node(mid=2)], [[0.3, 0.4]], sparse_vecs(1)))
        assert w._client.count("test_chunks").count == 1

    def test_replace_scoped_to_one_doc(self):
        w = self.make()
        w.ensure_collection(2)
        w.replace_doc_points(None, build_points([node(mid=1)], [[0.1, 0.2]], sparse_vecs(1)))
        w.replace_doc_points(None, build_points([node(mid=2)], [[0.1, 0.2]], sparse_vecs(1)))
        w.delete_doc_points(1)
        assert w._client.count("test_chunks").count == 1

    def test_repoint_moves_payload_to_the_new_sighting(self):
        # skipped_unchanged: vectors stay, but the ledger advances, so the
        # payload must follow or a later delete filters on nothing
        w = self.make()
        w.ensure_collection(2)
        w.replace_doc_points(None, build_points([node(mid=1)], [[0.1, 0.2]], sparse_vecs(1)))
        w.repoint_doc_points(1, 2)
        w.delete_doc_points(1)
        assert w._client.count("test_chunks").count == 1
        w.delete_doc_points(2)
        assert w._client.count("test_chunks").count == 0

    def test_ensure_collection_twice_is_noop(self):
        w = self.make()
        w.ensure_collection(2)
        w.ensure_collection(2)

    def test_indexed_payload_fields(self):
        # the retrieval stage filters on these; an unindexed field still filters
        # but by full scan. language is absent on purpose — it is the source's
        # configured constant, not detected, so it is not offered as a filter.
        assert set(PAYLOAD_INDEXES) == {
            "manifest_id",
            "source_system",
            "source_id",
            "doc_type",
            "source_url",
            "published_at",
        }
        assert PAYLOAD_INDEXES["published_at"] is models.PayloadSchemaType.DATETIME

    def test_delete_without_a_collection_is_noop(self):
        # a batch of only skipped_empty docs never calls ensure_collection
        w = self.make()
        w.delete_doc_points(1)

    def test_collection_has_named_dense_and_both_sparse(self):
        w = self.make()
        w.ensure_collection(dense_size=2)
        params = w._client.get_collection("test_chunks").config.params
        assert set(params.vectors) == {DENSE_VECTOR}
        assert set(params.sparse_vectors) == {SPARSE_VECTOR_DE, SPARSE_VECTOR_EN}

    def test_every_sparse_field_applies_idf(self):
        # IDF is per field — Qdrant keeps document-frequency stats separately
        w = self.make()
        w.ensure_collection(dense_size=2)
        sparse = w._client.get_collection("test_chunks").config.params.sparse_vectors
        assert all(
            sparse[name].modifier == models.Modifier.IDF for name in SPARSE_VECTORS
        )


class TestLedgerRow:
    DOC = {"manifest_id": 7, "text_sha": "t1", "parser_version": "v1"}

    def test_points_at_this_batch_sighting(self):
        row = ledger_row(self.DOC)
        assert row["manifest_id"] == 7
        assert row["status"] == "active"
        assert row["text_sha"] == "t1"

    def test_carries_only_state(self):
        # url/title/doc_version are read through manifest_id, not stored here
        assert set(ledger_row(self.DOC)) == {
            "manifest_id",
            "text_sha",
            "parser_version",
            "status",
        }

    def test_empty_status_passes_through(self):
        # a skipped_empty doc gets a ledger row with no vectors behind it
        row = ledger_row(self.DOC, "empty")
        assert row["status"] == "empty"
        assert row["manifest_id"] == 7


class TestBm25Embedder:
    """Conversion of fastembed output → qdrant SparseVector, with a fake model
    injected so the test stays offline (no fastembed download)."""

    def test_converts_fastembed_output_to_sparse_vectors(self):
        from types import SimpleNamespace

        from rag_embed.sparse import Bm25Embedder

        fake = SimpleNamespace(
            embed=lambda texts, batch_size=None: [
                SimpleNamespace(indices=[3, 7], values=[1.5, 2.0]) for _ in texts
            ]
        )
        out = Bm25Embedder("Qdrant/bm25", "german", model=fake).embed(["a", "b"])
        assert len(out) == 2
        assert out[0].indices == [3, 7]
        assert out[0].values == [1.5, 2.0]

    def test_empty_input_returns_empty_without_loading_model(self):
        from rag_embed.sparse import Bm25Embedder

        # model=None and no fastembed call: empty input must short-circuit
        assert Bm25Embedder("Qdrant/bm25", "german").embed([]) == []


class TestBm25Embedders:
    """The per-field container: one analyzer per sparse vector name."""

    def fake(self, index):
        from types import SimpleNamespace

        return SimpleNamespace(
            embed=lambda texts, batch_size=None: [
                SimpleNamespace(indices=[index], values=[1.0]) for _ in texts
            ]
        )

    def make(self):
        from rag_embed.sparse import Bm25Embedder, Bm25Embedders

        return Bm25Embedders(
            {
                name: Bm25Embedder("Qdrant/bm25", lang, model=self.fake(i))
                for i, (name, lang) in enumerate(SPARSE_VECTORS.items())
            }
        )

    def test_embeds_every_text_in_every_field(self):
        out = self.make().embed(["a", "b"])
        assert set(out) == set(SPARSE_VECTORS)
        assert all(len(v) == 2 for v in out.values())

    def test_fields_are_separate_spaces(self):
        # each analyzer stems differently, so the same text lands on different ids
        out = self.make().embed(["a"])
        assert out[SPARSE_VECTOR_DE][0].indices != out[SPARSE_VECTOR_EN][0].indices

    def test_for_fields_builds_one_embedder_per_language(self):
        from rag_embed.sparse import Bm25Embedders

        e = Bm25Embedders.for_fields(SPARSE_VECTORS, "Qdrant/bm25")
        # models stay unloaded (lazy), so this asserts wiring only — offline
        assert {n: emb.language for n, emb in e._embedders.items()} == SPARSE_VECTORS
