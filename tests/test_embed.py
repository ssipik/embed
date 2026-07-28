import httpx
import pytest
from qdrant_client import QdrantClient, models

from rag_embed.embed import build_points, ledger_row
from rag_embed.qdrant import (
    DENSE_VECTOR,
    NODE_SEQ_BITS,
    SPARSE_VECTOR,
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


def sparse_vecs(n):
    """n placeholder BM25 sparse vectors (one term each)."""
    return [models.SparseVector(indices=[1], values=[1.0]) for _ in range(n)]


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
        assert points[0].vector[SPARSE_VECTOR].indices == [1]

    def test_vector_count_mismatch_fails_loudly(self):
        with pytest.raises(ValueError):
            build_points([node()], [], sparse_vecs(0))
        with pytest.raises(ValueError):
            build_points([node()], [[0.1, 0.2]], sparse_vecs(0))


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

    def test_collection_has_named_dense_and_sparse(self):
        w = self.make()
        w.ensure_collection(dense_size=2)
        params = w._client.get_collection("test_chunks").config.params
        assert set(params.vectors) == {DENSE_VECTOR}
        assert set(params.sparse_vectors) == {SPARSE_VECTOR}


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
