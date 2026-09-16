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
    SPARSE_LEGS,
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
    def make(self, handler, batch_size=2, sleeps=None):
        transport = httpx.MockTransport(handler)
        return TeiClient(
            "http://tei",
            5,
            batch_size,
            client=httpx.Client(transport=transport),
            retry_delays=(1.0, 2.0),
            sleep=(sleeps if sleeps is not None else []).append,
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

    def test_http_error_propagates_after_retries(self):
        calls, sleeps = [], []

        def handler(request):
            calls.append(1)
            return httpx.Response(503)

        tei = self.make(handler, sleeps=sleeps)
        with pytest.raises(httpx.HTTPStatusError):
            tei.embed(["a"])
        assert len(calls) == 3
        assert sleeps == [1.0, 2.0]

    @pytest.mark.parametrize("status", [404, 502])
    def test_gateway_error_then_success_is_retried(self, status, caplog):
        responses = iter([httpx.Response(status), httpx.Response(200, json=[[1.0]])])
        sleeps = []
        tei = self.make(lambda request: next(responses), sleeps=sleeps)
        assert tei.embed(["a"]) == [[1.0]]
        assert sleeps == [1.0]
        assert "retrying in 1 s" in caplog.text

    def test_dropped_connection_is_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json=[[1.0]])

        assert self.make(handler).embed(["a"]) == [[1.0]]
        assert len(calls) == 2

    def test_client_error_is_not_retried(self):
        calls, sleeps = [], []

        def handler(request):
            calls.append(1)
            return httpx.Response(413)

        tei = self.make(handler, sleeps=sleeps)
        with pytest.raises(httpx.HTTPStatusError):
            tei.embed(["a"])
        assert len(calls) == 1
        assert sleeps == []


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

    def test_for_legs_builds_one_embedder_per_language(self):
        from rag_embed.sparse import Bm25Embedders

        e = Bm25Embedders.for_legs(SPARSE_LEGS)
        # models stay unloaded (lazy), so this asserts wiring only — offline
        assert {n: emb.language for n, emb in e._embedders.items()} == SPARSE_VECTORS


    def test_each_leg_embeds_with_its_own_model(self):
        """German is expected to diverge from `Qdrant/bm25` (decompounding), so
        the model has to come from the leg rather than one shared setting."""
        from rag_config import SparseLeg
        from rag_embed.sparse import Bm25Embedders

        legs = {
            "german": SparseLeg("german", "sparse_bm25_de", "some/german-model"),
            "english": SparseLeg("english", "sparse_bm25_en", "Qdrant/bm25"),
        }
        e = Bm25Embedders.for_legs(legs)
        assert e._embedders["sparse_bm25_de"].model_id == "some/german-model"
        assert e._embedders["sparse_bm25_en"].model_id == "Qdrant/bm25"


# --- run-level embed: work selection, grouping, group processing, workers ---

import json
import threading
import time

from rag_embed.embed import (
    Clients,
    CollectionGuard,
    pack_groups,
    process_group,
    run_parallel,
    select_work,
)
from rag_embed.settings import Settings


def batch_doc(mid, batch_id=1, status="staged", node_count=1, previous=None):
    return {
        "manifest_id": mid,
        "batch_id": batch_id,
        "status": status,
        "text_sha": f"t{mid}",
        "parser_version": "v1",
        "node_count": node_count,
        "previous_manifest_id": previous,
    }


class TestSelectWork:
    def test_newest_batch_row_is_the_documents_state(self):
        # plan is append-only: a re-planned doc has one row per batch
        rows = [batch_doc(1, batch_id=1, status="embedded"), batch_doc(1, batch_id=2, status="staged")]
        assert [(r["manifest_id"], r["batch_id"]) for r in select_work(rows)] == [(1, 2)]

    def test_pending_on_the_newest_row_raises(self):
        with pytest.raises(ValueError, match="not parsed"):
            select_work([batch_doc(1, status="pending")])

    def test_pending_on_an_older_row_is_ignored(self):
        rows = [batch_doc(1, batch_id=1, status="pending"), batch_doc(1, batch_id=2, status="staged")]
        assert len(select_work(rows)) == 1

    def test_done_and_failed_docs_are_not_work(self):
        rows = [
            batch_doc(1, status="embedded"),
            batch_doc(2, status="error"),
            batch_doc(3, status="skipped_unchanged"),
            batch_doc(4, status="skipped_empty"),
        ]
        assert [r["manifest_id"] for r in select_work(rows)] == [3, 4]


class TestPackGroups:
    def test_small_docs_fill_one_request(self):
        docs = [batch_doc(i, node_count=10) for i in range(4)]  # 40 chunks
        assert [[d["manifest_id"] for d in g] for g in pack_groups(docs, 32)] == [[0, 1, 2], [3]]

    def test_a_large_doc_gets_its_own_group(self):
        docs = [batch_doc(1, node_count=5), batch_doc(2, node_count=100), batch_doc(3, node_count=5)]
        assert [[d["manifest_id"] for d in g] for g in pack_groups(docs, 32)] == [[1], [2], [3]]

    def test_skipped_docs_are_grouped_apart_and_capped(self):
        docs = [batch_doc(i, status="skipped_unchanged", node_count=None) for i in range(5)]
        docs.append(batch_doc(9, node_count=3))
        groups = pack_groups(docs, 32, max_docs=2)
        assert [[d["manifest_id"] for d in g] for g in groups] == [[9], [0, 1], [2, 3], [4]]

    def test_doc_cap_applies_to_one_chunk_docs(self):
        docs = [batch_doc(i, node_count=1) for i in range(5)]
        assert [len(g) for g in pack_groups(docs, 32, max_docs=2)] == [2, 2, 1]

    def test_unknown_chunk_count_does_not_overfill_a_group(self):
        docs = [batch_doc(1, node_count=5), batch_doc(2, node_count=None)]
        assert len(pack_groups(docs, 32)) == 2

    def test_every_doc_lands_in_exactly_one_group(self):
        docs = [batch_doc(i, node_count=(i * 7) % 40 + 1) for i in range(50)]
        docs += [batch_doc(100 + i, status="skipped_empty") for i in range(7)]
        ids = [d["manifest_id"] for g in pack_groups(docs, 32) for d in g]
        assert sorted(ids) == sorted(d["manifest_id"] for d in docs)


class FakeSparse:
    def embed(self, texts):
        return sparse_vecs(len(texts))

    def close(self):
        pass


def tei_by_text_length(calls=None):
    """Mock TEI whose vector encodes the text, so a test can check vectors land on the right point."""

    def handler(request):
        inputs = json.loads(request.content)["inputs"]
        if calls is not None:
            calls.append(len(inputs))
        return httpx.Response(200, json=[[float(len(t)), 1.0] for t in inputs])

    return TeiClient("http://tei", 5, 32, client=httpx.Client(transport=httpx.MockTransport(handler)))


class TestProcessGroup:
    def clients(self, calls=None):
        writer = QdrantWriter("unused", "test_chunks", client=QdrantClient(":memory:"))
        return Clients(tei_by_text_length(calls), FakeSparse(), writer)

    def count(self, clients, mid=None):
        from rag_embed.qdrant import doc_filter

        flt = None if mid is None else doc_filter(mid)
        return clients.writer._client.count("test_chunks", count_filter=flt).count

    def test_mixed_group_maps_vectors_back_to_each_document(self):
        calls = []
        c = self.clients(calls)
        group = [batch_doc(1, node_count=2), batch_doc(2, node_count=1)]
        nodes = [node(1, 0, "a"), node(1, 1, "bb"), node(2, 0, "cccc")]
        result = process_group(group, nodes, c, CollectionGuard())
        assert calls == [3]  # one TEI request for the whole group
        # cosine collections store normalized vectors; the ratio still encodes the text length
        got = {
            p.id: round(p.vector[DENSE_VECTOR][0] / p.vector[DENSE_VECTOR][1], 3)
            for p in c.writer._client.retrieve(
                "test_chunks", [point_id(1, 0), point_id(1, 1), point_id(2, 0)], with_vectors=True
            )
        }
        assert got == {point_id(1, 0): 1.0, point_id(1, 1): 2.0, point_id(2, 0): 4.0}
        assert result.embedded_ids == [1, 2]
        assert result.counts == {"docs_embedded": 2, "docs_marked_only": 0, "docs_empty": 0, "points": 3}
        assert [row["status"] for _, row in result.ledger] == ["active", "active"]

    def test_new_sighting_replaces_the_previous_one(self):
        c = self.clients()
        guard = CollectionGuard()
        process_group([batch_doc(1)], [node(1, 0)], c, guard)
        process_group([batch_doc(5, previous=1)], [node(5, 0)], c, guard)
        assert self.count(c, 1) == 0
        assert self.count(c, 5) == 1

    def test_skipped_docs_repoint_and_delete(self):
        c = self.clients()
        guard = CollectionGuard()
        process_group([batch_doc(1), batch_doc(2)], [node(1, 0), node(2, 0)], c, guard)
        group = [
            batch_doc(3, status="skipped_unchanged", previous=1),
            batch_doc(4, status="skipped_empty", previous=2),
        ]
        result = process_group(group, [], c, guard)
        assert self.count(c, 3) == 1 and self.count(c, 1) == 0  # moved
        assert self.count(c, 2) == 0 and self.count(c, 4) == 0  # deleted
        assert result.embedded_ids == []
        assert [(prev, row["status"]) for prev, row in result.ledger] == [(1, "active"), (2, "empty")]

    def test_retry_after_a_failed_upsert_leaves_each_point_once(self):
        # Postgres is only written after the points landed, so a failure redoes the group
        c = self.clients()
        guard = CollectionGuard()
        group = [batch_doc(1, node_count=2), batch_doc(2)]
        nodes = [node(1, 0), node(1, 1), node(2, 0)]
        real_upsert = c.writer._client.upsert
        c.writer._client.upsert = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("qdrant down"))
        with pytest.raises(RuntimeError):
            process_group(group, nodes, c, guard)
        c.writer._client.upsert = real_upsert
        process_group(group, nodes, c, guard)
        assert self.count(c) == 3


class TestRunParallel:
    class FakeClients:
        def __init__(self, closed):
            self.closed = closed

        def close(self):
            self.closed.append(1)

    def test_every_group_is_processed_and_clients_closed(self):
        done, closed, lock = [], [], threading.Lock()

        def handle(clients, group):
            with lock:
                done.append(group[0])

        run_parallel([[i] for i in range(20)], 4, lambda: self.FakeClients(closed), handle)
        assert sorted(done) == list(range(20))
        assert len(closed) == 4

    def test_workers_run_at_the_same_time_up_to_the_limit(self):
        active, peak, lock = [0], [0], threading.Lock()

        def handle(clients, group):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.02)
            with lock:
                active[0] -= 1

        run_parallel([[i] for i in range(12)], 3, lambda: self.FakeClients([]), handle)
        assert peak[0] == 3

    def test_no_more_workers_than_groups(self):
        closed = []
        run_parallel([[1]], 8, lambda: self.FakeClients(closed), lambda c, g: None)
        assert len(closed) == 1

    def test_first_error_stops_new_groups_and_is_raised(self):
        started, lock = [], threading.Lock()

        def handle(clients, group):
            with lock:
                started.append(group[0])
            if group[0] == 0:
                raise RuntimeError("tei down")

        with pytest.raises(RuntimeError, match="tei down"):
            run_parallel([[i] for i in range(100)], 1, lambda: self.FakeClients([]), handle)
        assert started == [0]


class TestEmbedConcurrency:
    def settings(self, replicas=None, per_replica=2, override=None):
        return Settings(
            pg_dsn="", http_timeout=30, tei_timeout=300, user_agent="", tei_url="",
            qdrant_url="", qdrant_collection="c", embed_batch_size=32, sparse_avg_len=256.0,
            tei_replicas=replicas, clients_per_replica=per_replica,
            embed_concurrency_override=override,
        )

    def test_follows_the_healthy_replicas(self):
        assert self.settings(replicas=4).embed_concurrency == 8

    def test_unknown_or_zero_replicas_count_as_one(self):
        assert self.settings(replicas=None).embed_concurrency == 2
        assert self.settings(replicas=0).embed_concurrency == 2

    def test_override_wins(self):
        assert self.settings(replicas=4, override=3).embed_concurrency == 3

    def test_env_vars_are_read(self, monkeypatch):
        from rag_embed.settings import get_settings

        monkeypatch.setenv("RAG_TEI_REPLICAS", "3")
        monkeypatch.setenv("RAG_EMBED_CLIENTS_PER_REPLICA", "1")
        monkeypatch.delenv("RAG_EMBED_CONCURRENCY", raising=False)
        assert get_settings().embed_concurrency == 3


class TestRunWithoutBatches:
    """plan produces no batches when nothing changed; embed must succeed with nothing to do."""

    class EmptyEngine:
        def connect(self):
            from contextlib import contextmanager

            @contextmanager
            def conn():
                class Result:
                    def mappings(self):
                        return []

                class Conn:
                    def execute(self, stmt):
                        return Result()

                yield Conn()

            return conn()

    def test_no_batches_is_no_work(self):
        from rag_embed.embed import load_work

        assert load_work(self.EmptyEngine(), "quiet-day") == []

    def test_run_reports_zero_counts(self, monkeypatch):
        import rag_embed.embed as embed

        report = embed.run_embed_run(
            "quiet-day",
            engine=self.EmptyEngine(),
            new_clients=lambda: pytest.fail("no worker should start"),
            concurrency=2,
        )
        assert report.groups == 0
        assert report.counts == {"docs_embedded": 0, "docs_marked_only": 0, "docs_empty": 0, "points": 0}
