"""Thin client for the self-hosted TEI embedding service (the TEI service).

TEI (Hugging Face Text Embeddings Inference) runs as a standing Nomad
service (BGE-M3); this client only does REST calls — all embedding compute
lives server-side, so the batch job stays at ~0.5 core. Dense only: TEI's
/embed route can't emit BGE-M3's sparse head, so the sparse leg of hybrid
retrieval comes from a separate in-process BM25 producer (rag_embed.sparse).
"""

from __future__ import annotations

import httpx

__all__ = ["TeiClient"]


class TeiClient:
    def __init__(
        self, base_url: str, timeout: float, batch_size: int, client: httpx.Client | None = None
    ):
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in service-friendly sub-batches, preserving order.

        Any HTTP error propagates: an unreachable/failing TEI is an infra
        failure — the batch task must fail and retry, not limp on (the DAG
        health-checks TEI before dispatching, the TEI service).
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self._client.post(
                f"{self.base_url}/embed",
                json={"inputs": texts[start : start + self.batch_size], "truncate": True},
            )
            response.raise_for_status()
            vectors.extend(response.json())
        return vectors

    def close(self) -> None:
        self._client.close()
