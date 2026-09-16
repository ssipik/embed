"""Thin client for the self-hosted TEI embedding service (the TEI service).

TEI (Hugging Face Text Embeddings Inference) runs as a standing Nomad
service (BGE-M3); this client only does REST calls — all embedding compute
lives server-side, so the batch job stays at ~0.5 core. Dense only: TEI's
/embed route can't emit BGE-M3's sparse head, so the sparse leg of hybrid
retrieval comes from a separate in-process BM25 producer (rag_embed.sparse).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

__all__ = ["TeiClient"]

logger = logging.getLogger(__name__)

# Seconds to wait before each retry. Traefik drops a replica for at least one
# Consul check interval (10 s) when its health check fails, so the total wait
# (37 s) outlasts a few such intervals.
RETRY_DELAYS = (2.0, 5.0, 10.0, 20.0)
# Responses from traefik when it could not reach a TEI replica.
RETRY_STATUSES = frozenset({502, 503, 504})
# The connection failed or closed before a response; not timeouts, which already waited tei_timeout.
RETRY_ERRORS = (httpx.ConnectError, httpx.RemoteProtocolError)


class TeiClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        batch_size: int,
        client: httpx.Client | None = None,
        retry_delays: tuple[float, ...] = RETRY_DELAYS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._client = client or httpx.Client(timeout=timeout)
        self._retry_delays = retry_delays
        self._sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in service-friendly sub-batches, preserving order.

        - 502/503/504 or a dropped connection: the sub-batch is sent again after each of retry_delays
        - any other error, or the last failed retry: propagates, so the run fails (infra failure)
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._post_embed(texts[start : start + self.batch_size]))
        return vectors

    def _post_embed(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embed"
        attempts = len(self._retry_delays) + 1
        # None after the last attempt: nothing left to wait for, so raise
        for attempt, delay in enumerate((*self._retry_delays, None), start=1):
            try:
                response = self._client.post(url, json={"inputs": inputs, "truncate": True})
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, *RETRY_ERRORS) as exc:
                retryable = (
                    not isinstance(exc, httpx.HTTPStatusError)
                    or exc.response.status_code in RETRY_STATUSES
                )
                if not retryable or delay is None:
                    logger.error(
                        "TEI %s failed (attempt %d/%d, %d texts): %r",
                        url, attempt, attempts, len(inputs), exc,
                    )
                    raise
                logger.warning(
                    "TEI %s failed (attempt %d/%d, %d texts): %r; retrying in %.0f s",
                    url, attempt, attempts, len(inputs), exc, delay,
                )
                self._sleep(delay)

    def close(self) -> None:
        self._client.close()
