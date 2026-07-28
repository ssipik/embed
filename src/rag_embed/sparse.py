"""In-process BM25 sparse embedder (fastembed) for the sparse leg of hybrid retrieval.

BM25 is non-neural: tokenize → German stem → drop stopwords → hash to token ids →
term frequencies (length-normalized against avg_len). No IDF here — Qdrant applies IDF
at query time from the collection's document-frequency stats (the sparse field's
Modifier.IDF). Runs on CPU in ~microseconds per chunk, so unlike dense (TEI) there is
no server; fastembed pulls onnxruntime as a dependency but BM25 runs no ONNX model.

Cross-stage contract: the retrieval stage must embed queries with the same model id +
language, and both stages must pin the same fastembed version — its stemmer and token
hash define the sparse id space, so a drift silently breaks ingest/query matching.
"""

from __future__ import annotations

from qdrant_client import models

__all__ = ["Bm25Embedder"]


class Bm25Embedder:
    def __init__(
        self,
        model_id: str,
        language: str,
        avg_len: float = 256.0,
        batch_size: int = 32,
        model=None,
    ):
        self.model_id = model_id
        self.language = language
        self.avg_len = avg_len
        self.batch_size = batch_size
        self._model = model  # lazily loaded on first embed if None (tests inject a fake)

    def _get_model(self):
        if self._model is None:
            # imported here so the module loads without fastembed installed
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                model_name=self.model_id,
                language=self.language,
                avg_len=self.avg_len,
            )
        return self._model

    def embed(self, texts: list[str]) -> list[models.SparseVector]:
        """Document-side BM25 sparse vectors, order-preserving.

        Uses the document/passage path (term freqs + length norm). The query
        path (retrieval stage) is different — its values are term indicators and
        IDF does the weighting server-side.
        """
        if not texts:
            return []
        model = self._get_model()
        return [
            models.SparseVector(
                indices=[int(i) for i in e.indices],
                values=[float(v) for v in e.values],
            )
            for e in model.embed(texts, batch_size=self.batch_size)
        ]

    def close(self) -> None:
        pass
