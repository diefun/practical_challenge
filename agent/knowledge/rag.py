"""RAG retrieval: hybrid (dense + BM25) search over the document corpus in Qdrant."""

from __future__ import annotations

import atexit
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from agent.config import get_config


@lru_cache
def get_qdrant() -> QdrantClient:
    cfg = get_config()["qdrant"]
    client = QdrantClient(url=cfg["url"]) if cfg["url"] else QdrantClient(path=cfg["path"])
    atexit.register(client.close)
    return client


@lru_cache
def get_embedders() -> tuple[TextEmbedding, SparseTextEmbedding]:
    cfg = get_config()["qdrant"]
    return TextEmbedding(cfg["dense_model"]), SparseTextEmbedding(cfg["sparse_model"])


def search(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top document chunks for a question, fused with Reciprocal Rank Fusion."""
    cfg = get_config()["qdrant"]
    k = top_k or cfg["top_k"]
    dense, sparse = get_embedders()
    dq = next(iter(dense.query_embed(question))).tolist()
    sq = next(iter(sparse.query_embed(question)))
    res = get_qdrant().query_points(
        cfg["collection"],
        prefetch=[
            models.Prefetch(query=dq, using="dense", limit=k * 3),
            models.Prefetch(
                query=models.SparseVector(indices=sq.indices.tolist(), values=sq.values.tolist()),
                using="bm25",
                limit=k * 3,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [{**p.payload, "score": round(p.score, 4)} for p in res.points]
