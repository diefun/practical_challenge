"""Chunk the markdown corpus and index it in Qdrant (dense + BM25 sparse vectors)."""

from __future__ import annotations

import uuid

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import models

from agent.config import ROOT, get_config
from agent.knowledge.rag import get_embedders, get_qdrant

DOCS = ROOT / "data" / "docs"


def load_chunks() -> list[dict]:
    header_split = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2")], strip_headers=False)
    size_split = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = []
    for path in sorted(DOCS.glob("*.md")):
        _, front, body = path.read_text().split("---", 2)
        meta = yaml.safe_load(front)
        for section in size_split.split_documents(header_split.split_text(body)):
            heading = section.metadata.get("h2") or section.metadata.get("h1", "")
            chunks.append({
                "text": f"{meta['title']} — {heading}\n{section.page_content}",
                "source": path.name,
                "title": meta["title"],
                "doc_type": meta.get("doc_type"),
                "client": meta.get("client"),
                "project_id": meta.get("project_id"),
                "section": heading,
            })
    return chunks


def build() -> int:
    cfg = get_config()["qdrant"]
    client = get_qdrant()
    dense, sparse = get_embedders()
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    dense_vecs = list(dense.embed(texts))
    sparse_vecs = list(sparse.embed(texts))

    if client.collection_exists(cfg["collection"]):
        client.delete_collection(cfg["collection"])
    client.create_collection(
        cfg["collection"],
        vectors_config={"dense": models.VectorParams(size=len(dense_vecs[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upsert(cfg["collection"], points=[
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['source']}#{i}")),
            vector={
                "dense": d.tolist(),
                "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
            },
            payload=c,
        )
        for i, (c, d, s) in enumerate(zip(chunks, dense_vecs, sparse_vecs))
    ])
    return len(chunks)


if __name__ == "__main__":
    print(f"indexed {build()} chunks")
