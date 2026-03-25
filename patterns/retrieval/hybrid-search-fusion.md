# Hybrid Search Fusion

> Combine multiple retrieval strategies (keyword, semantic, graph) and fuse their ranked results into a single, superior ranking.

## Problem

No single retrieval method works well for all queries. Keyword search (BM25) excels at exact term matching -- product IDs, error codes, proper nouns -- but misses paraphrases and conceptual matches. Semantic search (embedding similarity) captures meaning but struggles with rare terms, acronyms, and precise identifiers. Knowledge graph traversal finds relationship-based answers but requires structured data and misses free-text nuances.

When you rely on a single retrieval strategy, you get systematic blind spots. A user searching for "OOM error in k8s pod" needs keyword precision for "OOM" and "k8s" combined with semantic understanding that they mean "out of memory in Kubernetes." Without fusion, you either miss the exact term matches or the conceptual ones.

## Solution

Hybrid Search Fusion runs multiple retrieval strategies in parallel against the same query, then merges their result lists into a single ranking using a score fusion algorithm. The most common fusion method is **Reciprocal Rank Fusion (RRF)**, which combines rankings without requiring score normalization across heterogeneous retrievers.

Each retriever contributes its own ranked list. RRF assigns each document a score based on its rank position across all lists: documents that appear near the top of multiple lists get boosted, while documents that appear in only one list still contribute. The result is a fused ranking that inherits the strengths of each retriever while compensating for individual weaknesses.

## How It Works

```
                          User Query
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
      +-------------+  +-------------+  +-------------+
      |  BM25/      |  |  Embedding  |  |  Knowledge  |
      |  Keyword    |  |  Semantic   |  |  Graph      |
      |  Search     |  |  Search     |  |  Traversal  |
      +-------------+  +-------------+  +-------------+
              |               |               |
              v               v               v
      [ rank list A ]  [ rank list B ]  [ rank list C ]
              |               |               |
              +---------------+---------------+
                              |
                              v
                   +--------------------+
                   | Reciprocal Rank    |
                   | Fusion (RRF)       |
                   |                    |
                   | score(d) = SUM     |
                   |  1 / (k + rank_i)  |
                   +--------------------+
                              |
                              v
                   +--------------------+
                   | Fused ranked list  |
                   | (top-N results)    |
                   +--------------------+
```

1. **Parse the query** -- Optionally extract keywords and entities for the keyword/graph retrievers while passing the full query to the semantic retriever.
2. **Dispatch retrievers in parallel** -- Each retriever runs independently and returns its own ranked list.
3. **Apply Reciprocal Rank Fusion** -- For each document across all lists, compute `score(d) = SUM(1 / (k + rank_i))` where `k` is a constant (typically 60) and `rank_i` is the document's rank in list `i`. Documents not present in a list are skipped.
4. **Sort by fused score** -- The combined score produces the final ranking.
5. **Return top-N** -- Truncate to the desired number of results.

## Implementation

### Python

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved document or chunk."""

    doc_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    """Interface for any retrieval strategy."""

    async def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        ...


@dataclass(frozen=True)
class FusedResult:
    """A chunk with its fused score and per-retriever rank breakdown."""

    chunk: RetrievedChunk
    fused_score: float
    rank_contributions: dict[str, int]  # retriever_name -> rank


class HybridSearchFuser:
    """
    Runs multiple retrievers in parallel and fuses their results
    using Reciprocal Rank Fusion (RRF).

    RRF score for document d:
        score(d) = SUM(1 / (k + rank_i(d))) for each retriever i

    where k is a smoothing constant (default 60) that prevents
    top-ranked documents from dominating excessively.
    """

    def __init__(
        self,
        retrievers: dict[str, Retriever],
        k: int = 60,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._k = k
        # Optional per-retriever weights (default: equal weight = 1.0)
        self._weights = weights or {name: 1.0 for name in retrievers}

    async def fuse(self, query: str, top_k: int = 10, per_retriever_k: int = 20) -> list[FusedResult]:
        """Run all retrievers in parallel and fuse their results."""

        # --- 1. Dispatch all retrievers concurrently ---
        retriever_names = list(self._retrievers.keys())
        tasks = [
            self._retrievers[name].retrieve(query, top_k=per_retriever_k)
            for name in retriever_names
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- 2. Collect per-retriever ranked lists ---
        # Map doc_id -> (best chunk seen, {retriever -> rank})
        doc_chunks: dict[str, RetrievedChunk] = {}
        doc_ranks: dict[str, dict[str, int]] = {}

        for retriever_name, results in zip(retriever_names, all_results):
            if isinstance(results, BaseException):
                # One retriever failing should not break the entire fusion.
                continue
            for rank, chunk in enumerate(results, start=1):
                if chunk.doc_id not in doc_chunks:
                    doc_chunks[chunk.doc_id] = chunk
                    doc_ranks[chunk.doc_id] = {}
                doc_ranks[chunk.doc_id][retriever_name] = rank

        # --- 3. Compute RRF scores ---
        scored: list[FusedResult] = []
        for doc_id, chunk in doc_chunks.items():
            ranks = doc_ranks[doc_id]
            fused_score = sum(
                self._weights.get(retriever, 1.0) * (1.0 / (self._k + rank))
                for retriever, rank in ranks.items()
            )
            scored.append(FusedResult(
                chunk=chunk,
                fused_score=fused_score,
                rank_contributions=ranks,
            ))

        # --- 4. Sort descending by fused score and return top-K ---
        scored.sort(key=lambda r: r.fused_score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Example retriever implementations
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Keyword search using BM25 (e.g., Elasticsearch, Meilisearch)."""

    def __init__(self, index_client) -> None:
        self._client = index_client

    async def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        # Replace with your actual BM25 client call.
        hits = await self._client.search(query, limit=top_k)
        return [
            RetrievedChunk(
                doc_id=hit["id"],
                content=hit["text"],
                source="bm25",
                metadata={"bm25_score": hit["score"]},
            )
            for hit in hits
        ]


class EmbeddingRetriever:
    """Semantic search using vector similarity (e.g., Pinecone, Weaviate)."""

    def __init__(self, vector_store, embed_fn) -> None:
        self._store = vector_store
        self._embed = embed_fn

    async def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        query_vector = await self._embed(query)
        results = await self._store.query(vector=query_vector, top_k=top_k)
        return [
            RetrievedChunk(
                doc_id=r["id"],
                content=r["text"],
                source="embedding",
                metadata={"similarity": r["score"]},
            )
            for r in results
        ]


class GraphRetriever:
    """Knowledge graph traversal (e.g., Neo4j, Amazon Neptune)."""

    def __init__(self, graph_client, entity_extractor) -> None:
        self._graph = graph_client
        self._extract = entity_extractor

    async def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        entities = await self._extract(query)
        if not entities:
            return []
        neighbors = await self._graph.get_neighbors(entities, depth=2, limit=top_k)
        return [
            RetrievedChunk(
                doc_id=n["id"],
                content=n["description"],
                source="graph",
                metadata={"entity": n["entity"], "relation": n["relation"]},
            )
            for n in neighbors
        ]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

async def main():
    # Wire up your actual clients here.
    fuser = HybridSearchFuser(
        retrievers={
            "keyword": bm25_retriever,
            "semantic": embedding_retriever,
            "graph": graph_retriever,
        },
        k=60,
        weights={"keyword": 1.0, "semantic": 1.2, "graph": 0.8},
    )

    results = await fuser.fuse("OOM error in k8s pod", top_k=10)

    for r in results:
        print(f"[{r.fused_score:.4f}] {r.chunk.doc_id}")
        print(f"  Ranks: {r.rank_contributions}")
        print(f"  {r.chunk.content[:120]}...")
```

### TypeScript

```typescript
interface RetrievedChunk {
  docId: string;
  content: string;
  source: string;
  metadata: Record<string, unknown>;
}

interface Retriever {
  retrieve(query: string, topK: number): Promise<RetrievedChunk[]>;
}

interface FusedResult {
  chunk: RetrievedChunk;
  fusedScore: number;
  rankContributions: Record<string, number>; // retrieverName -> rank
}

interface HybridSearchConfig {
  retrievers: Record<string, Retriever>;
  k?: number; // RRF smoothing constant (default: 60)
  weights?: Record<string, number>; // per-retriever weight (default: 1.0)
}

class HybridSearchFuser {
  private readonly retrievers: Record<string, Retriever>;
  private readonly k: number;
  private readonly weights: Record<string, number>;

  constructor(config: HybridSearchConfig) {
    this.retrievers = config.retrievers;
    this.k = config.k ?? 60;
    this.weights = config.weights ?? Object.fromEntries(
      Object.keys(config.retrievers).map((name) => [name, 1.0])
    );
  }

  async fuse(query: string, topK = 10, perRetrieverK = 20): Promise<FusedResult[]> {
    const retrieverNames = Object.keys(this.retrievers);

    // 1. Dispatch all retrievers in parallel
    const settledResults = await Promise.allSettled(
      retrieverNames.map((name) =>
        this.retrievers[name].retrieve(query, perRetrieverK)
      )
    );

    // 2. Collect per-retriever ranked lists
    const docChunks = new Map<string, RetrievedChunk>();
    const docRanks = new Map<string, Record<string, number>>();

    for (let i = 0; i < retrieverNames.length; i++) {
      const result = settledResults[i];
      if (result.status === "rejected") continue; // graceful degradation

      const retrieverName = retrieverNames[i];
      for (let rank = 0; rank < result.value.length; rank++) {
        const chunk = result.value[rank];
        if (!docChunks.has(chunk.docId)) {
          docChunks.set(chunk.docId, chunk);
          docRanks.set(chunk.docId, {});
        }
        docRanks.get(chunk.docId)![retrieverName] = rank + 1;
      }
    }

    // 3. Compute RRF scores
    const scored: FusedResult[] = [];

    for (const [docId, chunk] of docChunks) {
      const ranks = docRanks.get(docId)!;
      let fusedScore = 0;
      for (const [retrieverName, rank] of Object.entries(ranks)) {
        const weight = this.weights[retrieverName] ?? 1.0;
        fusedScore += weight * (1 / (this.k + rank));
      }
      scored.push({ chunk, fusedScore, rankContributions: ranks });
    }

    // 4. Sort descending and return top-K
    scored.sort((a, b) => b.fusedScore - a.fusedScore);
    return scored.slice(0, topK);
  }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

const fuser = new HybridSearchFuser({
  retrievers: {
    keyword: bm25Retriever,
    semantic: embeddingRetriever,
    graph: graphRetriever,
  },
  k: 60,
  weights: { keyword: 1.0, semantic: 1.2, graph: 0.8 },
});

const results = await fuser.fuse("OOM error in k8s pod", 10);

for (const r of results) {
  console.log(`[${r.fusedScore.toFixed(4)}] ${r.chunk.docId}`);
  console.log(`  Ranks: ${JSON.stringify(r.rankContributions)}`);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Eliminates single-retriever blind spots -- keyword catches exact terms, semantic catches meaning, graph catches relationships | Higher latency: parallel retrieval is bounded by the slowest retriever |
| RRF requires no score normalization across heterogeneous systems | More infrastructure to maintain (multiple search backends) |
| Graceful degradation: if one retriever fails, the others still produce results | Tuning weights and `k` requires experimentation per domain |
| Weighted variant lets you bias toward the retriever that matters most for your domain | Duplicate detection across retrievers requires consistent document IDs |
| Well-studied technique with strong empirical results in information retrieval benchmarks | Diminishing returns if your queries are homogeneous (all keyword-friendly or all semantic-friendly) |

## When to Use

- Your queries span a wide range of types: some exact-match, some conceptual, some relational.
- A single retriever measurably underperforms on a significant query segment.
- You have access to multiple search backends (e.g., Elasticsearch + a vector DB).
- You are building a general-purpose RAG system that must handle diverse user queries.
- You need robustness -- one retriever going down should not break the entire system.

## When NOT to Use

- Your queries are homogeneous and a single retriever handles them well. Adding fusion adds complexity without measurable gain.
- You are extremely latency-sensitive and cannot tolerate the overhead of multiple retriever calls (even in parallel).
- You only have one data source and one retrieval method available.
- Your document corpus is small enough that brute-force search with a single method achieves near-perfect recall.

## Related Patterns

- **[RAG Context Assembly](rag-context-assembly.md)** -- After fusion produces a ranked list, RAG Context Assembly handles deduplication, token budgeting, and source attribution before injecting into the prompt.
- **[Context-Aware Re-ranking](context-aware-reranking.md)** -- Hybrid fusion produces a first-pass ranking; re-ranking with conversation context further improves relevance.
- **[Just-in-Time Retrieval](just-in-time-retrieval.md)** -- Decides *when* to trigger the hybrid search, preventing unnecessary retrieval on turns that do not need it.
- **[Semantic Tool Selection](semantic-tool-selection.md)** -- Applies a similar multi-signal retrieval concept but for tool descriptions rather than knowledge chunks.

## Real-World Examples

- **Elasticsearch + kNN**: Elasticsearch 8.x natively supports hybrid queries combining BM25 with dense vector kNN search using RRF, making this pattern a first-class feature in one of the most widely deployed search engines.
- **Weaviate Hybrid Search**: Weaviate provides a built-in `hybrid` query type that fuses BM25 and vector search results with a configurable `alpha` parameter controlling the balance between keyword and semantic signals.
- **Pinecone Sparse-Dense**: Pinecone supports sparse (keyword) and dense (semantic) vectors in the same index, enabling hybrid search in a single query without managing separate backends.
- **Perplexity AI**: Combines multiple retrieval strategies (web search, index search, semantic matching) and fuses results before feeding them to the answer-generation model, producing answers that cite diverse high-quality sources.
- **Azure AI Search**: Microsoft's Azure AI Search provides a Semantic Ranker that fuses keyword (BM25) retrieval with a transformer-based re-ranker, following the hybrid search fusion pattern at infrastructure level.
