# Semantic Memory Indexing

> Index memories using semantic embeddings for retrieval by meaning rather than keywords, enabling "what do I know about X?" queries across all stored knowledge.

## Problem

As agents accumulate knowledge across sessions, retrieving the right memory at the right time becomes critical. Existing approaches have structural limitations:

- **Filesystem-as-Memory** organizes by path (`/preferences/editor.md`, `/decisions/auth-approach.md`). You must know *where* something was stored to find it. If you search for "how did we handle rate limiting?" but the memory is filed under `/decisions/api-throttling.md`, you miss it.
- **Episodic Memory** organizes by time (session snapshots). You can find similar past episodes, but not isolated facts that span multiple episodes.
- **Keyword search** matches exact terms. Searching for "authentication" misses memories that discuss "login flow", "OAuth", or "credential management."

Without semantic indexing, agents either fail to recall relevant knowledge or must scan every memory linearly, consuming context budget on irrelevant content.

## Solution

Create a **semantic index** over all stored memories -- regardless of how they are structured on disk or in a database. Each memory is embedded into a vector space, and retrieval is performed by embedding the query and finding the nearest neighbors. This enables retrieval by meaning: "what do I know about handling user authentication?" returns memories about OAuth, JWT, session management, and login flows even though none of those keywords appear in the query.

The semantic index sits alongside (not replacing) the structured storage. Filesystem-as-Memory or Episodic Memory handle the storage and organization; the semantic index handles discovery.

## How It Works

```
Memory Storage (any backend)
+---------------------------+
| /preferences/editor.md    |
| /decisions/auth.md        |
| /episodes/session-42.json |
| /facts/api-limits.md      |
+---------------------------+
        |
        | Index all memories
        v
+---------------------------+
| Semantic Index             |
| (Vector Store)             |
|                            |
| embed("Editor preferences  |
|   for VS Code...") -> [..]|
| embed("Chose JWT over     |
|   sessions because...")    |
|   -> [0.2, 0.8, ...]      |
| embed("Session 42: fixed  |
|   rate limiter...") -> [..]|
+---------------------------+
        |
        | Query: "How do we handle auth?"
        | embed(query) -> [0.19, 0.82, ...]
        | nearest neighbors -> auth.md, session-42
        v
+---------------------------+
| Retrieved Memories         |
| 1. decisions/auth.md       |
|    (similarity: 0.94)      |
| 2. episodes/session-42     |
|    (similarity: 0.78)      |
+---------------------------+
        |
        v
Injected into agent context
```

## Implementation

### Python

```python
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class MemoryEntry:
    """A single indexed memory with its source metadata."""
    memory_id: str
    content: str
    source_path: str          # Where the memory lives in storage
    source_type: str          # "file", "episode", "fact", "preference"
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """Combine content and metadata into a single string for embedding."""
        parts = [self.content]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)


@dataclass(frozen=True)
class SearchResult:
    """A memory retrieval result with similarity score."""
    memory: MemoryEntry
    score: float              # Similarity score (0-1, higher = more similar)


class SemanticMemoryIndex:
    """Indexes memories using semantic embeddings for retrieval by meaning.

    Works alongside any storage backend (filesystem, database, episodic store).
    The index handles discovery; the storage backend handles persistence.

    Requires:
        embed_fn: Callable that takes a string and returns a list of floats.
        vector_store: Object with add(id, embedding, metadata) and
                      query(embedding, top_k) -> list[{id, score, metadata}].
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        vector_store,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self._embed_fn = embed_fn
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._memories: dict[str, MemoryEntry] = {}

    def index_memory(
        self,
        content: str,
        source_path: str,
        source_type: str = "file",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Index a new memory for semantic retrieval."""
        memory_id = hashlib.sha256(
            f"{source_path}:{content[:100]}".encode()
        ).hexdigest()[:16]

        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            source_path=source_path,
            source_type=source_type,
            tags=tuple(tags or []),
            metadata=metadata or {},
        )

        # For long content, chunk and index each chunk separately
        chunks = self._chunk_content(entry)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{memory_id}:chunk-{i}"
            embedding = self._embed_fn(chunk)
            self._vector_store.add(
                id=chunk_id,
                embedding=embedding,
                metadata={
                    "memory_id": memory_id,
                    "chunk_index": i,
                    "source_path": source_path,
                    "source_type": source_type,
                    "tags": json.dumps(tags or []),
                },
            )

        self._memories[memory_id] = entry
        return entry

    def update_memory(
        self,
        source_path: str,
        new_content: str,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        """Re-index a memory when its source content changes."""
        # Find existing memory by source path
        existing = next(
            (m for m in self._memories.values() if m.source_path == source_path),
            None,
        )
        if existing is None:
            return None

        # Remove old embeddings
        self._vector_store.delete(prefix=existing.memory_id)

        # Re-index with new content
        return self.index_memory(
            content=new_content,
            source_path=source_path,
            source_type=existing.source_type,
            tags=tags or list(existing.tags),
            metadata={**existing.metadata, "previous_version": existing.memory_id},
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search memories by semantic similarity to the query."""
        query_embedding = self._embed_fn(query)

        # Retrieve more candidates than needed to allow for deduplication
        raw_results = self._vector_store.query(
            embedding=query_embedding,
            top_k=top_k * 3,
        )

        # Deduplicate: multiple chunks of the same memory may match
        seen_memory_ids: set[str] = set()
        results: list[SearchResult] = []

        for hit in raw_results:
            memory_id = hit["metadata"]["memory_id"]
            score = hit.get("score", 0.0)

            if memory_id in seen_memory_ids:
                continue
            if score < min_score:
                continue

            memory = self._memories.get(memory_id)
            if memory is None:
                continue

            # Filter by source type if specified
            if source_type and memory.source_type != source_type:
                continue

            seen_memory_ids.add(memory_id)
            results.append(SearchResult(memory=memory, score=score))

            if len(results) >= top_k:
                break

        return results

    def build_context_block(
        self,
        query: str,
        top_k: int = 3,
        max_chars: int = 8_000,
    ) -> str:
        """Build a formatted context block for injection into an LLM prompt."""
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        header = f"# Relevant Memories (query: {query[:60]})\n\n"
        blocks: list[str] = []
        total_chars = len(header)

        for sr in results:
            block = (
                f"## [{sr.memory.source_type}] {sr.memory.source_path}\n"
                f"**Relevance**: {sr.score:.2f}\n\n"
                f"{sr.memory.content}\n"
            )
            if total_chars + len(block) > max_chars:
                break
            blocks.append(block)
            total_chars += len(block)

        if not blocks:
            return ""

        return header + "\n---\n\n".join(blocks)

    def _chunk_content(self, entry: MemoryEntry) -> list[str]:
        """Split long content into overlapping chunks for indexing."""
        text = entry.to_embedding_text()
        if len(text) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self._chunk_overlap

        return chunks

    def get_stats(self) -> dict[str, Any]:
        """Return index statistics."""
        type_counts: dict[str, int] = {}
        for m in self._memories.values():
            type_counts[m.source_type] = type_counts.get(m.source_type, 0) + 1

        return {
            "total_memories": len(self._memories),
            "by_type": type_counts,
        }


# --- Usage Example ---

async def agent_with_semantic_memory(llm_client, embed_fn, vector_store):
    """An agent that uses semantic memory to recall relevant knowledge."""
    index = SemanticMemoryIndex(embed_fn, vector_store)

    # Index existing knowledge from various sources
    index.index_memory(
        content="We chose JWT tokens over session cookies for the API because "
                "our architecture is stateless and we need cross-service auth.",
        source_path="/decisions/auth-approach.md",
        source_type="decision",
        tags=["auth", "jwt", "architecture"],
    )

    index.index_memory(
        content="Rate limiting is set to 100 requests/minute per API key. "
                "We use a sliding window algorithm with Redis.",
        source_path="/decisions/api-throttling.md",
        source_type="decision",
        tags=["api", "rate-limiting", "redis"],
    )

    index.index_memory(
        content="The user prefers explicit error handling over try/catch blocks. "
                "Use Result types where available.",
        source_path="/preferences/coding-style.md",
        source_type="preference",
        tags=["coding-style", "errors"],
    )

    # Later: query by meaning, not keywords
    context = index.build_context_block(
        query="How should I handle authentication in the new microservice?"
    )
    print(f"Retrieved context:\n{context}")
    # Returns auth-approach.md even though query says "authentication"
    # and the memory says "JWT tokens"
```

### TypeScript

```typescript
import { createHash } from "crypto";

interface MemoryEntry {
  readonly memoryId: string;
  readonly content: string;
  readonly sourcePath: string;
  readonly sourceType: string;
  readonly tags: readonly string[];
  readonly createdAt: number;
  readonly updatedAt: number;
  readonly metadata: Record<string, unknown>;
}

interface SearchResult {
  readonly memory: MemoryEntry;
  readonly score: number;
}

interface VectorStore {
  add(id: string, embedding: number[], metadata: Record<string, unknown>): void;
  delete(prefix: string): void;
  query(
    embedding: number[],
    topK: number
  ): Array<{
    id: string;
    score: number;
    metadata: Record<string, unknown>;
  }>;
}

type EmbedFn = (text: string) => number[];

interface SemanticIndexConfig {
  readonly chunkSize: number;
  readonly chunkOverlap: number;
}

const DEFAULT_INDEX_CONFIG: SemanticIndexConfig = {
  chunkSize: 500,
  chunkOverlap: 50,
};

function createMemoryEntry(
  content: string,
  sourcePath: string,
  sourceType: string,
  tags: string[] = [],
  metadata: Record<string, unknown> = {}
): MemoryEntry {
  const memoryId = createHash("sha256")
    .update(`${sourcePath}:${content.slice(0, 100)}`)
    .digest("hex")
    .slice(0, 16);

  const now = Date.now();
  return {
    memoryId,
    content,
    sourcePath,
    sourceType,
    tags: Object.freeze([...tags]),
    createdAt: now,
    updatedAt: now,
    metadata: { ...metadata },
  };
}

function chunkContent(
  text: string,
  chunkSize: number,
  chunkOverlap: number
): string[] {
  if (text.length <= chunkSize) return [text];

  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + chunkSize));
    start += chunkSize - chunkOverlap;
  }
  return chunks;
}

class SemanticMemoryIndex {
  private readonly embedFn: EmbedFn;
  private readonly vectorStore: VectorStore;
  private readonly config: SemanticIndexConfig;
  private readonly memories: Map<string, MemoryEntry> = new Map();

  constructor(
    embedFn: EmbedFn,
    vectorStore: VectorStore,
    config: SemanticIndexConfig = DEFAULT_INDEX_CONFIG
  ) {
    this.embedFn = embedFn;
    this.vectorStore = vectorStore;
    this.config = config;
  }

  indexMemory(
    content: string,
    sourcePath: string,
    sourceType: string = "file",
    tags: string[] = [],
    metadata: Record<string, unknown> = {}
  ): MemoryEntry {
    const entry = createMemoryEntry(
      content,
      sourcePath,
      sourceType,
      tags,
      metadata
    );

    const embeddingText = tags.length > 0
      ? `${content}\nTags: ${tags.join(", ")}`
      : content;

    const chunks = chunkContent(
      embeddingText,
      this.config.chunkSize,
      this.config.chunkOverlap
    );

    chunks.forEach((chunk, i) => {
      const chunkId = `${entry.memoryId}:chunk-${i}`;
      const embedding = this.embedFn(chunk);
      this.vectorStore.add(chunkId, embedding, {
        memoryId: entry.memoryId,
        chunkIndex: i,
        sourcePath,
        sourceType,
        tags: JSON.stringify(tags),
      });
    });

    this.memories.set(entry.memoryId, entry);
    return entry;
  }

  search(
    query: string,
    topK: number = 5,
    sourceType?: string,
    minScore: number = 0.0
  ): SearchResult[] {
    const queryEmbedding = this.embedFn(query);
    const rawResults = this.vectorStore.query(queryEmbedding, topK * 3);

    const seen = new Set<string>();
    const results: SearchResult[] = [];

    for (const hit of rawResults) {
      const memoryId = hit.metadata.memoryId as string;
      if (seen.has(memoryId)) continue;
      if (hit.score < minScore) continue;

      const memory = this.memories.get(memoryId);
      if (!memory) continue;
      if (sourceType && memory.sourceType !== sourceType) continue;

      seen.add(memoryId);
      results.push({ memory, score: hit.score });

      if (results.length >= topK) break;
    }

    return results;
  }

  buildContextBlock(
    query: string,
    topK: number = 3,
    maxChars: number = 8_000
  ): string {
    const results = this.search(query, topK);
    if (results.length === 0) return "";

    const header = `# Relevant Memories (query: ${query.slice(0, 60)})\n\n`;
    const blocks: string[] = [];
    let totalChars = header.length;

    for (const sr of results) {
      const block = [
        `## [${sr.memory.sourceType}] ${sr.memory.sourcePath}`,
        `**Relevance**: ${sr.score.toFixed(2)}`,
        "",
        sr.memory.content,
        "",
      ].join("\n");

      if (totalChars + block.length > maxChars) break;
      blocks.push(block);
      totalChars += block.length;
    }

    if (blocks.length === 0) return "";
    return header + blocks.join("\n---\n\n");
  }

  getStats(): { totalMemories: number; byType: Record<string, number> } {
    const byType: Record<string, number> = {};
    for (const m of this.memories.values()) {
      byType[m.sourceType] = (byType[m.sourceType] ?? 0) + 1;
    }
    return { totalMemories: this.memories.size, byType };
  }
}

// --- Usage Example ---

function buildAgentWithSemanticMemory(
  embedFn: EmbedFn,
  vectorStore: VectorStore
): SemanticMemoryIndex {
  const index = new SemanticMemoryIndex(embedFn, vectorStore);

  index.indexMemory(
    "We chose JWT tokens over session cookies for the API because " +
      "our architecture is stateless and we need cross-service auth.",
    "/decisions/auth-approach.md",
    "decision",
    ["auth", "jwt", "architecture"]
  );

  index.indexMemory(
    "Rate limiting is set to 100 requests/minute per API key. " +
      "We use a sliding window algorithm with Redis.",
    "/decisions/api-throttling.md",
    "decision",
    ["api", "rate-limiting", "redis"]
  );

  // Query by meaning: "authentication" matches "JWT tokens"
  const context = index.buildContextBlock(
    "How should I handle authentication in the new microservice?"
  );
  console.log(`Retrieved context:\n${context}`);

  return index;
}

export { SemanticMemoryIndex, MemoryEntry, SearchResult, EmbedFn, VectorStore };
```

## Trade-offs

| Pros | Cons |
|------|------|
| Retrieval by meaning, not exact keywords | Requires embedding model (API cost or local compute) |
| Works across all memory types (files, episodes, facts) | Embedding quality determines retrieval quality |
| Handles vocabulary mismatch (query terms differ from stored terms) | Vector store adds infrastructure complexity |
| Scales to large memory stores without linear scanning | Chunking strategy requires tuning per content type |
| Composable with filesystem and episodic memory patterns | Semantic similarity can return false positives (topically related but irrelevant) |

## When to Use

- The agent has accumulated enough knowledge that browsing or keyword search is insufficient.
- Memories are stored in heterogeneous formats (files, episodes, key-value pairs) and need a unified retrieval layer.
- Queries are natural language ("how did we handle X?") rather than structured lookups ("get key Y").
- The vocabulary used in queries may differ from the vocabulary in stored memories.
- You need to answer "what do I know about X?" across all stored knowledge.

## When NOT to Use

- Small knowledge bases (fewer than ~50 memories) where linear scanning is fast and reliable.
- Highly structured data where exact key lookups are sufficient (user settings, configuration).
- When embedding latency or cost is prohibitive for the use case.
- When memories are short-lived and do not accumulate across sessions.
- When you need deterministic, exact-match retrieval (semantic search is probabilistic).

## Related Patterns

- **[Filesystem-as-Memory](filesystem-as-memory.md)** -- Provides the structured storage layer. Semantic indexing adds a retrieval layer on top.
- **[Episodic Memory](episodic-memory.md)** -- Episodes are one type of memory that can be semantically indexed. This pattern generalizes the approach to all memory types.
- **[Memory Consolidation](memory-consolidation.md)** -- After semantic indexing reveals overlapping or contradictory memories, consolidation merges and prunes them.
- **[Cross-Session State Sync](cross-session-state-sync.md)** -- The semantic index must be kept in sync when memories change across sessions.

## Real-World Examples

1. **ChatGPT Memory** -- OpenAI's memory feature stores facts about the user and retrieves them semantically when relevant to the current conversation. The user says "help me write a Python script" and the system recalls "user prefers type hints and pytest" from a prior conversation.

2. **Cursor / Codebase indexing** -- Cursor indexes the entire codebase using embeddings, enabling semantic search like "where is error handling done?" that returns files about error handling even if they do not contain the word "error."

3. **Notion AI** -- Notion indexes workspace content with embeddings, enabling "ask anything about your workspace" queries that return relevant pages regardless of how they were titled or organized.

4. **RAG pipelines** -- Every production RAG system is essentially semantic memory indexing applied to a document corpus. The pattern here applies the same technique to agent memory rather than static documents.
