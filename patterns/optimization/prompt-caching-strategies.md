# Prompt Caching Strategies

> Implement multi-level caching for LLM interactions -- from exact prompt-response pairs to semantic similarity matches to component-level reuse -- reducing cost and latency by avoiding redundant inference.

## Problem

KV-cache optimization (the provider-side prefix cache) is powerful but limited: it only helps when the *exact same prefix* is reused across calls. In practice, many applications have a different problem:

- **Repeated queries**: Users ask the same question in slightly different words. Each variation triggers a full LLM call, even though the answer is identical or near-identical.
- **Multi-step workflows**: An agent pipeline runs embedding generation, then re-ranking, then summarization. If the pipeline reruns on similar inputs, every step recomputes from scratch.
- **Expensive intermediate results**: Embeddings, classification labels, and extracted entities are computed per-request even when the same document has been processed before.
- **Cost explosion at scale**: A system handling 10,000 requests/day where 30% are near-duplicates wastes 3,000 calls worth of compute daily.

Without application-level caching, you pay full inference cost for every call regardless of whether you have seen that input (or something very close to it) before.

## Solution

Build a multi-level cache that sits between your application and the LLM provider. Each level trades off precision for hit rate:

1. **L1 -- Exact Match**: Hash the full prompt and return a cached response if the hash matches. Zero ambiguity, fastest lookup, but only catches identical prompts.
2. **L2 -- Semantic Similarity**: Embed the prompt and search a vector store for cached prompts above a similarity threshold. Catches paraphrases and minor variations.
3. **L3 -- Component-Level**: Cache individual components (embeddings, tool outputs, extracted entities) independently. Even when the full prompt is novel, its parts may have been computed before.

Each level has its own TTL (time-to-live) and invalidation strategy. L1 entries expire quickly (minutes) because exact matches are fragile. L2 entries last longer (hours) because semantic similarity is more durable. L3 entries persist longest (days) because embeddings and entities change only when the underlying data changes.

## How It Works

```
Incoming prompt
      |
      v
+------------------+
| L1: Exact Match  |  <-- Hash lookup, O(1)
| TTL: 5 minutes   |
+------------------+
      |
  HIT? --> Return cached response
      |
      v (MISS)
+------------------+
| L2: Semantic     |  <-- Vector similarity search, O(log n)
| Similarity > 0.95|
| TTL: 2 hours     |
+------------------+
      |
  HIT? --> Return cached response (with similarity score)
      |
      v (MISS)
+------------------+
| L3: Component    |  <-- Check if sub-parts are cached
| Cache            |
| TTL: 24 hours    |
+------------------+
      |
  PARTIAL HIT? --> Use cached components, compute only missing parts
      |
      v (FULL MISS)
+------------------+
| LLM Provider     |  <-- Full inference call
+------------------+
      |
      v
+------------------+
| Cache Write-Back |  <-- Store result at all applicable levels
+------------------+


Cache invalidation:
  L1: Time-based expiry (short TTL)
  L2: Time-based + explicit invalidation on source data change
  L3: Content-hash invalidation (re-cache when source document changes)
```

## Implementation

### Python

```python
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CacheEntry:
    """An immutable cached result with metadata."""
    key: str
    value: str
    created_at: float
    ttl_seconds: float
    similarity_score: float = 1.0
    hit_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def with_hit(self) -> "CacheEntry":
        """Return a new entry with incremented hit count."""
        return CacheEntry(
            key=self.key,
            value=self.value,
            created_at=self.created_at,
            ttl_seconds=self.ttl_seconds,
            similarity_score=self.similarity_score,
            hit_count=self.hit_count + 1,
            metadata=self.metadata,
        )


class EmbeddingProvider(Protocol):
    """Interface for embedding generation."""
    def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Interface for vector similarity search."""
    def search(
        self, embedding: list[float], top_k: int
    ) -> list[tuple[str, float]]: ...

    def upsert(self, key: str, embedding: list[float], metadata: dict) -> None: ...


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache behavior."""
    l1_ttl_seconds: float = 300.0       # 5 minutes
    l2_ttl_seconds: float = 7200.0      # 2 hours
    l3_ttl_seconds: float = 86400.0     # 24 hours
    l2_similarity_threshold: float = 0.95
    max_l1_entries: int = 10_000
    max_l2_entries: int = 50_000
    enable_l2: bool = True
    enable_l3: bool = True


class MultiLevelPromptCache:
    """
    A multi-level cache for LLM prompt-response pairs.

    L1: Exact match via content hash (fastest, most precise).
    L2: Semantic similarity via embedding comparison (catches paraphrases).
    L3: Component-level cache for reusable sub-computations.
    """

    def __init__(
        self,
        config: CacheConfig,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ):
        self._config = config
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._l1_cache: dict[str, CacheEntry] = {}
        self._l2_cache: dict[str, CacheEntry] = {}
        self._l3_cache: dict[str, CacheEntry] = {}
        self._stats = {
            "l1_hits": 0, "l2_hits": 0, "l3_hits": 0,
            "misses": 0, "total_lookups": 0,
        }

    def get(self, prompt: str, components: dict[str, str] | None = None) -> CacheEntry | None:
        """
        Look up a prompt across all cache levels.

        Args:
            prompt: The full prompt string.
            components: Optional dict of named sub-components for L3 lookup.

        Returns:
            A CacheEntry if found, None on full miss.
        """
        self._stats["total_lookups"] += 1

        # L1: Exact match
        l1_key = self._hash_prompt(prompt)
        l1_result = self._l1_lookup(l1_key)
        if l1_result is not None:
            self._stats["l1_hits"] += 1
            return l1_result

        # L2: Semantic similarity
        if self._config.enable_l2 and self._embedding_provider and self._vector_store:
            l2_result = self._l2_lookup(prompt)
            if l2_result is not None:
                self._stats["l2_hits"] += 1
                return l2_result

        # L3: Component-level
        if self._config.enable_l3 and components:
            l3_result = self._l3_lookup(components)
            if l3_result is not None:
                self._stats["l3_hits"] += 1
                return l3_result

        self._stats["misses"] += 1
        return None

    def put(
        self,
        prompt: str,
        response: str,
        components: dict[str, str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Store a prompt-response pair at all applicable cache levels.

        Args:
            prompt: The full prompt string.
            response: The LLM response to cache.
            components: Optional named sub-components for L3 caching.
            metadata: Optional metadata to attach to the cache entry.
        """
        now = time.time()
        meta = metadata or {}

        # L1: Store exact match
        l1_key = self._hash_prompt(prompt)
        self._l1_cache[l1_key] = CacheEntry(
            key=l1_key,
            value=response,
            created_at=now,
            ttl_seconds=self._config.l1_ttl_seconds,
            metadata=meta,
        )
        self._evict_l1_if_needed()

        # L2: Store embedding for semantic lookup
        if self._config.enable_l2 and self._embedding_provider and self._vector_store:
            embedding = self._embedding_provider.embed(prompt)
            self._vector_store.upsert(
                key=l1_key,
                embedding=embedding,
                metadata={"response": response, "created_at": now, **meta},
            )
            self._l2_cache[l1_key] = CacheEntry(
                key=l1_key,
                value=response,
                created_at=now,
                ttl_seconds=self._config.l2_ttl_seconds,
                metadata=meta,
            )

        # L3: Store individual components
        if self._config.enable_l3 and components:
            for comp_name, comp_value in components.items():
                comp_key = self._hash_prompt(f"{comp_name}:{comp_value}")
                self._l3_cache[comp_key] = CacheEntry(
                    key=comp_key,
                    value=comp_value,
                    created_at=now,
                    ttl_seconds=self._config.l3_ttl_seconds,
                    metadata={"component": comp_name, **meta},
                )

    def invalidate(self, prompt: str) -> None:
        """Remove a specific prompt from all cache levels."""
        l1_key = self._hash_prompt(prompt)
        self._l1_cache = {
            k: v for k, v in self._l1_cache.items() if k != l1_key
        }
        self._l2_cache = {
            k: v for k, v in self._l2_cache.items() if k != l1_key
        }

    def _l1_lookup(self, key: str) -> CacheEntry | None:
        entry = self._l1_cache.get(key)
        if entry is None or entry.is_expired:
            if entry is not None:
                self._l1_cache = {
                    k: v for k, v in self._l1_cache.items() if k != key
                }
            return None
        updated = entry.with_hit()
        self._l1_cache[key] = updated
        return updated

    def _l2_lookup(self, prompt: str) -> CacheEntry | None:
        embedding = self._embedding_provider.embed(prompt)
        results = self._vector_store.search(embedding, top_k=1)

        if not results:
            return None

        best_key, similarity = results[0]
        if similarity < self._config.l2_similarity_threshold:
            return None

        entry = self._l2_cache.get(best_key)
        if entry is None or entry.is_expired:
            return None

        return CacheEntry(
            key=entry.key,
            value=entry.value,
            created_at=entry.created_at,
            ttl_seconds=entry.ttl_seconds,
            similarity_score=similarity,
            hit_count=entry.hit_count + 1,
            metadata=entry.metadata,
        )

    def _l3_lookup(self, components: dict[str, str]) -> CacheEntry | None:
        """Check if all components are cached. Return combined result."""
        cached_components: dict[str, str] = {}
        for comp_name, comp_value in components.items():
            comp_key = self._hash_prompt(f"{comp_name}:{comp_value}")
            entry = self._l3_cache.get(comp_key)
            if entry is None or entry.is_expired:
                return None  # All components must be cached for a hit
            cached_components[comp_name] = entry.value

        # All components found -- return a synthetic cache entry
        combined = json.dumps(cached_components, sort_keys=True)
        return CacheEntry(
            key="l3-composite",
            value=combined,
            created_at=time.time(),
            ttl_seconds=self._config.l3_ttl_seconds,
            metadata={"source": "l3_component_cache"},
        )

    def _evict_l1_if_needed(self) -> None:
        """Evict oldest L1 entries if over capacity."""
        if len(self._l1_cache) <= self._config.max_l1_entries:
            return
        # Remove expired first
        self._l1_cache = {
            k: v for k, v in self._l1_cache.items() if not v.is_expired
        }
        # If still over, remove oldest entries
        if len(self._l1_cache) > self._config.max_l1_entries:
            sorted_entries = sorted(
                self._l1_cache.items(), key=lambda kv: kv[1].created_at
            )
            keep_count = self._config.max_l1_entries
            self._l1_cache = dict(sorted_entries[-keep_count:])

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:32]

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def hit_rate(self) -> float:
        total = self._stats["total_lookups"]
        if total == 0:
            return 0.0
        hits = self._stats["l1_hits"] + self._stats["l2_hits"] + self._stats["l3_hits"]
        return (hits / total) * 100
```

### TypeScript

```typescript
import { createHash } from "crypto";

interface CacheEntry {
  readonly key: string;
  readonly value: string;
  readonly createdAt: number;
  readonly ttlSeconds: number;
  readonly similarityScore: number;
  readonly hitCount: number;
  readonly metadata: Readonly<Record<string, unknown>>;
}

interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
}

interface VectorStore {
  search(
    embedding: number[],
    topK: number
  ): Promise<Array<{ key: string; similarity: number }>>;
  upsert(
    key: string,
    embedding: number[],
    metadata: Record<string, unknown>
  ): Promise<void>;
}

interface CacheConfig {
  readonly l1TtlSeconds: number;
  readonly l2TtlSeconds: number;
  readonly l3TtlSeconds: number;
  readonly l2SimilarityThreshold: number;
  readonly maxL1Entries: number;
  readonly enableL2: boolean;
  readonly enableL3: boolean;
}

interface CacheStats {
  l1Hits: number;
  l2Hits: number;
  l3Hits: number;
  misses: number;
  totalLookups: number;
}

const DEFAULT_CONFIG: CacheConfig = {
  l1TtlSeconds: 300,
  l2TtlSeconds: 7200,
  l3TtlSeconds: 86400,
  l2SimilarityThreshold: 0.95,
  maxL1Entries: 10_000,
  enableL2: true,
  enableL3: true,
};

function isExpired(entry: CacheEntry): boolean {
  return (Date.now() / 1000 - entry.createdAt) > entry.ttlSeconds;
}

function withHit(entry: CacheEntry): CacheEntry {
  return { ...entry, hitCount: entry.hitCount + 1 };
}

function hashPrompt(prompt: string): string {
  return createHash("sha256").update(prompt).digest("hex").slice(0, 32);
}

function createEntry(params: {
  key: string;
  value: string;
  ttlSeconds: number;
  similarityScore?: number;
  metadata?: Record<string, unknown>;
}): CacheEntry {
  return {
    key: params.key,
    value: params.value,
    createdAt: Date.now() / 1000,
    ttlSeconds: params.ttlSeconds,
    similarityScore: params.similarityScore ?? 1.0,
    hitCount: 0,
    metadata: Object.freeze(params.metadata ?? {}),
  };
}

class MultiLevelPromptCache {
  private l1Cache: ReadonlyMap<string, CacheEntry> = new Map();
  private l2Cache: ReadonlyMap<string, CacheEntry> = new Map();
  private l3Cache: ReadonlyMap<string, CacheEntry> = new Map();
  private readonly config: CacheConfig;
  private readonly embeddingProvider?: EmbeddingProvider;
  private readonly vectorStore?: VectorStore;
  private stats: CacheStats = {
    l1Hits: 0, l2Hits: 0, l3Hits: 0, misses: 0, totalLookups: 0,
  };

  constructor(
    config?: Partial<CacheConfig>,
    embeddingProvider?: EmbeddingProvider,
    vectorStore?: VectorStore
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.embeddingProvider = embeddingProvider;
    this.vectorStore = vectorStore;
  }

  async get(
    prompt: string,
    components?: Record<string, string>
  ): Promise<CacheEntry | null> {
    this.stats = { ...this.stats, totalLookups: this.stats.totalLookups + 1 };
    const l1Key = hashPrompt(prompt);

    // L1: Exact match
    const l1Entry = this.l1Cache.get(l1Key);
    if (l1Entry && !isExpired(l1Entry)) {
      this.stats = { ...this.stats, l1Hits: this.stats.l1Hits + 1 };
      this.l1Cache = new Map([...this.l1Cache, [l1Key, withHit(l1Entry)]]);
      return withHit(l1Entry);
    }

    // L2: Semantic similarity
    if (this.config.enableL2 && this.embeddingProvider && this.vectorStore) {
      const l2Result = await this.l2Lookup(prompt);
      if (l2Result) {
        this.stats = { ...this.stats, l2Hits: this.stats.l2Hits + 1 };
        return l2Result;
      }
    }

    // L3: Component-level
    if (this.config.enableL3 && components) {
      const l3Result = this.l3Lookup(components);
      if (l3Result) {
        this.stats = { ...this.stats, l3Hits: this.stats.l3Hits + 1 };
        return l3Result;
      }
    }

    this.stats = { ...this.stats, misses: this.stats.misses + 1 };
    return null;
  }

  async put(
    prompt: string,
    response: string,
    components?: Record<string, string>,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    const l1Key = hashPrompt(prompt);
    const meta = metadata ?? {};

    // L1
    const l1Entry = createEntry({
      key: l1Key,
      value: response,
      ttlSeconds: this.config.l1TtlSeconds,
      metadata: meta,
    });
    this.l1Cache = new Map([...this.l1Cache, [l1Key, l1Entry]]);
    this.evictL1IfNeeded();

    // L2
    if (this.config.enableL2 && this.embeddingProvider && this.vectorStore) {
      const embedding = await this.embeddingProvider.embed(prompt);
      await this.vectorStore.upsert(l1Key, embedding, {
        response, createdAt: Date.now() / 1000, ...meta,
      });
      const l2Entry = createEntry({
        key: l1Key,
        value: response,
        ttlSeconds: this.config.l2TtlSeconds,
        metadata: meta,
      });
      this.l2Cache = new Map([...this.l2Cache, [l1Key, l2Entry]]);
    }

    // L3
    if (this.config.enableL3 && components) {
      const newEntries = new Map(this.l3Cache);
      for (const [name, value] of Object.entries(components)) {
        const compKey = hashPrompt(`${name}:${value}`);
        newEntries.set(
          compKey,
          createEntry({
            key: compKey,
            value,
            ttlSeconds: this.config.l3TtlSeconds,
            metadata: { component: name, ...meta },
          })
        );
      }
      this.l3Cache = newEntries;
    }
  }

  invalidate(prompt: string): void {
    const key = hashPrompt(prompt);
    const newL1 = new Map(this.l1Cache);
    newL1.delete(key);
    this.l1Cache = newL1;

    const newL2 = new Map(this.l2Cache);
    newL2.delete(key);
    this.l2Cache = newL2;
  }

  private async l2Lookup(prompt: string): Promise<CacheEntry | null> {
    const embedding = await this.embeddingProvider!.embed(prompt);
    const results = await this.vectorStore!.search(embedding, 1);

    if (results.length === 0) return null;

    const { key, similarity } = results[0];
    if (similarity < this.config.l2SimilarityThreshold) return null;

    const entry = this.l2Cache.get(key);
    if (!entry || isExpired(entry)) return null;

    return { ...entry, similarityScore: similarity, hitCount: entry.hitCount + 1 };
  }

  private l3Lookup(
    components: Record<string, string>
  ): CacheEntry | null {
    const cached: Record<string, string> = {};

    for (const [name, value] of Object.entries(components)) {
      const compKey = hashPrompt(`${name}:${value}`);
      const entry = this.l3Cache.get(compKey);
      if (!entry || isExpired(entry)) return null;
      cached[name] = entry.value;
    }

    return createEntry({
      key: "l3-composite",
      value: JSON.stringify(cached),
      ttlSeconds: this.config.l3TtlSeconds,
      metadata: { source: "l3_component_cache" },
    });
  }

  private evictL1IfNeeded(): void {
    if (this.l1Cache.size <= this.config.maxL1Entries) return;

    // Remove expired
    const live = new Map(
      [...this.l1Cache].filter(([, v]) => !isExpired(v))
    );

    // If still over, keep newest
    if (live.size > this.config.maxL1Entries) {
      const sorted = [...live.entries()].sort(
        (a, b) => a[1].createdAt - b[1].createdAt
      );
      this.l1Cache = new Map(sorted.slice(-this.config.maxL1Entries));
    } else {
      this.l1Cache = live;
    }
  }

  get cacheStats(): Readonly<CacheStats> {
    return { ...this.stats };
  }

  get hitRate(): number {
    const total = this.stats.totalLookups;
    if (total === 0) return 0;
    const hits = this.stats.l1Hits + this.stats.l2Hits + this.stats.l3Hits;
    return (hits / total) * 100;
  }
}

export {
  MultiLevelPromptCache,
  CacheEntry,
  CacheConfig,
  CacheStats,
  EmbeddingProvider,
  VectorStore,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Dramatic cost reduction on repeated or similar queries (30-60% savings typical) | L2 semantic caching requires an embedding model, adding latency on cache misses |
| Layered approach catches different types of duplication | Stale cache entries can return outdated responses if TTLs are too generous |
| Component-level caching avoids redundant embedding and extraction work | Cache storage costs (especially vector store for L2) add infrastructure complexity |
| Hit rate metrics provide clear ROI visibility | Semantic similarity threshold tuning requires experimentation per domain |
| Each level can be enabled/disabled independently | Cache invalidation is inherently hard -- source data changes may not propagate immediately |
| Works with any LLM provider (application-level, not provider-dependent) | L2 cache misses are slower than no cache at all (embedding + search overhead) |

## When to Use

- High-volume applications where users frequently ask similar questions
- Multi-step agent pipelines where intermediate results are reusable
- Systems with expensive embedding or re-ranking steps that repeat on similar inputs
- Customer support, FAQ, or documentation-query systems with predictable question patterns
- Batch processing where many inputs share common sub-components

## When NOT to Use

- Applications where every query is genuinely unique (creative writing, open-ended chat)
- When response freshness is critical and even short TTLs are unacceptable
- Low-volume systems where the infrastructure overhead exceeds the savings
- When the LLM provider's built-in caching already covers your use case
- Safety-critical systems where a stale cached response could cause harm

## Related Patterns

- **KV-Cache Optimization** (Optimization): Provider-side prefix caching complements application-level prompt caching. KV-cache optimizes how the provider processes your prompt; prompt caching avoids sending the prompt at all.
- **Incremental Context Updates** (Optimization): When context changes incrementally between turns, component-level caching (L3) can reuse the unchanged portions.
- **Context Rot Detection** (Evaluation): Cached responses can become stale context rot. Use rot detection to identify when cached responses no longer align with current system state.

## Real-World Examples

- **GPTCache**: An open-source project that implements semantic caching for LLM calls. It demonstrates the L1 + L2 pattern with configurable similarity thresholds and multiple vector store backends.
- **Portkey AI Gateway**: Production API gateway that implements prompt caching across multiple LLM providers, reporting 20-40% cost reduction for enterprise customers with repetitive query patterns.
- **Langchain Caching**: Provides both exact-match and semantic caching as middleware, allowing developers to add caching to existing chains without restructuring their code.
- **Production RAG Systems**: Systems that cache document embeddings (L3) avoid re-embedding unchanged documents on every query. This is standard practice in any vector-search-backed application.
