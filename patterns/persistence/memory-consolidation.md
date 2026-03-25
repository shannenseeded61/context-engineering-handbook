# Memory Consolidation

> Periodically merge related memories, resolve contradictions, and prune stale entries to prevent memory bloat and maintain a coherent knowledge base.

## Problem

Agent memory systems accumulate entries over time without bound. Each session adds new memories, but old memories are rarely updated or removed. Over weeks and months:

- **Redundancy**: The same fact is stored multiple times with slightly different wording ("User prefers dark mode", "User likes dark themes", "Set dark mode for the user").
- **Contradiction**: Newer memories contradict older ones ("User prefers Python" from January, "User has switched to Rust" from March), but both remain active and can be retrieved.
- **Fragmentation**: Related knowledge is scattered across many small memories instead of being organized into coherent topics.
- **Bloat**: The memory store grows linearly with usage, slowing retrieval and increasing embedding/search costs.
- **Staleness**: Memories about temporary states ("User is working on the Q4 report") persist long after they are relevant.

Without consolidation, memory systems degrade over time -- retrieval returns increasingly noisy, contradictory, and outdated results.

## Solution

Implement a **MemoryConsolidator** that runs periodically (like human memory consolidation during sleep) to:

1. **Cluster** related memories by semantic similarity.
2. **Merge** overlapping memories within each cluster into unified entries.
3. **Resolve** contradictions by preferring newer information and flagging conflicts.
4. **Prune** memories that are stale, superseded, or below a relevance threshold.

The consolidator produces a cleaner, smaller memory store where each topic has a single authoritative entry rather than many fragmented ones.

## How It Works

```
Before Consolidation                    After Consolidation
+----------------------------------+    +----------------------------------+
| Memory Store (47 entries)         |    | Memory Store (18 entries)        |
|                                   |    |                                  |
| "User prefers dark mode"          |    | "User prefers dark mode across   |
| "User likes dark themes"          | -> |  all applications."              |
| "Set dark mode for the user"      |    |  (merged from 3 entries)         |
|                                   |    |                                  |
| "User prefers Python"             |    | "User has switched primary       |
| "User has switched to Rust"       | -> |  language from Python to Rust    |
|                                   |    |  (as of March 2026)."           |
|                                   |    |  (contradiction resolved)        |
|                                   |    |                                  |
| "Working on Q4 2025 report"       |    | [PRUNED - stale, Q4 2025 ended] |
| "Sprint 12 deadline is Friday"    | -> |                                  |
|                                   |    |                                  |
| "Auth uses JWT tokens"            |    | "Auth architecture: JWT tokens,  |
| "JWT secret rotates monthly"      | -> |  monthly secret rotation,        |
| "Auth service on port 8080"       |    |  service on port 8080."          |
+----------------------------------+    +----------------------------------+

Consolidation Pipeline:

  All Memories --> Cluster --> Merge --> Resolve --> Prune --> Updated Store
                  (group     (combine  (pick      (remove
                  by topic)  overlap)  newer)     stale)
```

## Implementation

### Python

```python
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Memory:
    """A single memory entry."""
    memory_id: str
    content: str
    source: str
    created_at: float
    updated_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryCluster:
    """A group of related memories identified by semantic similarity."""
    cluster_id: str
    topic: str
    memories: tuple[Memory, ...]
    centroid_text: str


@dataclass(frozen=True)
class ConsolidationResult:
    """Report of what the consolidation process did."""
    clusters_found: int
    memories_merged: int
    contradictions_resolved: int
    memories_pruned: int
    original_count: int
    final_count: int
    duration_seconds: float


class MemoryConsolidator:
    """Periodically consolidates memories to prevent bloat and maintain coherence.

    Requires:
        embed_fn: Callable that takes a string and returns an embedding vector.
        llm_fn: Callable that takes a prompt and returns a completion string.
                Used for intelligent merging and contradiction resolution.
        similarity_threshold: Minimum cosine similarity to cluster memories together.
        staleness_days: Memories not accessed in this many days are pruning candidates.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        llm_fn: Callable[[str], str],
        similarity_threshold: float = 0.82,
        staleness_days: int = 90,
        min_cluster_size: int = 2,
    ):
        self._embed_fn = embed_fn
        self._llm_fn = llm_fn
        self._similarity_threshold = similarity_threshold
        self._staleness_seconds = staleness_days * 86_400
        self._min_cluster_size = min_cluster_size

    async def consolidate(
        self, memories: list[Memory]
    ) -> tuple[list[Memory], ConsolidationResult]:
        """Run the full consolidation pipeline.

        Returns the consolidated memory list and a report.
        """
        start_time = time.time()
        original_count = len(memories)

        # Step 1: Prune stale memories
        active_memories, pruned_count = self._prune_stale(memories)

        # Step 2: Cluster related memories
        clusters = self._cluster_memories(active_memories)

        # Step 3: Merge and resolve within each cluster
        merged_memories: list[Memory] = []
        total_merged = 0
        total_contradictions = 0

        for cluster in clusters:
            if len(cluster.memories) < self._min_cluster_size:
                # Single-memory clusters pass through unchanged
                merged_memories.extend(cluster.memories)
                continue

            # Check for contradictions
            has_contradiction = self._detect_contradiction(cluster)

            if has_contradiction:
                resolved = self._resolve_contradiction(cluster)
                total_contradictions += 1
            else:
                resolved = self._merge_cluster(cluster)

            merged_memories.append(resolved)
            total_merged += len(cluster.memories) - 1

        duration = time.time() - start_time

        result = ConsolidationResult(
            clusters_found=len(clusters),
            memories_merged=total_merged,
            contradictions_resolved=total_contradictions,
            memories_pruned=pruned_count,
            original_count=original_count,
            final_count=len(merged_memories),
            duration_seconds=duration,
        )

        return merged_memories, result

    def _prune_stale(
        self, memories: list[Memory]
    ) -> tuple[list[Memory], int]:
        """Remove memories that have not been accessed recently."""
        now = time.time()
        active: list[Memory] = []
        pruned = 0

        for memory in memories:
            last_touch = max(memory.last_accessed, memory.updated_at)
            age = now - last_touch

            # Keep if recently accessed or frequently accessed
            if age < self._staleness_seconds or memory.access_count > 10:
                active.append(memory)
            else:
                pruned += 1

        return active, pruned

    def _cluster_memories(
        self, memories: list[Memory]
    ) -> list[MemoryCluster]:
        """Group memories by semantic similarity using greedy clustering."""
        if not memories:
            return []

        embeddings = [self._embed_fn(m.content) for m in memories]
        assigned: set[int] = set()
        clusters: list[MemoryCluster] = []

        for i, memory in enumerate(memories):
            if i in assigned:
                continue

            cluster_members = [memory]
            cluster_indices = [i]
            assigned.add(i)

            for j in range(i + 1, len(memories)):
                if j in assigned:
                    continue
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._similarity_threshold:
                    cluster_members.append(memories[j])
                    cluster_indices.append(j)
                    assigned.add(j)

            clusters.append(
                MemoryCluster(
                    cluster_id=f"cluster-{len(clusters)}",
                    topic=self._extract_topic(cluster_members),
                    memories=tuple(cluster_members),
                    centroid_text=cluster_members[0].content,
                )
            )

        return clusters

    def _detect_contradiction(self, cluster: MemoryCluster) -> bool:
        """Use the LLM to detect if memories in a cluster contradict each other."""
        if len(cluster.memories) < 2:
            return False

        contents = "\n".join(
            f"- [{m.memory_id}] ({self._format_date(m.created_at)}): {m.content}"
            for m in cluster.memories
        )

        prompt = (
            "Do the following memory entries contain any contradictions? "
            "Answer only YES or NO.\n\n"
            f"{contents}"
        )
        response = self._llm_fn(prompt).strip().upper()
        return "YES" in response

    def _resolve_contradiction(self, cluster: MemoryCluster) -> Memory:
        """Resolve contradictions by asking the LLM to produce a unified memory."""
        sorted_memories = sorted(
            cluster.memories, key=lambda m: m.updated_at
        )

        contents = "\n".join(
            f"- ({self._format_date(m.updated_at)}): {m.content}"
            for m in sorted_memories
        )

        prompt = (
            "The following memories contradict each other. "
            "Produce a single, accurate memory entry that:\n"
            "1. Prefers the most recent information\n"
            "2. Notes the change if relevant (e.g., 'switched from X to Y')\n"
            "3. Is concise (1-2 sentences)\n\n"
            f"Memories (oldest first):\n{contents}\n\n"
            "Consolidated memory:"
        )
        resolved_content = self._llm_fn(prompt).strip()

        newest = sorted_memories[-1]
        all_tags: set[str] = set()
        for m in cluster.memories:
            all_tags.update(m.tags)

        return Memory(
            memory_id=f"consolidated-{cluster.cluster_id}",
            content=resolved_content,
            source="consolidation:contradiction-resolution",
            created_at=sorted_memories[0].created_at,
            updated_at=time.time(),
            access_count=sum(m.access_count for m in cluster.memories),
            last_accessed=max(m.last_accessed for m in cluster.memories),
            tags=tuple(sorted(all_tags)),
            metadata={
                "merged_from": [m.memory_id for m in cluster.memories],
                "contradiction_resolved": True,
            },
        )

    def _merge_cluster(self, cluster: MemoryCluster) -> Memory:
        """Merge non-contradicting related memories into a single entry."""
        contents = "\n".join(
            f"- {m.content}" for m in cluster.memories
        )

        prompt = (
            "Merge the following related memory entries into a single, "
            "comprehensive entry. Preserve all unique information. "
            "Be concise (1-3 sentences).\n\n"
            f"Memories:\n{contents}\n\n"
            "Merged memory:"
        )
        merged_content = self._llm_fn(prompt).strip()

        all_tags: set[str] = set()
        for m in cluster.memories:
            all_tags.update(m.tags)

        return Memory(
            memory_id=f"consolidated-{cluster.cluster_id}",
            content=merged_content,
            source="consolidation:merge",
            created_at=min(m.created_at for m in cluster.memories),
            updated_at=time.time(),
            access_count=sum(m.access_count for m in cluster.memories),
            last_accessed=max(m.last_accessed for m in cluster.memories),
            tags=tuple(sorted(all_tags)),
            metadata={
                "merged_from": [m.memory_id for m in cluster.memories],
            },
        )

    def _extract_topic(self, memories: list[Memory]) -> str:
        """Extract a short topic label from a group of memories."""
        # Use the first memory's content as a simple topic proxy
        first_content = memories[0].content
        return first_content[:80].split(".")[0]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _format_date(timestamp: float) -> str:
        """Format a timestamp for display."""
        import datetime
        return datetime.datetime.fromtimestamp(
            timestamp, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%d")


# --- Usage Example ---

async def run_nightly_consolidation(
    memory_store,
    embed_fn,
    llm_fn,
):
    """Run memory consolidation as a scheduled background task."""
    consolidator = MemoryConsolidator(
        embed_fn=embed_fn,
        llm_fn=llm_fn,
        similarity_threshold=0.82,
        staleness_days=90,
    )

    # Load all memories from the store
    all_memories = memory_store.get_all()
    print(f"Starting consolidation of {len(all_memories)} memories...")

    consolidated, report = await consolidator.consolidate(all_memories)

    # Replace the memory store contents
    memory_store.replace_all(consolidated)

    print(
        f"Consolidation complete:\n"
        f"  Clusters found: {report.clusters_found}\n"
        f"  Memories merged: {report.memories_merged}\n"
        f"  Contradictions resolved: {report.contradictions_resolved}\n"
        f"  Stale memories pruned: {report.memories_pruned}\n"
        f"  {report.original_count} -> {report.final_count} entries\n"
        f"  Duration: {report.duration_seconds:.1f}s"
    )
```

### TypeScript

```typescript
import { createHash } from "crypto";

interface Memory {
  readonly memoryId: string;
  readonly content: string;
  readonly source: string;
  readonly createdAt: number;
  readonly updatedAt: number;
  readonly accessCount: number;
  readonly lastAccessed: number;
  readonly tags: readonly string[];
  readonly metadata: Record<string, unknown>;
}

interface MemoryCluster {
  readonly clusterId: string;
  readonly topic: string;
  readonly memories: readonly Memory[];
}

interface ConsolidationResult {
  readonly clustersFound: number;
  readonly memoriesMerged: number;
  readonly contradictionsResolved: number;
  readonly memoriesPruned: number;
  readonly originalCount: number;
  readonly finalCount: number;
  readonly durationMs: number;
}

type EmbedFn = (text: string) => number[];
type LLMFn = (prompt: string) => string;

interface ConsolidatorConfig {
  readonly similarityThreshold: number;
  readonly stalenessDays: number;
  readonly minClusterSize: number;
}

const DEFAULT_CONSOLIDATOR_CONFIG: ConsolidatorConfig = {
  similarityThreshold: 0.82,
  stalenessDays: 90,
  minClusterSize: 2,
};

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function pruneStale(
  memories: readonly Memory[],
  stalenessMs: number
): { active: Memory[]; prunedCount: number } {
  const now = Date.now();
  const active: Memory[] = [];
  let prunedCount = 0;

  for (const memory of memories) {
    const lastTouch = Math.max(memory.lastAccessed, memory.updatedAt);
    const age = now - lastTouch;

    if (age < stalenessMs || memory.accessCount > 10) {
      active.push(memory);
    } else {
      prunedCount++;
    }
  }

  return { active, prunedCount };
}

function clusterMemories(
  memories: readonly Memory[],
  embedFn: EmbedFn,
  similarityThreshold: number
): MemoryCluster[] {
  if (memories.length === 0) return [];

  const embeddings = memories.map((m) => embedFn(m.content));
  const assigned = new Set<number>();
  const clusters: MemoryCluster[] = [];

  for (let i = 0; i < memories.length; i++) {
    if (assigned.has(i)) continue;

    const members: Memory[] = [memories[i]];
    assigned.add(i);

    for (let j = i + 1; j < memories.length; j++) {
      if (assigned.has(j)) continue;
      const sim = cosineSimilarity(embeddings[i], embeddings[j]);
      if (sim >= similarityThreshold) {
        members.push(memories[j]);
        assigned.add(j);
      }
    }

    clusters.push({
      clusterId: `cluster-${clusters.length}`,
      topic: members[0].content.slice(0, 80).split(".")[0],
      memories: Object.freeze(members),
    });
  }

  return clusters;
}

function detectContradiction(
  cluster: MemoryCluster,
  llmFn: LLMFn
): boolean {
  if (cluster.memories.length < 2) return false;

  const contents = cluster.memories
    .map(
      (m) =>
        `- [${m.memoryId}] (${new Date(m.createdAt).toISOString().slice(0, 10)}): ${m.content}`
    )
    .join("\n");

  const response = llmFn(
    `Do the following memory entries contain any contradictions? Answer only YES or NO.\n\n${contents}`
  );
  return response.trim().toUpperCase().includes("YES");
}

function mergeCluster(
  cluster: MemoryCluster,
  llmFn: LLMFn,
  resolveContradiction: boolean
): Memory {
  const sorted = [...cluster.memories].sort(
    (a, b) => a.updatedAt - b.updatedAt
  );

  const contents = sorted
    .map(
      (m) =>
        `- (${new Date(m.updatedAt).toISOString().slice(0, 10)}): ${m.content}`
    )
    .join("\n");

  const prompt = resolveContradiction
    ? `The following memories contradict each other. Produce a single, accurate memory that prefers the most recent information and notes the change. Be concise (1-2 sentences).\n\nMemories (oldest first):\n${contents}\n\nConsolidated memory:`
    : `Merge the following related memory entries into a single, comprehensive entry. Preserve all unique information. Be concise (1-3 sentences).\n\nMemories:\n${contents}\n\nMerged memory:`;

  const mergedContent = llmFn(prompt).trim();

  const allTags = new Set<string>();
  for (const m of cluster.memories) {
    for (const tag of m.tags) allTags.add(tag);
  }

  return {
    memoryId: `consolidated-${cluster.clusterId}`,
    content: mergedContent,
    source: resolveContradiction
      ? "consolidation:contradiction-resolution"
      : "consolidation:merge",
    createdAt: Math.min(...cluster.memories.map((m) => m.createdAt)),
    updatedAt: Date.now(),
    accessCount: cluster.memories.reduce((sum, m) => sum + m.accessCount, 0),
    lastAccessed: Math.max(...cluster.memories.map((m) => m.lastAccessed)),
    tags: Object.freeze([...allTags].sort()),
    metadata: {
      mergedFrom: cluster.memories.map((m) => m.memoryId),
      contradictionResolved: resolveContradiction,
    },
  };
}

function consolidate(
  memories: readonly Memory[],
  embedFn: EmbedFn,
  llmFn: LLMFn,
  config: ConsolidatorConfig = DEFAULT_CONSOLIDATOR_CONFIG
): { consolidated: Memory[]; result: ConsolidationResult } {
  const startTime = Date.now();
  const originalCount = memories.length;

  // Step 1: Prune stale
  const stalenessMs = config.stalenessDays * 86_400_000;
  const { active, prunedCount } = pruneStale(memories, stalenessMs);

  // Step 2: Cluster
  const clusters = clusterMemories(
    active,
    embedFn,
    config.similarityThreshold
  );

  // Step 3: Merge and resolve
  const consolidated: Memory[] = [];
  let totalMerged = 0;
  let totalContradictions = 0;

  for (const cluster of clusters) {
    if (cluster.memories.length < config.minClusterSize) {
      consolidated.push(...cluster.memories);
      continue;
    }

    const hasContradiction = detectContradiction(cluster, llmFn);
    if (hasContradiction) totalContradictions++;

    consolidated.push(mergeCluster(cluster, llmFn, hasContradiction));
    totalMerged += cluster.memories.length - 1;
  }

  return {
    consolidated,
    result: {
      clustersFound: clusters.length,
      memoriesMerged: totalMerged,
      contradictionsResolved: totalContradictions,
      memoriesPruned: prunedCount,
      originalCount,
      finalCount: consolidated.length,
      durationMs: Date.now() - startTime,
    },
  };
}

// --- Usage Example ---

function runNightlyConsolidation(
  memories: Memory[],
  embedFn: EmbedFn,
  llmFn: LLMFn
): void {
  console.log(`Starting consolidation of ${memories.length} memories...`);

  const { consolidated, result } = consolidate(memories, embedFn, llmFn);

  console.log(
    [
      "Consolidation complete:",
      `  Clusters found: ${result.clustersFound}`,
      `  Memories merged: ${result.memoriesMerged}`,
      `  Contradictions resolved: ${result.contradictionsResolved}`,
      `  Stale memories pruned: ${result.memoriesPruned}`,
      `  ${result.originalCount} -> ${result.finalCount} entries`,
      `  Duration: ${result.durationMs}ms`,
    ].join("\n")
  );
}

export {
  consolidate,
  Memory,
  MemoryCluster,
  ConsolidationResult,
  ConsolidatorConfig,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Prevents unbounded memory growth | LLM calls for merging and contradiction resolution add cost |
| Resolves contradictions that confuse retrieval | Merging can lose nuance from the original individual memories |
| Reduces retrieval noise by eliminating duplicates | Aggressive pruning may remove memories that become relevant later |
| Consolidated memories are more coherent and useful | Clustering quality depends on embedding model quality |
| Audit trail preserves lineage of merged memories | Running consolidation on large memory stores can be slow |

## When to Use

- Long-lived agents that accumulate hundreds or thousands of memories over weeks and months.
- When retrieval quality is degrading because of noisy, redundant, or contradictory memories.
- Systems where memory storage or embedding search costs scale with entry count.
- When users report the agent "contradicting itself" or "forgetting things it should know."
- As a scheduled background process (nightly, weekly) rather than on every interaction.

## When NOT to Use

- New agents with small memory stores where consolidation overhead is not justified.
- When every individual memory entry must be preserved for audit or compliance reasons.
- Real-time systems where consolidation latency would block user interactions (run it asynchronously).
- When memories are already well-structured (e.g., strictly typed database records that are updated in place).
- When the LLM cost of merging exceeds the storage cost of keeping all memories.

## Related Patterns

- **[Episodic Memory](episodic-memory.md)** -- Episodes are a primary input to consolidation. Related episodes about the same topic can be merged into a single knowledge entry.
- **[Semantic Memory Indexing](semantic-memory-indexing.md)** -- After consolidation, the semantic index must be rebuilt to reflect the merged memory store.
- **[Filesystem-as-Memory](filesystem-as-memory.md)** -- File-based memories can be consolidated by rewriting files that contain redundant or contradictory information.
- **[Cross-Session State Sync](cross-session-state-sync.md)** -- Consolidation results must be propagated to all active sessions.

## Real-World Examples

1. **ChatGPT Memory management** -- OpenAI periodically consolidates user memories, merging related facts and removing outdated information. Users can see and manage their consolidated memories in settings.

2. **Human memory consolidation** -- During sleep, the brain consolidates short-term memories into long-term storage, merging related experiences and discarding irrelevant details. This biological process directly inspires the pattern.

3. **Knowledge graph maintenance** -- Enterprise knowledge graphs require periodic deduplication and conflict resolution as multiple sources contribute overlapping facts.

4. **Git squash merges** -- Squashing many small commits into a single coherent commit before merging to main is conceptually identical: consolidate many incremental changes into one meaningful unit.
