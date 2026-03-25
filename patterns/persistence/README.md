# Context Persistence Patterns

Context persistence is the practice of **preserving knowledge and decisions across sessions** so that agents and LLM-powered systems do not start from zero every time. Without persistence, every conversation is amnesia -- the model re-asks questions it already answered, re-discovers preferences it already learned, and repeats mistakes it already corrected.

These patterns answer the question: *How do I give my system durable memory that survives session boundaries?*

## Decision Tree

```
Start here: What is your persistence challenge?
|
|-- "My agent forgets everything between sessions"
|     --> Episodic Memory (capture and recall past session episodes)
|
|-- "I need simple, transparent memory that humans can read and edit"
|     --> Filesystem-as-Memory (structured files on disk)
|
|-- "I want to find memories by meaning, not keywords"
|     --> Semantic Memory Indexing (vector-based retrieval)
|
|-- "The same agent runs on multiple devices and they disagree"
|     --> Cross-Session State Sync (synchronize across instances)
|
|-- "My memory store is bloated with duplicates and contradictions"
|     --> Memory Consolidation (merge, resolve, prune)
|
|-- "I want rich semantic recall of past experiences"
|     --> Episodic Memory
|
|-- "I need memory that works with git, code review, and existing tools"
|     --> Filesystem-as-Memory
|
|-- "I need both human-editable knowledge AND semantic recall"
|     --> Filesystem-as-Memory (primary) + Semantic Memory Indexing (retrieval layer)
|
|-- "My agent contradicts itself because old and new memories conflict"
|     --> Memory Consolidation
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Episodic Memory](episodic-memory.md) | Store and retrieve complete session snapshots indexed for semantic search | Rich contextual recall across sessions |
| [Filesystem-as-Memory](filesystem-as-memory.md) | Use structured files on disk as a persistent, human-readable memory store | Transparency, simplicity, and tool compatibility |
| [Semantic Memory Indexing](semantic-memory-indexing.md) | Index all memories using embeddings for retrieval by meaning | Find knowledge by what it means, not where it is stored |
| [Cross-Session State Sync](cross-session-state-sync.md) | Synchronize agent state across concurrent sessions and instances | Consistent behavior across devices and contexts |
| [Memory Consolidation](memory-consolidation.md) | Periodically merge related memories, resolve contradictions, and prune stale entries | Prevent memory bloat and maintain coherence over time |

## How They Compose

These five patterns represent different layers of a complete memory system:

- **Filesystem-as-Memory** is the simplest, most transparent approach. Memory lives in files that humans can read, edit, and version-control. It works with existing tools (editors, grep, git) and requires no additional infrastructure. This is where most projects should start.
- **Episodic Memory** adds semantic indexing on top of raw storage. It captures richer context (tool usage, decision rationale, outcomes) and enables similarity-based retrieval. It shines when the volume of past experience is large enough that file browsing becomes impractical.
- **Semantic Memory Indexing** provides a unified retrieval layer across all memory types. It sits alongside (not replacing) filesystem and episodic storage, enabling "what do I know about X?" queries that work regardless of how or where the memory was stored.
- **Cross-Session State Sync** handles the concurrent access problem. When multiple sessions or devices interact with the same agent, the sync layer ensures that knowledge learned in one session is available in all others, with proper conflict resolution.
- **Memory Consolidation** is the maintenance layer. Over time, memories accumulate redundancy, contradictions, and staleness. Consolidation runs periodically to merge, resolve, and prune, keeping the memory store coherent and efficient.

A mature system layers all five: filesystem-based memory for structured knowledge, episodic memory for interaction history, semantic indexing for retrieval, cross-session sync for consistency, and consolidation for long-term health. Each layer can be adopted independently and incrementally.
