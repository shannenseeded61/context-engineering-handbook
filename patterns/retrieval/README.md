# Context Retrieval Patterns

Context retrieval is the practice of **fetching the right external knowledge** at the right time and assembling it into a form the model can use effectively. Rather than stuffing everything into the prompt upfront or hoping the model knows the answer, retrieval patterns dynamically pull in relevant information from vector stores, tool registries, APIs, and file systems.

These patterns answer the question: *How do I get the right information into the context window when the model needs it?*

## Decision Tree

```
Start here: What is your retrieval challenge?
|
|-- "I'm pre-loading too much context that might never be used"
|     --> Just-in-Time Retrieval
|
|-- "I'm doing RAG but my retrieved chunks are noisy, redundant, or blow my token budget"
|     --> RAG Context Assembly
|
|-- "My agent has too many tools to fit all their descriptions in the prompt"
|     --> Semantic Tool Selection
|
|-- "No single retrieval method covers all my query types (keyword, semantic, relational)"
|     --> Hybrid Search Fusion
|
|-- "My retrieved results are relevant to the query but not to the conversation context"
|     --> Context-Aware Re-ranking
|
|-- "My corpus has versioned or time-sensitive content and users get stale answers"
|     --> Temporal Context Selection
|
|-- "I need lazy-loaded context AND clean chunk assembly"
|     --> Just-in-Time Retrieval + RAG Context Assembly
|
|-- "My agent needs to pick tools AND retrieve knowledge for those tools"
|     --> Semantic Tool Selection + RAG Context Assembly
|
|-- "I need broad recall from multiple sources AND precise ordering for my conversation"
|     --> Hybrid Search Fusion + Context-Aware Re-ranking
|
|-- "I have versioned docs AND multi-turn conversations"
|     --> Temporal Context Selection + Context-Aware Re-ranking
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Just-in-Time Retrieval](just-in-time-retrieval.md) | Fetch context only at the moment it's needed | Freshness and token efficiency |
| [RAG Context Assembly](rag-context-assembly.md) | Retrieve, rank, deduplicate, and budget retrieved chunks | Retrieval quality and coherence |
| [Semantic Tool Selection](semantic-tool-selection.md) | Dynamically select relevant tool descriptions via embedding similarity | Scalability with large tool catalogs |
| [Hybrid Search Fusion](hybrid-search-fusion.md) | Combine keyword, semantic, and graph retrieval with Reciprocal Rank Fusion | Eliminates single-retriever blind spots |
| [Context-Aware Re-ranking](context-aware-reranking.md) | Re-rank results using full conversation context, not just the latest query | Resolves ambiguity in multi-turn conversations |
| [Temporal Context Selection](temporal-context-selection.md) | Prioritize recent content, apply time-decay, and resolve versioned documents | Prevents stale and version-confused answers |

## How They Compose

These six patterns address different stages and dimensions of the retrieval pipeline:

- **Just-in-Time Retrieval** decides *when* to fetch. It prevents wasteful pre-loading by triggering retrieval only when the conversation state or user intent demands it.
- **RAG Context Assembly** decides *what makes it into the prompt*. Once retrieval fires, raw chunks need ranking, deduplication, token budgeting, and source attribution before they belong in a context window.
- **Semantic Tool Selection** is a specialized retrieval problem for *capabilities rather than knowledge*. When an agent has dozens or hundreds of tools, this pattern narrows the tool menu to only what is relevant.
- **Hybrid Search Fusion** addresses *retrieval breadth*. By combining multiple retrieval strategies (keyword, semantic, graph) and fusing their rankings, it ensures no query type falls through the cracks of a single retriever.
- **Context-Aware Re-ranking** addresses *conversational precision*. After first-stage retrieval produces candidates, re-ranking with conversation history, established entities, and task state produces dramatically better relevance ordering.
- **Temporal Context Selection** addresses *time correctness*. When versioned or time-sensitive content exists, it ensures the system returns the right version and suppresses stale or deprecated information.

A production agent often combines several of these: tool descriptions are selected semantically (Semantic Tool Selection), knowledge retrieval is deferred until the agent actually needs it (Just-in-Time), multiple search backends are fused for broad recall (Hybrid Search Fusion), results are filtered for temporal correctness (Temporal Context Selection), re-ranked against conversation context (Context-Aware Re-ranking), and assembled carefully before injection (RAG Context Assembly).
