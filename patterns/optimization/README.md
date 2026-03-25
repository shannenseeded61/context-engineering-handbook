# Context Optimization Patterns

Context optimization is the practice of **making the most of every token in your context window** -- reducing cost, improving latency, and preserving the highest-signal information. Optimization patterns do not change *what* you put in context; they change *how efficiently* that context is structured, cached, and maintained.

These patterns answer the question: *How do I get better results from the same (or fewer) tokens?*

## Decision Tree

```
Start here: What is your optimization challenge?
|
|-- "My LLM calls are slow and expensive, especially repeated ones"
|     --> KV-Cache Optimization (maximize cache hit rates)
|
|-- "My agent keeps repeating the same mistakes in a session"
|     --> Error Preservation (keep full error context available)
|
|-- "I'm making many similar calls and paying full price each time"
|     --> KV-Cache Optimization + Prompt Caching Strategies
|
|-- "Errors get summarized away and the agent loses debugging context"
|     --> Error Preservation
|
|-- "Users ask similar questions and I pay for full inference every time"
|     --> Prompt Caching Strategies (multi-level application cache)
|
|-- "Context assembly from multiple sources is slow and sequential"
|     --> Parallel Context Assembly (concurrent fetches with degradation)
|
|-- "My agent loop rebuilds context from scratch every turn"
|     --> Incremental Context Updates (diff-based patching)
|
|-- "I need both cost efficiency AND robust self-correction"
|     --> KV-Cache Optimization + Error Preservation
|
|-- "I need fast context assembly AND minimal per-turn overhead"
|     --> Parallel Context Assembly + Incremental Context Updates
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [KV-Cache Optimization](kv-cache-optimization.md) | Structure context for maximum cache reuse across LLM calls | Latency and cost reduction (up to 7x) |
| [Error Preservation](error-preservation.md) | Preserve full error context instead of summarizing it away | Faster self-correction, fewer repeated mistakes |
| [Prompt Caching Strategies](prompt-caching-strategies.md) | Multi-level application cache (exact, semantic, component) for LLM interactions | Cost reduction (30-60%) on repeated or similar queries |
| [Parallel Context Assembly](parallel-context-assembly.md) | Fetch context from multiple sources concurrently with graceful degradation | Latency reduction from sum-of-all to max-of-any (3-5x) |
| [Incremental Context Updates](incremental-context-updates.md) | Apply diffs instead of rebuilding context from scratch each turn | Eliminate redundant computation (90%+ reuse typical) |

## How They Compose

These patterns optimize different dimensions of context efficiency and work together naturally:

- **KV-Cache Optimization** is a structural optimization. It focuses on *how context is arranged* so that LLM providers can reuse cached computations. The key insight is that identical prefixes are free after the first call.
- **Error Preservation** is a signal-quality optimization. It focuses on *what stays in context* when space is limited. The key insight is that error details are the highest-value tokens for self-correction, yet they are often the first to be compressed away.
- **Prompt Caching Strategies** is an application-level optimization. It caches prompt-response pairs at multiple fidelity levels (exact match, semantic similarity, component-level) to avoid redundant LLM calls entirely. It complements KV-cache optimization: KV-cache makes calls cheaper, prompt caching avoids calls altogether.
- **Parallel Context Assembly** is a latency optimization. It focuses on *how fast context is gathered* by dispatching independent fetches concurrently. The key insight is that most context sources are independent and can be fetched in parallel.
- **Incremental Context Updates** is a per-turn efficiency optimization. It focuses on *how little work is needed between turns* by tracking what changed and patching only the delta. The key insight is that context is 90%+ stable between turns in most agent loops.

A fully optimized system might use all five: **parallel assembly** gathers context from multiple sources on the first turn, **incremental updates** minimize work on subsequent turns, **KV-cache optimization** structures the assembled context for provider-side caching, **prompt caching** catches repeated queries at the application level, and **error preservation** ensures debugging context survives all of these optimizations.
