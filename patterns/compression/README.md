# Context Compression Patterns

Context compression is the practice of **reducing the token footprint of existing context** without losing the information the model needs to continue performing well. As conversations grow, tool outputs accumulate, and agent loops iterate, the context window fills up. Compression patterns reclaim that space by summarizing, truncating, or masking content that has already served its purpose.

These patterns answer the question: *How do I keep a long-running session from hitting the context ceiling?*

## Decision Tree

```
Start here: What is filling your context window?
|
|-- "The conversation history is too long; old turns are crowding out new ones"
|     |
|     |-- "I need recent turns at full detail but can tolerate lossy older context"
|     |     --> Hierarchical Summarization
|     |
|     |-- "A single summary of old turns is sufficient"
|     |     --> Conversation Compaction
|
|-- "Tool outputs and observations are bloating the context"
|     --> Observation Masking
|
|-- "Multiple components (history, RAG, tools) are all competing for space"
|     --> Token Budget Allocation (to manage the overall budget)
|         + per-component strategies (Compaction, Masking, Distillation)
|
|-- "I need lossless compression, not summarization"
|     --> Observation Masking (deterministic, no LLM call)
|
|-- "I need the model to retain nuanced decisions from earlier in the session"
|     --> Conversation Compaction (LLM-powered summarization preserves semantics)
|
|-- "I have a clear current task and need maximum compression"
|     --> Lossy Context Distillation (extracts only task-relevant facts)
|
|-- "Both conversation turns AND tool outputs are growing unbounded"
|     --> Observation Masking first (cheaper), then Conversation Compaction
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Conversation Compaction](conversation-compaction.md) | Summarize older conversation turns into condensed fact blocks | Unbounded session length with semantic preservation |
| [Observation Masking](observation-masking.md) | Selectively hide or truncate tool outputs that have been consumed | 50% cost savings without LLM summarization overhead |
| [Hierarchical Summarization](hierarchical-summarization.md) | Summarize context at multiple levels of detail -- full for recent, medium for mid-range, brief for old | Preserves temporal resolution gradient while bounding total tokens |
| [Token Budget Allocation](token-budget-allocation.md) | Explicitly budget the context window across competing components with enforced limits | Predictable, observable context window management across all sources |
| [Lossy Context Distillation](lossy-context-distillation.md) | Extract only task-relevant facts from context, discarding everything else | Maximum compression (90%+) when the current task is well-defined |

## How They Compose

These five patterns operate on different aspects of the context management problem and complement each other naturally:

- **Observation Masking** is the first line of defense. It targets tool outputs (file contents, API responses, command results) that were useful when they appeared but are no longer needed. It is fast, deterministic, and requires no additional LLM calls.
- **Conversation Compaction** is the next intervention. It targets the conversational turns themselves, distilling multi-turn reasoning chains into compact summaries that preserve decisions, facts, and user preferences.
- **Hierarchical Summarization** extends compaction with a resolution gradient -- recent turns stay verbatim, mid-range turns get medium-detail summaries, and old turns get brief summaries. Use it when a flat compaction loses too much recent nuance.
- **Token Budget Allocation** sits above the other patterns as an orchestration layer. It partitions the context window into named slots (system prompt, history, RAG, tools, output reserve) and triggers the appropriate compression strategy when any slot exceeds its budget.
- **Lossy Context Distillation** provides maximum compression when you have a clear current task. Unlike summarization (which preserves structure), distillation extracts only the facts relevant to the task at hand. It is the most aggressive compression option and is best used when context pressure is severe.

A production agent typically layers these patterns: observation masking runs continuously to keep tool outputs lean, conversation compaction or hierarchical summarization triggers periodically for history management, token budget allocation coordinates the overall strategy, and lossy distillation is applied when a specific sub-task needs a maximally focused context. Together, they can sustain sessions that would otherwise exhaust the context window within minutes.
