# Context Evaluation Patterns

Context evaluation is the practice of **measuring and monitoring the quality of your context over time**. Building good context is necessary but not sufficient -- you also need to know when context has degraded, become contradictory, or drifted from its intended purpose. Evaluation patterns provide the observability layer for context engineering.

These patterns answer the question: *How do I know when my context is failing, and what do I do about it?*

## Decision Tree

```
Start here: What is your evaluation challenge?
|
|-- "My agent works well initially but degrades over long sessions"
|     --> Context Rot Detection (monitor and remediate context decay)
|
|-- "I suspect my context has contradictory or stale information"
|     --> Context Rot Detection
|
|-- "The model hallucinates when it doesn't have enough context"
|     --> Context Coverage Analysis (detect missing information pre-generation)
|
|-- "I need automated quality checks on context health"
|     --> Context Rot Detection + Context Coverage Analysis
|
|-- "My production agents drift from their instructions over time"
|     --> Context Rot Detection (instruction adherence monitoring)
|
|-- "My context window is bloated and I don't know what to remove"
|     --> Ablation Testing (measure each component's contribution)
|
|-- "I want to optimize my prompt but don't know which parts matter"
|     --> Ablation Testing
|
|-- "I need both real-time monitoring AND offline optimization"
|     --> Context Rot Detection (online) + Ablation Testing (offline)
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Context Rot Detection](context-rot-detection.md) | Monitor context health and trigger remediation when quality degrades | Reliability over long-running sessions |
| [Context Coverage Analysis](context-coverage-analysis.md) | Check whether context contains all information needed for the current query | Prevent hallucination from missing context |
| [Ablation Testing](ablation-testing.md) | Systematically remove context components to measure their contribution | Data-driven context optimization (30-50% token savings typical) |

## How They Compose

These patterns evaluate context from different angles and timeframes:

- **Context Rot Detection** is a runtime monitor. It watches context health continuously during long sessions, detecting instruction drift, contradiction accumulation, and staleness. It answers: *Is my context degrading right now?*
- **Context Coverage Analysis** is a per-query pre-flight check. Before each LLM call, it verifies that the context contains the information needed to answer the current query. It answers: *Do I have what I need for this specific request?*
- **Ablation Testing** is an offline analysis tool. It runs periodically (not per-request) to measure which context components actually contribute to output quality. It answers: *Which parts of my context are worth keeping?*

Together they form a complete evaluation strategy: ablation testing **designs** the optimal context (what to include), coverage analysis **validates** the context per-query (is everything present), and rot detection **monitors** the context over time (is it still healthy).

## Why Evaluation Matters

Context engineering without evaluation is like software engineering without tests. You might build something that works initially, but you have no way to know when it breaks. This is especially dangerous because context degradation is *silent* -- the model does not tell you it has forgotten your instructions or that its context is full of contradictions. It simply produces worse output.

Evaluation patterns close this feedback loop. They provide automated detection of context problems, enabling either human-in-the-loop review or automatic remediation. As context engineering matures as a discipline, expect this category to grow with patterns for A/B testing context strategies, measuring context ROI, and benchmarking context quality across providers.
