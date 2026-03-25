<div align="center">

# Context Engineering Handbook

### The practitioner's guide to building effective context for AI agents and LLM applications.

**15 patterns** | **7 categories** | **Python + TypeScript** | **Benchmarks** | **4 framework integrations**

[![GitHub Stars](https://img.shields.io/github/stars/ypollak2/context-engineering-handbook?style=flat&logo=github&label=Stars)](https://github.com/ypollak2/context-engineering-handbook/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Patterns](https://img.shields.io/badge/Patterns-15%20shipped-orange.svg)](#pattern-catalog)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](#examples)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](#examples)

</div>

---

Context engineering is the discipline of building the right information environment so an LLM can solve your actual problem. It was [named by Tobi Lutke](https://x.com/tobi/status/1925629163972653200) and [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) in 2025, and it's quickly becoming the single most important skill in AI engineering.

**This is not another blog post or awesome-list.** This is a pattern catalog: 15 battle-tested patterns with runnable code, decision frameworks, and documented anti-patterns. Pick a problem, find the pattern, ship it.

---

## Table of Contents

- [Why Context Engineering](#why-context-engineering)
- [Quick Start](#quick-start)
- [Pattern Catalog](#pattern-catalog)
- [Interactive Decision Tree](#interactive-decision-tree)
- [Pattern Structure](#how-each-pattern-is-structured)
- [Anti-Patterns](#anti-patterns)
- [Benchmarks](#benchmarks)
- [Framework Integrations](#framework-integrations)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Star History](#star-history)
- [License](#license)

## Why Context Engineering

Most LLM failures aren't model failures -- they're context failures. You gave the model the wrong information, too much information, or information in the wrong structure. Context engineering fixes this systematically.

Anthropic, Manus, LangChain, and others have published foundational articles on the topic. But until now, there was no single resource that combines a **comprehensive taxonomy** + **runnable code** + **decision frameworks** for practitioners who ship AI to production.

## Quick Start

**Find the right pattern for your problem:**

```
Your agent is forgetting things mid-conversation?
  --> Conversation Compaction (#7) or Episodic Memory (#11)

Your RAG pipeline returns relevant chunks but the LLM still hallucinates?
  --> RAG Context Assembly (#5) or Few-Shot Curation (#3)

Your system prompt is a wall of text and the model ignores half of it?
  --> System Prompt Architecture (#1) or Progressive Disclosure (#2)

Your agent calls the wrong tools?
  --> Semantic Tool Selection (#6) or Observation Masking (#8)

Your multi-agent system produces inconsistent results?
  --> Sub-Agent Delegation (#9) or Multi-Agent Context Orchestration (#10)

Your context window is filling up and responses are degrading?
  --> KV-Cache Optimization (#13) or Context Rot Detection (#15)

Your agent keeps repeating the same mistakes?
  --> Error Preservation (#14) or Filesystem-as-Memory (#12)
```

Or use the [Interactive Decision Tree](interactive/) for a guided walkthrough.

## Pattern Catalog

### Construction -- Building context from scratch

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 1 | [System Prompt Architecture](patterns/construction/system-prompt-architecture.md) | Structure system prompts for maximum instruction adherence | Low |
| 2 | [Progressive Disclosure](patterns/construction/progressive-disclosure.md) | Reveal context incrementally based on task state | Medium |
| 3 | [Few-Shot Curation](patterns/construction/few-shot-curation.md) | Select and order examples for optimal in-context learning | Medium |

### Retrieval -- Pulling the right context at the right time

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 4 | [Just-in-Time Retrieval](patterns/retrieval/just-in-time-retrieval.md) | Fetch context only when the model signals it needs it | Medium |
| 5 | [RAG Context Assembly](patterns/retrieval/rag-context-assembly.md) | Assemble retrieved chunks into coherent, structured context | High |
| 6 | [Semantic Tool Selection](patterns/retrieval/semantic-tool-selection.md) | Dynamically select which tools to present based on the task | Medium |

### Compression -- Fitting more signal into fewer tokens

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 7 | [Conversation Compaction](patterns/compression/conversation-compaction.md) | Summarize conversation history without losing critical details | Medium |
| 8 | [Observation Masking](patterns/compression/observation-masking.md) | Filter tool outputs to keep only what matters | Low |

### Isolation -- Scoping context to prevent contamination

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 9 | [Sub-Agent Delegation](patterns/isolation/sub-agent-delegation.md) | Spawn focused sub-agents with minimal, task-specific context | High |
| 10 | [Multi-Agent Context Orchestration](patterns/isolation/multi-agent-context-orchestration.md) | Coordinate context flow across multiple collaborating agents | High |

### Persistence -- Remembering across sessions and runs

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 11 | [Episodic Memory](patterns/persistence/episodic-memory.md) | Store and retrieve task-specific memories across sessions | Medium |
| 12 | [Filesystem-as-Memory](patterns/persistence/filesystem-as-memory.md) | Use structured files as durable, inspectable agent memory | Low |

### Optimization -- Squeezing more performance from your context budget

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 13 | [KV-Cache Optimization](patterns/optimization/kv-cache-optimization.md) | Structure prompts to maximize key-value cache hit rates | Medium |
| 14 | [Error Preservation](patterns/optimization/error-preservation.md) | Persist error context to prevent repeated failures | Low |

### Evaluation -- Measuring context quality over time

| # | Pattern | Description | Complexity |
|---|---------|-------------|------------|
| 15 | [Context Rot Detection](patterns/evaluation/context-rot-detection.md) | Detect when accumulated context degrades model performance | High |

<details>
<summary><strong>Coming in v2</strong> (20 additional patterns)</summary>

| Category | Pattern | Status |
|----------|---------|--------|
| Construction | Dynamic Persona Assembly | Planned |
| Construction | Schema-Guided Generation | Planned |
| Construction | Template Composition | Planned |
| Construction | Constraint Injection | Planned |
| Retrieval | Hybrid Search Fusion | Planned |
| Retrieval | Context-Aware Re-ranking | Planned |
| Retrieval | Temporal Context Selection | Planned |
| Compression | Hierarchical Summarization | Planned |
| Compression | Token Budget Allocation | Planned |
| Compression | Lossy Context Distillation | Planned |
| Isolation | Sandbox Contexts | Planned |
| Isolation | Role-Based Context Partitioning | Planned |
| Persistence | Semantic Memory Indexing | Planned |
| Persistence | Cross-Session State Sync | Planned |
| Persistence | Memory Consolidation | Planned |
| Optimization | Prompt Caching Strategies | Planned |
| Optimization | Parallel Context Assembly | Planned |
| Optimization | Incremental Context Updates | Planned |
| Evaluation | Context Coverage Analysis | Planned |
| Evaluation | Ablation Testing | Planned |

</details>

## Interactive Decision Tree

Not sure which pattern to use? The [Interactive Decision Tree](interactive/) walks you through a series of questions about your problem and recommends the best pattern.

```
What are you trying to solve?
  |
  |-- Agent isn't following instructions --> Construction patterns
  |-- Agent lacks the right knowledge   --> Retrieval patterns
  |-- Context window filling up         --> Compression patterns
  |-- Cross-contamination between tasks --> Isolation patterns
  |-- Agent forgets between sessions    --> Persistence patterns
  |-- Slow or expensive inference       --> Optimization patterns
  |-- Quality degrading over time       --> Evaluation patterns
```

## How Each Pattern is Structured

Every pattern follows a consistent template so you can evaluate and implement quickly:

```
patterns/<category>/<pattern-name>.md    # Full pattern documentation with inline code
```

Each pattern includes:

| Section | Purpose |
|---------|---------|
| **Problem** | The specific failure mode this pattern addresses |
| **Context** | When you'd encounter this problem |
| **Solution** | The pattern itself, with architecture diagram |
| **Implementation** | Step-by-step guide with code |
| **Decision Tree** | When to use this vs. alternatives |
| **Anti-Patterns** | Common mistakes when applying this pattern |
| **Metrics** | How to measure if it's working |
| **References** | Papers, blog posts, prior art |

## Anti-Patterns

Knowing what NOT to do is just as important. The [anti-patterns directory](anti-patterns/) documents common context engineering mistakes:

- **The Kitchen Sink** -- Dumping everything into the system prompt
- **Context Amnesia** -- Losing critical details during compaction
- **The Echo Chamber** -- Agent outputs become repetitive over long sessions
- **Stale Context Poisoning** -- Retrieved context is outdated but presented as current
- **Tool Schema Overload** -- Including all tool schemas regardless of relevance
- **The Infinite Loop** -- Retrying failures with no new information
- **Context Isolation Neglect** -- Running all work in a single context window

## Benchmarks

Measure how well your context engineering is working with 5 benchmarks:

| Benchmark | What It Measures | Low Score Means |
|-----------|-----------------|-----------------|
| [Needle in Haystack](benchmarks/) | Fact retrieval across context positions | Apply Progressive Disclosure |
| [Instruction Adherence](benchmarks/) | System prompt rule compliance | Apply System Prompt Architecture |
| [Compression Fidelity](benchmarks/) | Info preservation after compaction | Apply Conversation Compaction |
| [Retrieval Relevance](benchmarks/) | Retrieved chunk usefulness | Apply RAG Context Assembly |
| [Token Efficiency](benchmarks/) | Signal-to-noise ratio | Apply Observation Masking |

```bash
# Python
cd benchmarks/python && pip install -r requirements.txt
python runner.py --all --model gpt-4o

# TypeScript
cd benchmarks/typescript && npm install
npx tsx src/runner.ts --all --model gpt-4o
```

See the [benchmarks README](benchmarks/README.md) for score interpretation and full docs.

## Framework Integrations

Apply handbook patterns using your framework of choice:

| Framework | Patterns | Languages |
|-----------|----------|-----------|
| [LangChain](integrations/langchain/) | Progressive Disclosure, Conversation Compaction, RAG Assembly, Tool Selection, Sub-Agent Delegation | Python, TypeScript |
| [LlamaIndex](integrations/llamaindex/) | RAG Assembly, Episodic Memory, Context Rot Detection | Python, TypeScript |
| [Semantic Kernel](integrations/semantic-kernel/) | System Prompt Architecture, Tool Selection, KV-Cache Optimization | Python |
| [Vercel AI SDK](integrations/vercel-ai-sdk/) | Progressive Disclosure, Conversation Compaction, Error Preservation | TypeScript |

See the [integrations README](integrations/README.md) for setup guides and full docs.

## Examples

Every pattern ships with runnable examples in both Python and TypeScript.

**Python:**

```bash
cd examples/python
pip install -r requirements.txt
python run_example.py --pattern system-prompt-architecture
```

**TypeScript:**

```bash
cd examples/typescript
npm install
npx tsx run-example.ts --pattern system-prompt-architecture
```

Browse all examples in the [examples directory](examples/).

## Roadmap

- [x] **v1.0** -- 15 core patterns with Python + TypeScript examples
- [x] **v1.1** -- Interactive decision tree (HTML/JS)
- [x] **v1.2** -- Anti-patterns documentation (7 anti-patterns)
- [ ] **v2.0** -- 20 additional patterns (35 total)
- [x] **v2.1** -- Benchmark suite for context quality evaluation
- [x] **v2.2** -- Framework integrations (LangChain, LlamaIndex, Semantic Kernel, Vercel AI SDK)
- [ ] **v3.0** -- Visual context debugger

## Contributing

Context engineering is a young discipline and evolving fast. Contributions are welcome.

**Ways to contribute:**

- Add a new pattern (use the [pattern template](patterns/TEMPLATE.md))
- Improve an existing pattern's examples or documentation
- Add an anti-pattern you've encountered in production
- Port examples to additional languages (Go, Rust, Java)
- Fix bugs or improve clarity

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

<!-- ## Star History -->

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=ypollak2/context-engineering-handbook&type=Date)](https://star-history.com/#ypollak2/context-engineering-handbook&Date) -->

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for developers who ship AI to production.**

[Report an Issue](https://github.com/ypollak2/context-engineering-handbook/issues) | [Request a Pattern](https://github.com/ypollak2/context-engineering-handbook/issues/new?labels=pattern-request) | [Discussions](https://github.com/ypollak2/context-engineering-handbook/discussions)

</div>
