# Parallel Context Assembly

> Assemble context from multiple sources concurrently rather than sequentially, reducing latency from the sum of all fetch times to the maximum of any single fetch.

## Problem

Building a rich context for an LLM call often requires data from multiple independent sources:

- **RAG retrieval**: Search a vector store for relevant documents (200-500ms)
- **Tool schema loading**: Fetch available tool definitions from a registry (50-100ms)
- **Memory recall**: Load conversation history and user preferences from a database (100-300ms)
- **User profile lookup**: Retrieve user-specific configuration and permissions (50-150ms)
- **External API calls**: Fetch real-time data from third-party services (200-2000ms)

When these fetches run sequentially, total latency is the **sum** of all fetch times. For five sources averaging 300ms each, that is 1,500ms of dead time before the LLM call even begins. In agent loops that assemble context on every iteration, this compounds into seconds of pure waiting.

The sources are independent -- the RAG results do not depend on the user profile, and the tool schemas do not depend on the memory recall. Yet sequential execution treats them as if each depends on the last.

## Solution

Dispatch all independent context fetches concurrently and merge the results. Use timeouts and fallbacks so that a slow or failing source does not block the entire assembly. The total latency becomes the **maximum** of any single fetch rather than the sum.

Structure the assembler around three principles:

1. **Concurrent dispatch**: Launch all independent fetches simultaneously.
2. **Timeout isolation**: Each source has its own timeout. If one source is slow, the others are already complete.
3. **Graceful degradation**: If a source fails or times out, continue with partial context rather than failing entirely. Mark the missing source so the LLM knows what context is unavailable.

## How It Works

```
Sequential assembly (bad):
  RAG -------->|
               Tool Schema -->|
                              Memory ------>|
                                            Profile ->|
                                                      API -------->|
  |<------------------- total: sum of all (1500ms) ----------------->|

Parallel assembly (good):
  RAG -------->|
  Tool Schema -->|
  Memory ------>|
  Profile ->|
  API -------->|
  |<-- total: max of any (500ms) -->|

With timeout + degradation:
  RAG -------->|    (200ms, success)
  Tool Schema -->|  (100ms, success)
  Memory ------>|   (300ms, success)
  Profile ->|       (50ms, success)
  API --------X     (timeout at 500ms, degraded)
  |<-- total: 500ms -->|
  Context assembled with 4/5 sources + degradation notice
```

## Implementation

### Python

```python
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable


class SourceStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class SourceResult:
    """Result from a single context source fetch."""
    name: str
    status: SourceStatus
    content: str
    latency_ms: float
    error_message: str = ""
    token_estimate: int = 0

    @property
    def is_available(self) -> bool:
        return self.status == SourceStatus.SUCCESS


@dataclass(frozen=True)
class AssembledContext:
    """The fully assembled context from all sources."""
    sources: tuple[SourceResult, ...]
    total_latency_ms: float
    degraded: bool
    degradation_notice: str = ""

    @property
    def available_sources(self) -> tuple[SourceResult, ...]:
        return tuple(s for s in self.sources if s.is_available)

    @property
    def failed_sources(self) -> tuple[SourceResult, ...]:
        return tuple(s for s in self.sources if not s.is_available)

    @property
    def total_tokens_estimate(self) -> int:
        return sum(s.token_estimate for s in self.available_sources)

    def to_context_string(self) -> str:
        """Merge all available source content into a single context string."""
        sections = []
        for source in self.available_sources:
            if source.content:
                sections.append(f"## {source.name}\n{source.content}")

        if self.degraded:
            sections.append(
                f"## Context Assembly Notice\n{self.degradation_notice}"
            )

        return "\n\n".join(sections)


@dataclass(frozen=True)
class ContextSourceConfig:
    """Configuration for a single context source."""
    name: str
    fetch_fn: Callable[..., Awaitable[str]]
    timeout_seconds: float = 2.0
    required: bool = False
    priority: int = 0  # Higher = assembled first in context


class ParallelContextAssembler:
    """
    Assembles context from multiple independent sources concurrently.

    Each source is fetched in parallel with its own timeout. Failed or
    timed-out sources degrade gracefully, and the assembled context
    indicates which sources are missing.
    """

    def __init__(self, sources: list[ContextSourceConfig]):
        self._sources = sorted(sources, key=lambda s: -s.priority)
        self._stats: list[dict[str, Any]] = []

    async def assemble(self, **kwargs: Any) -> AssembledContext:
        """
        Fetch all sources concurrently and merge results.

        Args:
            **kwargs: Passed to each source's fetch function.

        Returns:
            AssembledContext with all available source content.
        """
        start = time.monotonic()

        tasks = [
            self._fetch_source(source, **kwargs)
            for source in self._sources
        ]
        results = await asyncio.gather(*tasks)

        total_latency = (time.monotonic() - start) * 1000
        sorted_results = tuple(results)

        # Check for degradation
        failed = [r for r in sorted_results if not r.is_available]
        required_failures = [
            r for r in failed
            if any(s.name == r.name and s.required for s in self._sources)
        ]

        if required_failures:
            raise ContextAssemblyError(
                f"Required source(s) failed: "
                f"{', '.join(r.name for r in required_failures)}"
            )

        degraded = len(failed) > 0
        degradation_notice = ""
        if degraded:
            notices = [
                f"- {r.name}: {r.status.value}"
                f"{' (' + r.error_message + ')' if r.error_message else ''}"
                for r in failed
            ]
            degradation_notice = (
                "The following context sources were unavailable:\n"
                + "\n".join(notices)
            )

        assembled = AssembledContext(
            sources=sorted_results,
            total_latency_ms=total_latency,
            degraded=degraded,
            degradation_notice=degradation_notice,
        )

        self._stats = [
            *self._stats,
            {
                "latency_ms": total_latency,
                "sources_total": len(sorted_results),
                "sources_available": len(assembled.available_sources),
                "degraded": degraded,
            },
        ]

        return assembled

    async def _fetch_source(
        self, source: ContextSourceConfig, **kwargs: Any
    ) -> SourceResult:
        """Fetch a single source with timeout and error handling."""
        start = time.monotonic()
        try:
            content = await asyncio.wait_for(
                source.fetch_fn(**kwargs),
                timeout=source.timeout_seconds,
            )
            latency = (time.monotonic() - start) * 1000
            return SourceResult(
                name=source.name,
                status=SourceStatus.SUCCESS,
                content=content,
                latency_ms=latency,
                token_estimate=len(content) // 4,
            )
        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            return SourceResult(
                name=source.name,
                status=SourceStatus.TIMEOUT,
                content="",
                latency_ms=latency,
                error_message=f"Timed out after {source.timeout_seconds}s",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return SourceResult(
                name=source.name,
                status=SourceStatus.ERROR,
                content="",
                latency_ms=latency,
                error_message=str(e),
            )

    @property
    def average_latency_ms(self) -> float:
        if not self._stats:
            return 0.0
        return sum(s["latency_ms"] for s in self._stats) / len(self._stats)

    @property
    def degradation_rate(self) -> float:
        if not self._stats:
            return 0.0
        degraded = sum(1 for s in self._stats if s["degraded"])
        return (degraded / len(self._stats)) * 100


class ContextAssemblyError(Exception):
    """Raised when a required context source fails."""
    pass
```

### TypeScript

```typescript
type SourceStatus = "success" | "timeout" | "error" | "skipped";

interface SourceResult {
  readonly name: string;
  readonly status: SourceStatus;
  readonly content: string;
  readonly latencyMs: number;
  readonly errorMessage: string;
  readonly tokenEstimate: number;
}

interface AssembledContext {
  readonly sources: readonly SourceResult[];
  readonly totalLatencyMs: number;
  readonly degraded: boolean;
  readonly degradationNotice: string;
}

interface ContextSourceConfig {
  readonly name: string;
  readonly fetchFn: (params: Record<string, unknown>) => Promise<string>;
  readonly timeoutMs?: number;
  readonly required?: boolean;
  readonly priority?: number;
}

interface AssemblyStats {
  latencyMs: number;
  sourcesTotal: number;
  sourcesAvailable: number;
  degraded: boolean;
}

class ContextAssemblyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContextAssemblyError";
  }
}

function availableSources(ctx: AssembledContext): readonly SourceResult[] {
  return ctx.sources.filter((s) => s.status === "success");
}

function failedSources(ctx: AssembledContext): readonly SourceResult[] {
  return ctx.sources.filter((s) => s.status !== "success");
}

function contextToString(ctx: AssembledContext): string {
  const sections: string[] = [];

  for (const source of availableSources(ctx)) {
    if (source.content) {
      sections.push(`## ${source.name}\n${source.content}`);
    }
  }

  if (ctx.degraded) {
    sections.push(
      `## Context Assembly Notice\n${ctx.degradationNotice}`
    );
  }

  return sections.join("\n\n");
}

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`Timed out after ${timeoutMs}ms`)),
      timeoutMs
    );
    promise.then(
      (value) => { clearTimeout(timer); resolve(value); },
      (error) => { clearTimeout(timer); reject(error); }
    );
  });
}

class ParallelContextAssembler {
  private readonly sources: readonly ContextSourceConfig[];
  private stats: readonly AssemblyStats[] = [];

  constructor(sources: ContextSourceConfig[]) {
    this.sources = Object.freeze(
      [...sources].sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0))
    );
  }

  async assemble(
    params: Record<string, unknown> = {}
  ): Promise<AssembledContext> {
    const start = performance.now();

    const results = await Promise.all(
      this.sources.map((source) => this.fetchSource(source, params))
    );

    const totalLatencyMs = performance.now() - start;
    const failed = results.filter((r) => r.status !== "success");
    const requiredFailures = failed.filter((r) =>
      this.sources.some((s) => s.name === r.name && s.required)
    );

    if (requiredFailures.length > 0) {
      throw new ContextAssemblyError(
        `Required source(s) failed: ${requiredFailures.map((r) => r.name).join(", ")}`
      );
    }

    const degraded = failed.length > 0;
    const degradationNotice = degraded
      ? "The following context sources were unavailable:\n" +
        failed
          .map(
            (r) =>
              `- ${r.name}: ${r.status}${r.errorMessage ? ` (${r.errorMessage})` : ""}`
          )
          .join("\n")
      : "";

    const assembled: AssembledContext = {
      sources: Object.freeze(results),
      totalLatencyMs,
      degraded,
      degradationNotice,
    };

    const available = availableSources(assembled);
    this.stats = Object.freeze([
      ...this.stats,
      {
        latencyMs: totalLatencyMs,
        sourcesTotal: results.length,
        sourcesAvailable: available.length,
        degraded,
      },
    ]);

    return assembled;
  }

  private async fetchSource(
    source: ContextSourceConfig,
    params: Record<string, unknown>
  ): Promise<SourceResult> {
    const start = performance.now();
    const timeoutMs = source.timeoutMs ?? 2000;

    try {
      const content = await withTimeout(
        source.fetchFn(params),
        timeoutMs
      );
      const latencyMs = performance.now() - start;

      return {
        name: source.name,
        status: "success",
        content,
        latencyMs,
        errorMessage: "",
        tokenEstimate: Math.floor(content.length / 4),
      };
    } catch (error) {
      const latencyMs = performance.now() - start;
      const message = error instanceof Error ? error.message : String(error);
      const isTimeout = message.includes("Timed out");

      return {
        name: source.name,
        status: isTimeout ? "timeout" : "error",
        content: "",
        latencyMs,
        errorMessage: message,
        tokenEstimate: 0,
      };
    }
  }

  get averageLatencyMs(): number {
    if (this.stats.length === 0) return 0;
    const total = this.stats.reduce((sum, s) => sum + s.latencyMs, 0);
    return total / this.stats.length;
  }

  get degradationRate(): number {
    if (this.stats.length === 0) return 0;
    const degraded = this.stats.filter((s) => s.degraded).length;
    return (degraded / this.stats.length) * 100;
  }
}

export {
  ParallelContextAssembler,
  ContextAssemblyError,
  AssembledContext,
  SourceResult,
  ContextSourceConfig,
  availableSources,
  failedSources,
  contextToString,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Latency reduced from sum-of-all to max-of-any (often 3-5x improvement) | Increased concurrency load on downstream services |
| Graceful degradation prevents one slow source from blocking everything | Partial context may produce lower-quality LLM responses |
| Per-source timeouts provide fine-grained control | Debugging concurrent failures is harder than sequential ones |
| Source-level metrics enable targeted optimization | Requires async/concurrent programming patterns |
| Required vs. optional sources let you balance reliability and speed | Resource consumption spikes at assembly time (all fetches at once) |
| Degradation notices give the LLM explicit awareness of missing context | Merging results from different sources requires careful ordering |

## When to Use

- Agent loops where context assembly happens on every iteration and latency compounds
- Applications that pull context from 3+ independent sources
- Real-time applications where time-to-first-token matters (chat, coding assistants)
- Systems with heterogeneous source latencies (fast local DB + slow external API)
- Any pipeline where sequential fetch latency exceeds your latency budget

## When NOT to Use

- When you have only 1-2 context sources and sequential assembly is fast enough
- When sources have dependencies on each other (source B needs source A's output)
- When downstream services cannot handle concurrent load spikes
- Batch processing where latency per-request is not a concern
- When partial context is unacceptable and all sources are strictly required

## Related Patterns

- **KV-Cache Optimization** (Optimization): After parallel assembly, structure the assembled context with stable components first to maximize KV-cache hits on subsequent calls.
- **Prompt Caching Strategies** (Optimization): Cache the assembled context or its components to avoid repeated parallel fetches for similar queries.
- **Incremental Context Updates** (Optimization): On subsequent turns, only re-fetch sources whose data has changed rather than re-assembling everything from scratch.
- **Context Coverage Analysis** (Evaluation): After assembly with degradation, use coverage analysis to determine whether the missing sources leave critical information gaps.

## Real-World Examples

- **Claude Code**: Assembles context from multiple sources in parallel -- file system reads, git state, terminal output, and memory files -- before each LLM call. This keeps the agent loop responsive even when individual sources are slow.
- **Perplexity**: Runs web search, knowledge base lookup, and conversation history retrieval concurrently to minimize time-to-first-token on search queries.
- **Manus**: Fetches tool definitions, agent memory, and task state in parallel during agent loop initialization. Their architecture documentation emphasizes concurrent context assembly as a latency optimization.
- **Production RAG Systems**: Standard pattern to run embedding generation and metadata filtering in parallel, then merge results before the LLM call. Systems with re-ranking add it as a dependent step after the parallel phase.
