# Token Budget Allocation

> Explicitly budget the context window across competing demands -- system prompt, conversation history, retrieved context, tool schemas, and working memory -- enforcing limits by triggering compression when any component exceeds its allocation.

## Problem

The context window is a shared resource consumed by multiple components simultaneously: system prompts, conversation history, RAG retrievals, tool definitions, scratchpad/working memory, and the model's own output. Without explicit budgeting, components compete in an unmanaged free-for-all where the loudest consumer (usually conversation history or RAG retrievals) crowds out everything else.

Without token budget allocation:
- A single large RAG retrieval can consume half the context window, leaving insufficient room for conversation history.
- Tool schemas (which grow linearly with the number of available tools) silently eat into the space available for actual content.
- System prompts grow over time as developers add instructions, unaware of their impact on the overall budget.
- There is no early warning when context pressure builds -- the system hits the hard limit abruptly and fails.
- Compression strategies (compaction, masking, distillation) fire reactively at the last moment instead of proactively managing headroom.

The core insight is that **context window management is a resource allocation problem**, and resource allocation problems are solved with budgets, not hope.

## Solution

Define a budget that partitions the context window into named slots, each with a maximum token allocation. A budget manager tracks the current usage of each slot and enforces limits. When a component exceeds its budget, the manager triggers a slot-specific compression strategy (truncation, summarization, eviction) to bring it back within bounds.

The budget is defined at system initialization and can be adjusted at runtime based on the current task. For example, a code-generation task might allocate more to tool schemas, while a research task might allocate more to retrieved context.

## How It Works

```
Context Window: 128,000 tokens
+-------------------------------------------------------------------+
|                        TOKEN BUDGET                                |
+-------------------------------------------------------------------+
| Component          | Budget  | Current | Status                   |
|--------------------|---------|---------|--------------------------|
| System Prompt      |  5,000  |  3,200  | OK                       |
| Tool Schemas       | 10,000  |  8,500  | OK                       |
| Conversation Hist. | 40,000  | 41,200  | OVER -> trigger compaction|
| RAG Context        | 30,000  | 22,000  | OK                       |
| Working Memory     | 15,000  |  9,800  | OK                       |
| Output Reserve     | 20,000  |    --   | Reserved for generation  |
| Safety Margin      |  8,000  |    --   | Buffer for estimation err|
+-------------------------------------------------------------------+
                                  Total:   84,700 / 128,000

When "Conversation History" exceeds 40,000:
  1. Budget manager detects overage
  2. Triggers conversation compaction strategy
  3. Compaction reduces history to ~25,000 tokens
  4. Budget is back within bounds

When "RAG Context" would exceed 30,000:
  1. Budget manager intercepts before insertion
  2. Truncates or re-ranks retrieved chunks
  3. Only highest-relevance chunks are included
  4. Stays within budget
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable


class ComponentName(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_SCHEMAS = "tool_schemas"
    CONVERSATION_HISTORY = "conversation_history"
    RAG_CONTEXT = "rag_context"
    WORKING_MEMORY = "working_memory"
    OUTPUT_RESERVE = "output_reserve"
    SAFETY_MARGIN = "safety_margin"


@dataclass(frozen=True)
class ComponentBudget:
    name: ComponentName
    max_tokens: int
    current_tokens: int = 0
    compressible: bool = True  # Can this component be compressed?
    priority: int = 0          # Higher = harder to compress (last resort)

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def is_over_budget(self) -> bool:
        return self.current_tokens > self.max_tokens

    @property
    def utilization(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return self.current_tokens / self.max_tokens


@dataclass(frozen=True)
class BudgetOverage:
    component: ComponentName
    budget: int
    actual: int
    overage: int


@dataclass(frozen=True)
class BudgetSnapshot:
    components: tuple[ComponentBudget, ...]
    total_capacity: int

    @property
    def total_used(self) -> int:
        return sum(c.current_tokens for c in self.components)

    @property
    def total_remaining(self) -> int:
        return self.total_capacity - self.total_used

    @property
    def overages(self) -> list[BudgetOverage]:
        return [
            BudgetOverage(
                component=c.name,
                budget=c.max_tokens,
                actual=c.current_tokens,
                overage=c.current_tokens - c.max_tokens,
            )
            for c in self.components
            if c.is_over_budget
        ]


# Type alias for compression strategies
CompressionStrategy = Callable[[str, int], Awaitable[str]]
# Takes (content, target_tokens) -> compressed_content


@dataclass(frozen=True)
class TokenBudgetAllocator:
    """Manages token budgets across context window components.

    Immutable: all state changes return new instances.
    """
    total_capacity: int
    budgets: dict[ComponentName, ComponentBudget] = field(default_factory=dict)
    strategies: dict[ComponentName, CompressionStrategy] = field(default_factory=dict)

    def _validate_budgets(self) -> None:
        """Verify that budgets do not exceed total capacity."""
        total_budgeted = sum(b.max_tokens for b in self.budgets.values())
        if total_budgeted > self.total_capacity:
            raise ValueError(
                f"Total budgeted tokens ({total_budgeted}) exceed "
                f"capacity ({self.total_capacity})"
            )

    @staticmethod
    def create(
        total_capacity: int,
        allocations: dict[ComponentName, tuple[int, bool, int]],
        strategies: dict[ComponentName, CompressionStrategy] | None = None,
    ) -> "TokenBudgetAllocator":
        """Factory method to create an allocator.

        allocations: {name: (max_tokens, compressible, priority)}
        """
        budgets = {
            name: ComponentBudget(
                name=name,
                max_tokens=max_tokens,
                compressible=compressible,
                priority=priority,
            )
            for name, (max_tokens, compressible, priority) in allocations.items()
        }

        allocator = TokenBudgetAllocator(
            total_capacity=total_capacity,
            budgets=budgets,
            strategies=strategies or {},
        )
        allocator._validate_budgets()
        return allocator

    def update_usage(
        self, component: ComponentName, token_count: int
    ) -> "TokenBudgetAllocator":
        """Record the current token usage for a component. Returns a new allocator."""
        if component not in self.budgets:
            raise KeyError(f"Unknown component: {component}")

        updated_budget = ComponentBudget(
            name=self.budgets[component].name,
            max_tokens=self.budgets[component].max_tokens,
            current_tokens=token_count,
            compressible=self.budgets[component].compressible,
            priority=self.budgets[component].priority,
        )

        new_budgets = {**self.budgets, component: updated_budget}
        return TokenBudgetAllocator(
            total_capacity=self.total_capacity,
            budgets=new_budgets,
            strategies=self.strategies,
        )

    def snapshot(self) -> BudgetSnapshot:
        """Get a snapshot of current budget state."""
        return BudgetSnapshot(
            components=tuple(self.budgets.values()),
            total_capacity=self.total_capacity,
        )

    def check_overage(self, component: ComponentName) -> BudgetOverage | None:
        """Check if a component is over budget."""
        budget = self.budgets.get(component)
        if budget and budget.is_over_budget:
            return BudgetOverage(
                component=component,
                budget=budget.max_tokens,
                actual=budget.current_tokens,
                overage=budget.current_tokens - budget.max_tokens,
            )
        return None

    def get_all_overages(self) -> list[BudgetOverage]:
        """Get all components that are over budget, sorted by priority (lowest first)."""
        overages = [
            self.check_overage(name)
            for name in self.budgets
        ]
        valid = [o for o in overages if o is not None]
        return sorted(valid, key=lambda o: self.budgets[o.component].priority)

    async def enforce_budget(
        self, component: ComponentName, content: str, token_count: int
    ) -> tuple["TokenBudgetAllocator", str]:
        """Enforce the budget for a component, compressing if necessary.

        Returns (updated_allocator, possibly_compressed_content).
        """
        budget = self.budgets[component]

        if token_count <= budget.max_tokens:
            return self.update_usage(component, token_count), content

        if not budget.compressible:
            raise ValueError(
                f"Component '{component.value}' exceeds budget "
                f"({token_count} > {budget.max_tokens}) and is not compressible"
            )

        strategy = self.strategies.get(component)
        if strategy is None:
            raise ValueError(
                f"Component '{component.value}' exceeds budget but no "
                f"compression strategy is registered"
            )

        target_tokens = int(budget.max_tokens * 0.8)  # Compress to 80% of budget
        compressed = await strategy(content, target_tokens)
        compressed_tokens = len(compressed.split()) * 2  # Rough estimate

        return self.update_usage(component, compressed_tokens), compressed

    def remaining_for(self, component: ComponentName) -> int:
        """How many tokens remain in a component's budget."""
        budget = self.budgets.get(component)
        return budget.remaining if budget else 0

    def reallocate(
        self, source: ComponentName, target: ComponentName, tokens: int
    ) -> "TokenBudgetAllocator":
        """Move budget allocation from one component to another.

        Returns a new allocator. Useful for dynamic rebalancing.
        """
        source_budget = self.budgets[source]
        target_budget = self.budgets[target]

        if tokens > source_budget.remaining:
            raise ValueError(
                f"Cannot reallocate {tokens} tokens from {source.value}: "
                f"only {source_budget.remaining} available"
            )

        new_source = ComponentBudget(
            name=source_budget.name,
            max_tokens=source_budget.max_tokens - tokens,
            current_tokens=source_budget.current_tokens,
            compressible=source_budget.compressible,
            priority=source_budget.priority,
        )
        new_target = ComponentBudget(
            name=target_budget.name,
            max_tokens=target_budget.max_tokens + tokens,
            current_tokens=target_budget.current_tokens,
            compressible=target_budget.compressible,
            priority=target_budget.priority,
        )

        new_budgets = {**self.budgets, source: new_source, target: new_target}
        return TokenBudgetAllocator(
            total_capacity=self.total_capacity,
            budgets=new_budgets,
            strategies=self.strategies,
        )


# --- Usage Example ---

async def truncate_rag(content: str, target_tokens: int) -> str:
    """Simple truncation strategy for RAG context."""
    chunks = content.split("\n---\n")
    result_chunks: list[str] = []
    current_tokens = 0
    for chunk in chunks:
        chunk_tokens = len(chunk.split()) * 2
        if current_tokens + chunk_tokens > target_tokens:
            break
        result_chunks.append(chunk)
        current_tokens += chunk_tokens
    return "\n---\n".join(result_chunks)


async def example_usage():
    """Demonstrates token budget allocation in a RAG agent."""

    async def summarize_history(content: str, target_tokens: int) -> str:
        # In production, call an LLM here
        return f"[Summarized to ~{target_tokens} tokens]: {content[:200]}..."

    allocator = TokenBudgetAllocator.create(
        total_capacity=128_000,
        allocations={
            ComponentName.SYSTEM_PROMPT:        (5_000, False, 10),  # Not compressible
            ComponentName.TOOL_SCHEMAS:         (10_000, False, 9),
            ComponentName.CONVERSATION_HISTORY: (40_000, True, 3),
            ComponentName.RAG_CONTEXT:          (30_000, True, 2),
            ComponentName.WORKING_MEMORY:       (15_000, True, 5),
            ComponentName.OUTPUT_RESERVE:       (20_000, False, 10),
            ComponentName.SAFETY_MARGIN:        (8_000, False, 10),
        },
        strategies={
            ComponentName.CONVERSATION_HISTORY: summarize_history,
            ComponentName.RAG_CONTEXT: truncate_rag,
        },
    )

    # Simulate adding content
    allocator = allocator.update_usage(ComponentName.SYSTEM_PROMPT, 3_200)
    allocator = allocator.update_usage(ComponentName.TOOL_SCHEMAS, 8_500)

    # RAG retrieval comes in under budget
    rag_content = "chunk1\n---\nchunk2\n---\nchunk3"
    allocator, rag_content = await allocator.enforce_budget(
        ComponentName.RAG_CONTEXT, rag_content, 22_000
    )

    # Conversation history exceeds budget -- triggers compression
    history_content = "Very long conversation..."
    allocator, history_content = await allocator.enforce_budget(
        ComponentName.CONVERSATION_HISTORY, history_content, 45_000
    )

    # Check the budget state
    snapshot = allocator.snapshot()
    print(f"Total used: {snapshot.total_used} / {snapshot.total_capacity}")
    for component in snapshot.components:
        print(f"  {component.name.value}: {component.current_tokens}/{component.max_tokens} "
              f"({component.utilization:.0%})")
```

### TypeScript

```typescript
type ComponentName =
  | "system_prompt"
  | "tool_schemas"
  | "conversation_history"
  | "rag_context"
  | "working_memory"
  | "output_reserve"
  | "safety_margin";

interface ComponentBudget {
  readonly name: ComponentName;
  readonly maxTokens: number;
  readonly currentTokens: number;
  readonly compressible: boolean;
  readonly priority: number; // Higher = harder to compress
}

interface BudgetOverage {
  readonly component: ComponentName;
  readonly budget: number;
  readonly actual: number;
  readonly overage: number;
}

interface BudgetSnapshot {
  readonly components: readonly ComponentBudget[];
  readonly totalCapacity: number;
  readonly totalUsed: number;
  readonly totalRemaining: number;
  readonly overages: readonly BudgetOverage[];
}

type CompressionStrategy = (
  content: string,
  targetTokens: number
) => Promise<string>;

interface AllocatorConfig {
  readonly totalCapacity: number;
  readonly allocations: Record<
    ComponentName,
    { maxTokens: number; compressible: boolean; priority: number }
  >;
}

interface TokenBudgetAllocator {
  readonly totalCapacity: number;
  readonly budgets: Readonly<Record<string, ComponentBudget>>;
  readonly strategies: Readonly<Record<string, CompressionStrategy>>;
}

function createAllocator(
  config: AllocatorConfig,
  strategies: Partial<Record<ComponentName, CompressionStrategy>> = {}
): TokenBudgetAllocator {
  const budgets: Record<string, ComponentBudget> = {};

  for (const [name, alloc] of Object.entries(config.allocations)) {
    budgets[name] = {
      name: name as ComponentName,
      maxTokens: alloc.maxTokens,
      currentTokens: 0,
      compressible: alloc.compressible,
      priority: alloc.priority,
    };
  }

  const totalBudgeted = Object.values(budgets).reduce(
    (sum, b) => sum + b.maxTokens,
    0
  );

  if (totalBudgeted > config.totalCapacity) {
    throw new Error(
      `Total budgeted tokens (${totalBudgeted}) exceed capacity (${config.totalCapacity})`
    );
  }

  return {
    totalCapacity: config.totalCapacity,
    budgets,
    strategies: strategies as Record<string, CompressionStrategy>,
  };
}

function updateUsage(
  allocator: TokenBudgetAllocator,
  component: ComponentName,
  tokenCount: number
): TokenBudgetAllocator {
  const budget = allocator.budgets[component];
  if (!budget) {
    throw new Error(`Unknown component: ${component}`);
  }

  return {
    ...allocator,
    budgets: {
      ...allocator.budgets,
      [component]: { ...budget, currentTokens: tokenCount },
    },
  };
}

function getSnapshot(allocator: TokenBudgetAllocator): BudgetSnapshot {
  const components = Object.values(allocator.budgets);
  const totalUsed = components.reduce((s, c) => s + c.currentTokens, 0);

  const overages: BudgetOverage[] = components
    .filter((c) => c.currentTokens > c.maxTokens)
    .map((c) => ({
      component: c.name,
      budget: c.maxTokens,
      actual: c.currentTokens,
      overage: c.currentTokens - c.maxTokens,
    }))
    .sort(
      (a, b) =>
        (allocator.budgets[a.component]?.priority ?? 0) -
        (allocator.budgets[b.component]?.priority ?? 0)
    );

  return {
    components,
    totalCapacity: allocator.totalCapacity,
    totalUsed,
    totalRemaining: allocator.totalCapacity - totalUsed,
    overages,
  };
}

async function enforceBudget(
  allocator: TokenBudgetAllocator,
  component: ComponentName,
  content: string,
  tokenCount: number
): Promise<{ allocator: TokenBudgetAllocator; content: string }> {
  const budget = allocator.budgets[component];
  if (!budget) {
    throw new Error(`Unknown component: ${component}`);
  }

  if (tokenCount <= budget.maxTokens) {
    return {
      allocator: updateUsage(allocator, component, tokenCount),
      content,
    };
  }

  if (!budget.compressible) {
    throw new Error(
      `Component '${component}' exceeds budget (${tokenCount} > ${budget.maxTokens}) and is not compressible`
    );
  }

  const strategy = allocator.strategies[component];
  if (!strategy) {
    throw new Error(
      `Component '${component}' exceeds budget but no compression strategy is registered`
    );
  }

  const targetTokens = Math.floor(budget.maxTokens * 0.8);
  const compressed = await strategy(content, targetTokens);
  const compressedTokens = Math.ceil(compressed.split(/\s+/).length * 1.5);

  return {
    allocator: updateUsage(allocator, component, compressedTokens),
    content: compressed,
  };
}

function reallocate(
  allocator: TokenBudgetAllocator,
  source: ComponentName,
  target: ComponentName,
  tokens: number
): TokenBudgetAllocator {
  const sourceBudget = allocator.budgets[source];
  const targetBudget = allocator.budgets[target];

  if (!sourceBudget || !targetBudget) {
    throw new Error(`Unknown component: ${source} or ${target}`);
  }

  const sourceRemaining = sourceBudget.maxTokens - sourceBudget.currentTokens;
  if (tokens > sourceRemaining) {
    throw new Error(
      `Cannot reallocate ${tokens} from ${source}: only ${sourceRemaining} available`
    );
  }

  return {
    ...allocator,
    budgets: {
      ...allocator.budgets,
      [source]: { ...sourceBudget, maxTokens: sourceBudget.maxTokens - tokens },
      [target]: { ...targetBudget, maxTokens: targetBudget.maxTokens + tokens },
    },
  };
}

// --- Usage Example ---

async function example(): Promise<void> {
  const truncateRag: CompressionStrategy = async (content, targetTokens) => {
    const chunks = content.split("\n---\n");
    const result: string[] = [];
    let currentTokens = 0;
    for (const chunk of chunks) {
      const chunkTokens = Math.ceil(chunk.split(/\s+/).length * 1.5);
      if (currentTokens + chunkTokens > targetTokens) break;
      result.push(chunk);
      currentTokens += chunkTokens;
    }
    return result.join("\n---\n");
  };

  const summarizeHistory: CompressionStrategy = async (content, targetTokens) => {
    // In production, call an LLM here
    return `[Summarized to ~${targetTokens} tokens]: ${content.slice(0, 200)}...`;
  };

  let allocator = createAllocator(
    {
      totalCapacity: 128_000,
      allocations: {
        system_prompt:        { maxTokens: 5_000,  compressible: false, priority: 10 },
        tool_schemas:         { maxTokens: 10_000, compressible: false, priority: 9 },
        conversation_history: { maxTokens: 40_000, compressible: true,  priority: 3 },
        rag_context:          { maxTokens: 30_000, compressible: true,  priority: 2 },
        working_memory:       { maxTokens: 15_000, compressible: true,  priority: 5 },
        output_reserve:       { maxTokens: 20_000, compressible: false, priority: 10 },
        safety_margin:        { maxTokens: 8_000,  compressible: false, priority: 10 },
      },
    },
    {
      conversation_history: summarizeHistory,
      rag_context: truncateRag,
    }
  );

  // Update fixed components
  allocator = updateUsage(allocator, "system_prompt", 3_200);
  allocator = updateUsage(allocator, "tool_schemas", 8_500);

  // Enforce budget on dynamic components
  const ragResult = await enforceBudget(
    allocator,
    "rag_context",
    "chunk1\n---\nchunk2\n---\nchunk3",
    22_000
  );
  allocator = ragResult.allocator;

  const historyResult = await enforceBudget(
    allocator,
    "conversation_history",
    "Very long conversation...",
    45_000
  );
  allocator = historyResult.allocator;

  // Inspect the budget
  const snapshot = getSnapshot(allocator);
  console.log(`Total: ${snapshot.totalUsed} / ${snapshot.totalCapacity}`);
  for (const c of snapshot.components) {
    const pct = c.maxTokens > 0 ? Math.round((c.currentTokens / c.maxTokens) * 100) : 0;
    console.log(`  ${c.name}: ${c.currentTokens}/${c.maxTokens} (${pct}%)`);
  }
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Makes context window usage explicit and observable | Requires upfront design decisions about allocation ratios |
| Prevents any single component from monopolizing the context | Fixed budgets may not reflect the actual needs of a specific query |
| Triggers compression proactively, before hitting hard limits | Over-engineering risk: simple applications may not need per-component budgets |
| Enables dynamic reallocation based on task type | Token counting is approximate (especially across different tokenizers) |
| Composable with all other compression patterns | Adds a coordination layer between components that must be maintained |

## When to Use

- RAG systems where retrieval size varies dramatically per query.
- Multi-tool agents where tool schema count can change at runtime.
- Production systems that need predictable context window behavior across diverse workloads.
- Any system that combines multiple context sources (system prompt + history + retrieved docs + tools) and needs to manage their relative priorities.
- When you need observability into *why* context is being compressed and *which* component triggered it.

## When NOT to Use

- Simple chatbots with a single context source (just conversation history).
- When the context window is large enough that you never approach the limit.
- Prototyping and exploration where the overhead of budget management is not justified.
- When all context components are roughly the same size and grow at the same rate.

## Related Patterns

- **[Conversation Compaction](conversation-compaction.md)** -- A compression strategy that can be registered as the compressor for the conversation history budget slot.
- **[Observation Masking](observation-masking.md)** -- A compression strategy for tool output slots. Register it as the compressor for tool-related budget components.
- **[Hierarchical Summarization](hierarchical-summarization.md)** -- Can be used as the compression strategy for the conversation history slot, providing tiered compression within its allocated budget.
- **[Lossy Context Distillation](lossy-context-distillation.md)** -- A compression strategy for RAG context or working memory slots where only task-relevant facts should survive.

## Real-World Examples

1. **Anthropic's system prompt guidance** -- Anthropic recommends reserving specific portions of the context window for system prompts versus user content. This is implicit budget allocation -- making it explicit with enforced limits prevents drift.

2. **LangChain's `ConversationTokenBufferMemory`** -- Manages conversation history within a fixed token budget, automatically trimming older messages when the budget is exceeded. This is a single-slot version of the full budget allocation pattern.

3. **OpenAI's function calling token accounting** -- The OpenAI API counts function/tool definitions against the context window. Production applications must account for this when deciding how much conversation history to include -- an implicit budget allocation that benefits from being made explicit.

4. **Enterprise RAG pipelines** -- Production RAG systems at companies like Notion, Glean, and Perplexity must balance retrieved document tokens against conversation context. They implement token budgets (often called "context windows policies") to ensure retrieval does not crowd out the user's actual question.
