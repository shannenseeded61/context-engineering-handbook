# Lossy Context Distillation

> Intentionally discard low-value context by using an LLM to extract only the facts, decisions, and state that matter for the current task -- everything else is dropped.

## Problem

Summarization preserves the *structure* of a conversation: it produces a shorter version of what was said. But structure is not always what you need. When the context window is under pressure and you need maximum compression, you need to extract only the *signal* -- the specific facts and decisions that are relevant to the task at hand -- and discard everything else, including structural elements like turn-taking, topic transitions, and tangential discussions that a summarizer would faithfully preserve in compressed form.

Without lossy context distillation:
- Summaries retain structurally important but task-irrelevant content (e.g., a summary of a debugging tangent that was ultimately abandoned).
- Compression ratios are limited because the summarizer tries to be faithful to the original structure.
- Context from different sources (conversation, RAG, tool outputs) carries forward information that was relevant to a previous sub-task but not the current one.
- The model wastes attention on context that is technically accurate but not actionable for the current step.

The key distinction from summarization: **summarization asks "what happened?" while distillation asks "what matters right now?"** Distillation is task-aware and lossy by design -- it intentionally discards information that a summarizer would preserve.

## Solution

Take the raw context (conversation history, RAG retrievals, tool outputs, or any combination) along with a description of the current task. Pass both to an LLM with a distillation prompt that instructs it to extract only the facts, decisions, constraints, and state that are relevant to completing the current task. The output is a minimal, flat list of task-relevant facts -- no narrative structure, no turn markers, no topic headers.

The distillation is explicitly lossy: information that is true but not relevant to the current task is dropped. This is the correct trade-off when context pressure is high and task focus is clear.

## How It Works

```
Input: Raw Context (2,500 tokens)          Current Task
+------------------------------------+    +-------------------+
| Turn 1: User asked about auth bug  |    | "Add rate limiting|
| Turn 2: Debugged JWT expiry issue  |    |  to the /api/v2   |
| Turn 3: Fixed with refresh tokens  |    |  endpoints using  |
| Turn 4: Discussed Redis vs Memcache|    |  Redis"           |
| Turn 5: Chose Redis for caching    |    +-------------------+
| Turn 6: Tangent about CI pipeline  |
| Turn 7: Back to API, added logging |
| Turn 8: User wants rate limiting   |
| Turn 9: Discussed sliding window   |
| Turn 10: Agreed on 100 req/min     |
+------------------------------------+

                    |
                    v  [LLM Distillation Pass]
                    |  "Extract only facts relevant
                    |   to the current task"
                    v

Output: Distilled Context (180 tokens)
+------------------------------------+
| TASK-RELEVANT FACTS:               |
| - Redis is the chosen cache/store  |
| - Rate limiting target: /api/v2/*  |
| - Limit: 100 requests/min per key  |
| - Algorithm: sliding window        |
| - Redis connection already config'd|
|   in config/redis.py               |
| - Existing middleware in            |
|   middleware/auth.py (add alongside)|
+------------------------------------+

Dropped (not relevant to current task):
  - JWT auth debugging details (turns 1-3)
  - CI pipeline tangent (turn 6)
  - Logging discussion (turn 7)
  - Redis vs Memcached comparison rationale

Compression: 2,500 -> 180 tokens (93% reduction)
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    token_count: int


@dataclass(frozen=True)
class DistillationResult:
    distilled_context: str
    distilled_token_count: int
    original_token_count: int
    task_description: str
    fact_count: int

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return 1.0 - (self.distilled_token_count / self.original_token_count)

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.distilled_token_count


DISTILLATION_PROMPT = """You are a context distillation engine. Your job is to extract ONLY the facts, decisions, and state that are directly relevant to the current task. Drop everything else.

CURRENT TASK:
{task}

RAW CONTEXT:
{context}

INSTRUCTIONS:
1. Read the current task carefully.
2. Scan the raw context for information that would help complete THIS SPECIFIC task.
3. Extract relevant facts as a flat bullet list. Each bullet should be a self-contained fact.
4. Include: decisions made, constraints specified, file paths, variable names, configuration values, architectural choices -- but ONLY if they relate to the current task.
5. Exclude: reasoning chains, tangential discussions, historical context that does not affect the current task, social pleasantries, exploratory ideas that were abandoned.
6. If a piece of context is ambiguously relevant, include it -- err on the side of keeping potentially useful facts.
7. Do NOT add information that is not in the raw context. Do NOT infer or speculate.

OUTPUT FORMAT:
Return ONLY a bullet list of task-relevant facts. No headers, no prose, no explanation.
If nothing in the context is relevant to the task, return: "No relevant context found."

TASK-RELEVANT FACTS:"""


@dataclass(frozen=True)
class ContextDistiller:
    """Extracts task-relevant facts from raw context, discarding everything else.

    Immutable: all operations return new data.
    """
    distillation_model: str = "claude-sonnet-4-20250514"
    max_distilled_tokens: int = 500
    min_context_tokens: int = 200  # Don't bother distilling if context is already small

    def should_distill(self, total_tokens: int) -> bool:
        """Check if distillation is worthwhile given the context size."""
        return total_tokens > self.min_context_tokens

    def _format_context(self, messages: list[Message]) -> str:
        """Format messages into a flat text block for the distillation prompt."""
        return "\n".join(
            f"[{m.role.value}]: {m.content}" for m in messages
        )

    def _format_raw_context(self, raw_context: str | list[Message]) -> str:
        """Accept either raw text or a message list."""
        if isinstance(raw_context, str):
            return raw_context
        return self._format_context(raw_context)

    def build_distillation_prompt(
        self, raw_context: str | list[Message], task: str
    ) -> str:
        """Build the distillation prompt for the LLM."""
        context_text = self._format_raw_context(raw_context)
        return DISTILLATION_PROMPT.format(task=task, context=context_text)

    def _count_facts(self, distilled: str) -> int:
        """Count the number of extracted facts (bullet points)."""
        lines = [line.strip() for line in distilled.strip().split("\n")]
        return sum(1 for line in lines if line.startswith("- ") or line.startswith("* "))

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate. Use tiktoken in production."""
        return len(text.split()) * 2

    async def distill(
        self,
        raw_context: str | list[Message],
        task: str,
        llm_client,  # Any client with async `complete(prompt) -> str`
    ) -> DistillationResult:
        """Distill raw context into task-relevant facts.

        Returns a DistillationResult. Does not mutate inputs.
        """
        context_text = self._format_raw_context(raw_context)
        original_tokens = self._estimate_tokens(context_text)

        if not self.should_distill(original_tokens):
            return DistillationResult(
                distilled_context=context_text,
                distilled_token_count=original_tokens,
                original_token_count=original_tokens,
                task_description=task,
                fact_count=0,
            )

        prompt = self.build_distillation_prompt(context_text, task)
        distilled = await llm_client.complete(prompt)
        distilled = distilled.strip()

        distilled_tokens = self._estimate_tokens(distilled)
        fact_count = self._count_facts(distilled)

        return DistillationResult(
            distilled_context=distilled,
            distilled_token_count=distilled_tokens,
            original_token_count=original_tokens,
            task_description=task,
            fact_count=fact_count,
        )

    async def distill_multiple(
        self,
        sources: list[tuple[str, str | list[Message]]],  # [(label, context), ...]
        task: str,
        llm_client,
    ) -> DistillationResult:
        """Distill multiple context sources into a single set of task-relevant facts.

        Useful when you have conversation history + RAG results + tool outputs
        and want a single distilled block.
        """
        combined_parts: list[str] = []
        total_original_tokens = 0

        for label, source in sources:
            text = self._format_raw_context(source)
            total_original_tokens += self._estimate_tokens(text)
            combined_parts.append(f"--- {label} ---\n{text}")

        combined = "\n\n".join(combined_parts)

        if not self.should_distill(total_original_tokens):
            return DistillationResult(
                distilled_context=combined,
                distilled_token_count=total_original_tokens,
                original_token_count=total_original_tokens,
                task_description=task,
                fact_count=0,
            )

        prompt = self.build_distillation_prompt(combined, task)
        distilled = await llm_client.complete(prompt)
        distilled = distilled.strip()

        return DistillationResult(
            distilled_context=distilled,
            distilled_token_count=self._estimate_tokens(distilled),
            original_token_count=total_original_tokens,
            task_description=task,
            fact_count=self._count_facts(distilled),
        )


# --- Usage Example ---

async def example_usage():
    """Demonstrates context distillation in a multi-source agent."""
    from some_llm_client import LLMClient  # Replace with your client

    client = LLMClient(model="claude-sonnet-4-20250514")
    distiller = ContextDistiller(
        max_distilled_tokens=500,
        min_context_tokens=200,
    )

    # Single-source distillation
    conversation = [
        Message(role=Role.USER, content="Fix the auth bug in login.py", token_count=20),
        Message(role=Role.ASSISTANT, content="I found a JWT expiry issue...", token_count=300),
        Message(role=Role.USER, content="Good. Now add rate limiting to /api/v2", token_count=25),
    ]

    result = await distiller.distill(
        raw_context=conversation,
        task="Add rate limiting to /api/v2 endpoints using Redis",
        llm_client=client,
    )
    print(f"Distilled {result.original_token_count} -> {result.distilled_token_count} tokens")
    print(f"Compression: {result.compression_ratio:.0%}")
    print(f"Facts extracted: {result.fact_count}")
    print(result.distilled_context)

    # Multi-source distillation
    rag_results = "Redis INCR command docs: INCR key increments the value..."
    tool_output = "$ cat config/redis.py\nREDIS_URL=redis://localhost:6379/0"

    multi_result = await distiller.distill_multiple(
        sources=[
            ("Conversation History", conversation),
            ("Retrieved Documentation", rag_results),
            ("Tool Output", tool_output),
        ],
        task="Add rate limiting to /api/v2 endpoints using Redis",
        llm_client=client,
    )
    print(f"\nMulti-source: {multi_result.original_token_count} -> {multi_result.distilled_token_count} tokens")
```

### TypeScript

```typescript
type Role = "user" | "assistant" | "system";

interface Message {
  readonly role: Role;
  readonly content: string;
  readonly tokenCount: number;
}

interface DistillationResult {
  readonly distilledContext: string;
  readonly distilledTokenCount: number;
  readonly originalTokenCount: number;
  readonly taskDescription: string;
  readonly factCount: number;
  readonly compressionRatio: number;
  readonly tokensSaved: number;
}

interface LLMClient {
  complete(prompt: string): Promise<string>;
}

interface DistillerConfig {
  readonly maxDistilledTokens: number;
  readonly minContextTokens: number;
}

const DEFAULT_DISTILLER_CONFIG: DistillerConfig = {
  maxDistilledTokens: 500,
  minContextTokens: 200,
};

const DISTILLATION_PROMPT = `You are a context distillation engine. Your job is to extract ONLY the facts, decisions, and state that are directly relevant to the current task. Drop everything else.

CURRENT TASK:
{task}

RAW CONTEXT:
{context}

INSTRUCTIONS:
1. Read the current task carefully.
2. Scan the raw context for information that would help complete THIS SPECIFIC task.
3. Extract relevant facts as a flat bullet list. Each bullet should be a self-contained fact.
4. Include: decisions made, constraints specified, file paths, variable names, configuration values, architectural choices -- but ONLY if they relate to the current task.
5. Exclude: reasoning chains, tangential discussions, historical context that does not affect the current task, social pleasantries, exploratory ideas that were abandoned.
6. If a piece of context is ambiguously relevant, include it -- err on the side of keeping potentially useful facts.
7. Do NOT add information that is not in the raw context. Do NOT infer or speculate.

OUTPUT FORMAT:
Return ONLY a bullet list of task-relevant facts. No headers, no prose, no explanation.
If nothing in the context is relevant to the task, return: "No relevant context found."

TASK-RELEVANT FACTS:`;

function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.5);
}

function formatMessages(messages: readonly Message[]): string {
  return messages.map((m) => `[${m.role}]: ${m.content}`).join("\n");
}

function formatRawContext(context: string | readonly Message[]): string {
  if (typeof context === "string") return context;
  return formatMessages(context);
}

function countFacts(distilled: string): number {
  return distilled
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- ") || line.startsWith("* ")).length;
}

function buildDistillationPrompt(
  context: string | readonly Message[],
  task: string
): string {
  const contextText = formatRawContext(context);
  return DISTILLATION_PROMPT.replace("{task}", task).replace(
    "{context}",
    contextText
  );
}

function createResult(
  distilled: string,
  originalTokens: number,
  task: string
): DistillationResult {
  const distilledTokenCount = estimateTokens(distilled);
  return {
    distilledContext: distilled,
    distilledTokenCount,
    originalTokenCount: originalTokens,
    taskDescription: task,
    factCount: countFacts(distilled),
    compressionRatio:
      originalTokens > 0 ? 1.0 - distilledTokenCount / originalTokens : 0,
    tokensSaved: originalTokens - distilledTokenCount,
  };
}

async function distill(
  rawContext: string | readonly Message[],
  task: string,
  client: LLMClient,
  config: DistillerConfig = DEFAULT_DISTILLER_CONFIG
): Promise<DistillationResult> {
  const contextText = formatRawContext(rawContext);
  const originalTokens = estimateTokens(contextText);

  if (originalTokens <= config.minContextTokens) {
    return createResult(contextText, originalTokens, task);
  }

  const prompt = buildDistillationPrompt(contextText, task);
  const distilled = (await client.complete(prompt)).trim();

  return createResult(distilled, originalTokens, task);
}

interface LabeledSource {
  readonly label: string;
  readonly context: string | readonly Message[];
}

async function distillMultiple(
  sources: readonly LabeledSource[],
  task: string,
  client: LLMClient,
  config: DistillerConfig = DEFAULT_DISTILLER_CONFIG
): Promise<DistillationResult> {
  let totalOriginalTokens = 0;
  const parts: string[] = [];

  for (const { label, context } of sources) {
    const text = formatRawContext(context);
    totalOriginalTokens += estimateTokens(text);
    parts.push(`--- ${label} ---\n${text}`);
  }

  const combined = parts.join("\n\n");

  if (totalOriginalTokens <= config.minContextTokens) {
    return createResult(combined, totalOriginalTokens, task);
  }

  const prompt = buildDistillationPrompt(combined, task);
  const distilled = (await client.complete(prompt)).trim();

  return createResult(distilled, totalOriginalTokens, task);
}

// --- Usage Example ---

async function example(client: LLMClient): Promise<void> {
  const conversation: Message[] = [
    { role: "user", content: "Fix the auth bug in login.py", tokenCount: 20 },
    {
      role: "assistant",
      content: "I found a JWT expiry issue in the refresh token handler...",
      tokenCount: 300,
    },
    {
      role: "user",
      content: "Good. Now add rate limiting to /api/v2",
      tokenCount: 25,
    },
  ];

  // Single-source distillation
  const result = await distill(
    conversation,
    "Add rate limiting to /api/v2 endpoints using Redis",
    client
  );

  console.log(
    `Distilled: ${result.originalTokenCount} -> ${result.distilledTokenCount} tokens`
  );
  console.log(`Compression: ${Math.round(result.compressionRatio * 100)}%`);
  console.log(`Facts: ${result.factCount}`);
  console.log(result.distilledContext);

  // Multi-source distillation
  const multiResult = await distillMultiple(
    [
      { label: "Conversation History", context: conversation },
      {
        label: "Retrieved Documentation",
        context: "Redis INCR command docs: INCR key increments...",
      },
      {
        label: "Tool Output",
        context: "$ cat config/redis.py\nREDIS_URL=redis://localhost:6379/0",
      },
    ],
    "Add rate limiting to /api/v2 endpoints using Redis",
    client
  );

  console.log(
    `\nMulti-source: ${multiResult.originalTokenCount} -> ${multiResult.distilledTokenCount} tokens`
  );
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Highest compression ratios of any LLM-based compression pattern (often 90%+) | Aggressively lossy -- discarded information cannot be recovered |
| Task-aware: keeps only what matters for the current step | Requires a clear task description; vague tasks lead to poor distillation |
| Works across heterogeneous context sources (conversation, RAG, tools) | LLM call cost and latency per distillation pass |
| Output is a flat fact list, easy for the model to scan quickly | The distilling LLM may misjudge relevance, dropping critical facts |
| Eliminates structural overhead that summarization preserves | Not suitable when task context changes rapidly (the distillation becomes stale) |

## When to Use

- When context pressure is severe and you need maximum compression beyond what summarization provides.
- When the current task is clearly defined and you can articulate what information is relevant.
- When combining multiple context sources (conversation + RAG + tools) into a single, lean context block.
- After summarization has already been applied but the result is still too large for the available budget.
- In multi-step agent workflows where each step has a distinct sub-task and only needs context relevant to that step.

## When NOT to Use

- When the task is vague or exploratory ("help me figure out what to do next") -- distillation needs a clear target.
- When you need to preserve the narrative flow of a conversation for the user's benefit (use summarization instead).
- When context is already small enough that the LLM call cost exceeds the token savings.
- When the same context needs to serve multiple different tasks -- distillation is task-specific, so you would need to re-distill for each task.
- When information recovery is important -- distillation is deliberately irreversible.

## Related Patterns

- **[Conversation Compaction](conversation-compaction.md)** -- Preserves structure while compressing. Use compaction when you need a faithful (if shorter) record; use distillation when you need only the task-relevant signal.
- **[Observation Masking](observation-masking.md)** -- Deterministic compression of tool outputs. Apply masking first (it is free), then distill the remaining context if still over budget.
- **[Hierarchical Summarization](hierarchical-summarization.md)** -- Provides tiered compression. Distillation can be applied to the brief tier for even more aggressive compression.
- **[Token Budget Allocation](token-budget-allocation.md)** -- Distillation can serve as the compression strategy for any budget slot that needs maximum compression.

## Real-World Examples

1. **Devin's task-scoped context** -- Cognition's Devin agent maintains a focused context relevant to the current coding sub-task, effectively distilling the broader session history into task-relevant facts before each action.

2. **RAG post-processing pipelines** -- Production RAG systems at companies like Perplexity and You.com do not pass raw retrieved chunks to the model. They extract relevant passages and facts, discarding irrelevant sections of retrieved documents -- a form of context distillation.

3. **GitHub Copilot Workspace** -- When generating a plan from an issue, Copilot Workspace distills the repository context into only the files and facts relevant to the specific issue, rather than including the entire codebase context.

4. **Medical AI assistants** -- Clinical AI tools distill patient records into task-relevant facts (e.g., for a drug interaction check, only current medications and allergies are extracted from a potentially massive medical history).
