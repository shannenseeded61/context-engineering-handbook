# Hierarchical Summarization

> Summarize context at multiple levels of detail -- full resolution for recent turns, medium summary for mid-range, brief summary for old context -- preserving temporal ordering while maximizing token efficiency.

## Problem

Flat compression treats all context equally: either everything is at full resolution, or everything is compressed to the same degree. This creates a false choice between retaining detail (and running out of context) and aggressive compression (and losing nuance from recent interactions).

Without hierarchical summarization:
- Recent context that the model is actively working with gets compressed just as aggressively as ancient history.
- Old context that only needs a brief anchor is kept at full resolution, wasting tokens.
- A single compaction pass loses the temporal gradient -- the model cannot distinguish "just discussed" from "mentioned 50 turns ago."
- Repeated compaction passes (re-summarizing summaries) degrade quality unpredictably because each pass operates without awareness of the original detail level.

The fundamental insight is that **context value decays with age, but does not drop to zero.** A pyramid structure matches compression level to recency, giving the model high-fidelity access to what it needs most while retaining the gist of everything else.

## Solution

Divide the conversation history into tiers based on recency. Each tier has a different compression level:

1. **Full tier** -- The most recent N turns, kept verbatim.
2. **Medium tier** -- Turns older than the full tier but within a mid-range window, summarized into structured bullet points that preserve decisions, facts, and state.
3. **Brief tier** -- Everything older than the medium tier, distilled into a short paragraph capturing only the essential trajectory of the conversation.

As new turns arrive, content cascades down the tiers: full-tier content ages into the medium tier and gets summarized; medium-tier summaries age into the brief tier and get further compressed. The result is a sliding window where detail resolution decreases gracefully with age.

## How It Works

```
Conversation Timeline (newest on right):
Turn 1    Turn 5    Turn 10   Turn 15   Turn 20   Turn 25   Turn 30
|---------|---------|---------|---------|---------|---------|

                    BRIEF TIER           MEDIUM TIER        FULL TIER
                    (turns 1-15)         (turns 16-24)      (turns 25-30)
                    +---------------+    +---------------+  +---------------+
                    | ~50 tokens    |    | ~300 tokens   |  | ~2,000 tokens |
                    | "Session      |    | - Decided on  |  | [verbatim     |
                    |  started with |    |   Redis cache |  |  messages]    |
                    |  auth bug,    |    | - Fixed N+1   |  |               |
                    |  migrated to  |    |   query in    |  |               |
                    |  JWT, added   |    |   users.py    |  |               |
                    |  rate limits" |    | - User wants  |  |               |
                    +---------------+    |   <100ms p99  |  |               |
                                         +---------------+  +---------------+

Total: ~2,350 tokens instead of ~10,000 at full resolution

As Turn 31 arrives:
  - Turn 25 moves from FULL -> MEDIUM (gets summarized)
  - Oldest MEDIUM content moves to BRIEF (gets further compressed)
  - Turn 31 enters FULL tier at full resolution
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


class Tier(str, Enum):
    FULL = "full"
    MEDIUM = "medium"
    BRIEF = "brief"


@dataclass(frozen=True)
class Message:
    role: Role
    content: str
    token_count: int
    turn_index: int


@dataclass(frozen=True)
class TierSummary:
    tier: Tier
    content: str
    token_count: int
    turn_range: tuple[int, int]  # (start_index, end_index) inclusive
    source_turn_count: int


@dataclass(frozen=True)
class TierConfig:
    full_turns: int = 6          # Keep last 6 turns verbatim
    medium_turns: int = 12       # Summarize the 12 turns before that
    medium_max_tokens: int = 400  # Budget for medium-tier summary
    brief_max_tokens: int = 100   # Budget for brief-tier summary


@dataclass(frozen=True)
class HierarchicalContext:
    """The assembled context across all tiers."""
    brief_summary: TierSummary | None
    medium_summary: TierSummary | None
    full_messages: tuple[Message, ...]
    total_token_count: int


MEDIUM_SUMMARIZATION_PROMPT = """Summarize these conversation turns into structured bullet points.
Preserve: decisions made, facts discovered, files modified, current state, user preferences.
Discard: reasoning chains, exploratory tangents, verbose explanations.
Keep it under {max_tokens} tokens. Use bullet points, not prose.

TURNS:
{turns}"""

BRIEF_SUMMARIZATION_PROMPT = """Compress this conversation summary into a single short paragraph (2-3 sentences).
Capture only: the overall task, key decisions, and current trajectory.
Keep it under {max_tokens} tokens.

SUMMARY TO COMPRESS:
{summary}"""


@dataclass(frozen=True)
class HierarchicalSummarizer:
    """Manages a three-tier sliding window over conversation history.

    Immutable: every operation returns new data structures.
    """
    config: TierConfig = field(default_factory=TierConfig)

    def partition_messages(
        self, messages: list[Message]
    ) -> tuple[list[Message], list[Message], list[Message]]:
        """Split messages into (brief_candidates, medium_candidates, full_messages).

        System messages are excluded from partitioning and handled separately.
        """
        non_system = [m for m in messages if m.role != Role.SYSTEM]
        total = len(non_system)

        full_start = max(0, total - self.config.full_turns)
        medium_start = max(0, full_start - self.config.medium_turns)

        brief_candidates = non_system[:medium_start]
        medium_candidates = non_system[medium_start:full_start]
        full_messages = non_system[full_start:]

        return brief_candidates, medium_candidates, full_messages

    def _format_turns(self, messages: list[Message]) -> str:
        return "\n".join(
            f"[{m.role.value}] (turn {m.turn_index}): {m.content}"
            for m in messages
        )

    def build_medium_prompt(self, messages: list[Message]) -> str:
        return MEDIUM_SUMMARIZATION_PROMPT.format(
            max_tokens=self.config.medium_max_tokens,
            turns=self._format_turns(messages),
        )

    def build_brief_prompt(self, existing_summary: str) -> str:
        return BRIEF_SUMMARIZATION_PROMPT.format(
            max_tokens=self.config.brief_max_tokens,
            summary=existing_summary,
        )

    async def build_hierarchical_context(
        self,
        messages: list[Message],
        llm_client,  # Any client with async `complete(prompt) -> str`
        existing_brief: TierSummary | None = None,
        existing_medium: TierSummary | None = None,
    ) -> HierarchicalContext:
        """Build a hierarchical context from the full message history.

        Returns a new HierarchicalContext. Does not mutate inputs.
        """
        brief_candidates, medium_candidates, full_messages = self.partition_messages(messages)

        # --- Medium tier ---
        medium_summary: TierSummary | None = None
        if medium_candidates:
            prompt = self.build_medium_prompt(medium_candidates)
            summary_text = await llm_client.complete(prompt)
            medium_summary = TierSummary(
                tier=Tier.MEDIUM,
                content=summary_text,
                token_count=len(summary_text.split()) * 2,  # Rough estimate
                turn_range=(medium_candidates[0].turn_index, medium_candidates[-1].turn_index),
                source_turn_count=len(medium_candidates),
            )

        # --- Brief tier ---
        brief_summary: TierSummary | None = None
        if brief_candidates:
            # If we already have a brief summary covering these turns, extend it
            # with the medium summary that just aged out. Otherwise, summarize
            # the brief candidates directly.
            if existing_brief and existing_medium:
                combined = f"{existing_brief.content}\n\nAdditional context:\n{existing_medium.content}"
            else:
                combined = self._format_turns(brief_candidates)

            prompt = self.build_brief_prompt(combined)
            brief_text = await llm_client.complete(prompt)
            brief_summary = TierSummary(
                tier=Tier.BRIEF,
                content=brief_text,
                token_count=len(brief_text.split()) * 2,
                turn_range=(brief_candidates[0].turn_index, brief_candidates[-1].turn_index),
                source_turn_count=len(brief_candidates),
            )

        # --- Assemble ---
        total_tokens = sum(m.token_count for m in full_messages)
        if medium_summary:
            total_tokens += medium_summary.token_count
        if brief_summary:
            total_tokens += brief_summary.token_count

        return HierarchicalContext(
            brief_summary=brief_summary,
            medium_summary=medium_summary,
            full_messages=tuple(full_messages),
            total_token_count=total_tokens,
        )

    def render_context(self, ctx: HierarchicalContext) -> list[Message]:
        """Render the hierarchical context into a flat message list for the LLM."""
        rendered: list[Message] = []

        if ctx.brief_summary:
            rendered.append(Message(
                role=Role.SYSTEM,
                content=(
                    f"[BRIEF SUMMARY - turns {ctx.brief_summary.turn_range[0]}"
                    f"-{ctx.brief_summary.turn_range[1]}]\n{ctx.brief_summary.content}"
                ),
                token_count=ctx.brief_summary.token_count,
                turn_index=-2,
            ))

        if ctx.medium_summary:
            rendered.append(Message(
                role=Role.SYSTEM,
                content=(
                    f"[DETAILED SUMMARY - turns {ctx.medium_summary.turn_range[0]}"
                    f"-{ctx.medium_summary.turn_range[1]}]\n{ctx.medium_summary.content}"
                ),
                token_count=ctx.medium_summary.token_count,
                turn_index=-1,
            ))

        rendered.extend(ctx.full_messages)
        return rendered


# --- Usage Example ---

async def example_usage():
    """Demonstrates hierarchical summarization in an agent loop."""
    from some_llm_client import LLMClient  # Replace with your client

    client = LLMClient(model="claude-sonnet-4-20250514")
    summarizer = HierarchicalSummarizer(
        config=TierConfig(
            full_turns=6,
            medium_turns=12,
            medium_max_tokens=400,
            brief_max_tokens=100,
        )
    )

    messages: list[Message] = []
    turn_counter = 0
    brief_cache: TierSummary | None = None
    medium_cache: TierSummary | None = None

    while True:
        user_input = await get_user_input()
        turn_counter += 1
        messages.append(Message(
            role=Role.USER,
            content=user_input,
            token_count=len(user_input.split()) * 2,
            turn_index=turn_counter,
        ))

        # Rebuild hierarchical context when we have enough turns
        if len(messages) > summarizer.config.full_turns:
            ctx = await summarizer.build_hierarchical_context(
                messages, client,
                existing_brief=brief_cache,
                existing_medium=medium_cache,
            )
            brief_cache = ctx.brief_summary
            medium_cache = ctx.medium_summary
            rendered = summarizer.render_context(ctx)
        else:
            rendered = messages

        response = await client.chat(rendered)
        turn_counter += 1
        messages.append(Message(
            role=Role.ASSISTANT,
            content=response,
            token_count=len(response.split()) * 2,
            turn_index=turn_counter,
        ))
```

### TypeScript

```typescript
type Role = "user" | "assistant" | "system";
type Tier = "full" | "medium" | "brief";

interface Message {
  readonly role: Role;
  readonly content: string;
  readonly tokenCount: number;
  readonly turnIndex: number;
}

interface TierSummary {
  readonly tier: Tier;
  readonly content: string;
  readonly tokenCount: number;
  readonly turnRange: readonly [number, number];
  readonly sourceTurnCount: number;
}

interface TierConfig {
  readonly fullTurns: number;
  readonly mediumTurns: number;
  readonly mediumMaxTokens: number;
  readonly briefMaxTokens: number;
}

interface HierarchicalContext {
  readonly briefSummary: TierSummary | null;
  readonly mediumSummary: TierSummary | null;
  readonly fullMessages: readonly Message[];
  readonly totalTokenCount: number;
}

interface LLMClient {
  complete(prompt: string): Promise<string>;
}

const DEFAULT_TIER_CONFIG: TierConfig = {
  fullTurns: 6,
  mediumTurns: 12,
  mediumMaxTokens: 400,
  briefMaxTokens: 100,
};

const MEDIUM_PROMPT_TEMPLATE = `Summarize these conversation turns into structured bullet points.
Preserve: decisions made, facts discovered, files modified, current state, user preferences.
Discard: reasoning chains, exploratory tangents, verbose explanations.
Keep it under {maxTokens} tokens. Use bullet points, not prose.

TURNS:
{turns}`;

const BRIEF_PROMPT_TEMPLATE = `Compress this conversation summary into a single short paragraph (2-3 sentences).
Capture only: the overall task, key decisions, and current trajectory.
Keep it under {maxTokens} tokens.

SUMMARY TO COMPRESS:
{summary}`;

function partitionMessages(
  messages: readonly Message[],
  config: TierConfig
): {
  briefCandidates: readonly Message[];
  mediumCandidates: readonly Message[];
  fullMessages: readonly Message[];
} {
  const nonSystem = messages.filter((m) => m.role !== "system");
  const total = nonSystem.length;

  const fullStart = Math.max(0, total - config.fullTurns);
  const mediumStart = Math.max(0, fullStart - config.mediumTurns);

  return {
    briefCandidates: nonSystem.slice(0, mediumStart),
    mediumCandidates: nonSystem.slice(mediumStart, fullStart),
    fullMessages: nonSystem.slice(fullStart),
  };
}

function formatTurns(messages: readonly Message[]): string {
  return messages
    .map((m) => `[${m.role}] (turn ${m.turnIndex}): ${m.content}`)
    .join("\n");
}

function estimateTokens(text: string): number {
  return Math.ceil(text.split(/\s+/).length * 1.5);
}

async function buildHierarchicalContext(
  messages: readonly Message[],
  client: LLMClient,
  config: TierConfig = DEFAULT_TIER_CONFIG,
  existingBrief: TierSummary | null = null,
  existingMedium: TierSummary | null = null
): Promise<HierarchicalContext> {
  const { briefCandidates, mediumCandidates, fullMessages } =
    partitionMessages(messages, config);

  // --- Medium tier ---
  let mediumSummary: TierSummary | null = null;
  if (mediumCandidates.length > 0) {
    const prompt = MEDIUM_PROMPT_TEMPLATE
      .replace("{maxTokens}", String(config.mediumMaxTokens))
      .replace("{turns}", formatTurns(mediumCandidates));

    const summaryText = await client.complete(prompt);
    mediumSummary = {
      tier: "medium",
      content: summaryText,
      tokenCount: estimateTokens(summaryText),
      turnRange: [
        mediumCandidates[0].turnIndex,
        mediumCandidates[mediumCandidates.length - 1].turnIndex,
      ],
      sourceTurnCount: mediumCandidates.length,
    };
  }

  // --- Brief tier ---
  let briefSummary: TierSummary | null = null;
  if (briefCandidates.length > 0) {
    const sourceText =
      existingBrief && existingMedium
        ? `${existingBrief.content}\n\nAdditional context:\n${existingMedium.content}`
        : formatTurns(briefCandidates);

    const prompt = BRIEF_PROMPT_TEMPLATE
      .replace("{maxTokens}", String(config.briefMaxTokens))
      .replace("{summary}", sourceText);

    const briefText = await client.complete(prompt);
    briefSummary = {
      tier: "brief",
      content: briefText,
      tokenCount: estimateTokens(briefText),
      turnRange: [
        briefCandidates[0].turnIndex,
        briefCandidates[briefCandidates.length - 1].turnIndex,
      ],
      sourceTurnCount: briefCandidates.length,
    };
  }

  // --- Assemble ---
  const fullTokens = fullMessages.reduce((s, m) => s + m.tokenCount, 0);
  const totalTokenCount =
    fullTokens +
    (mediumSummary?.tokenCount ?? 0) +
    (briefSummary?.tokenCount ?? 0);

  return {
    briefSummary,
    mediumSummary,
    fullMessages,
    totalTokenCount,
  };
}

function renderContext(ctx: HierarchicalContext): readonly Message[] {
  const rendered: Message[] = [];

  if (ctx.briefSummary) {
    rendered.push({
      role: "system",
      content: `[BRIEF SUMMARY - turns ${ctx.briefSummary.turnRange[0]}-${ctx.briefSummary.turnRange[1]}]\n${ctx.briefSummary.content}`,
      tokenCount: ctx.briefSummary.tokenCount,
      turnIndex: -2,
    });
  }

  if (ctx.mediumSummary) {
    rendered.push({
      role: "system",
      content: `[DETAILED SUMMARY - turns ${ctx.mediumSummary.turnRange[0]}-${ctx.mediumSummary.turnRange[1]}]\n${ctx.mediumSummary.content}`,
      tokenCount: ctx.mediumSummary.tokenCount,
      turnIndex: -1,
    });
  }

  rendered.push(...ctx.fullMessages);
  return rendered;
}

// --- Usage Example ---

async function agentLoop(client: LLMClient): Promise<void> {
  const config: TierConfig = {
    fullTurns: 6,
    mediumTurns: 12,
    mediumMaxTokens: 400,
    briefMaxTokens: 100,
  };

  let messages: Message[] = [];
  let turnCounter = 0;
  let briefCache: TierSummary | null = null;
  let mediumCache: TierSummary | null = null;

  while (true) {
    const userInput = await getUserInput();
    turnCounter += 1;
    messages = [
      ...messages,
      {
        role: "user",
        content: userInput,
        tokenCount: estimateTokens(userInput),
        turnIndex: turnCounter,
      },
    ];

    let rendered: readonly Message[];
    if (messages.length > config.fullTurns) {
      const ctx = await buildHierarchicalContext(
        messages,
        client,
        config,
        briefCache,
        mediumCache
      );
      briefCache = ctx.briefSummary;
      mediumCache = ctx.mediumSummary;
      rendered = renderContext(ctx);
    } else {
      rendered = messages;
    }

    const response = await client.complete(formatForChat(rendered));
    turnCounter += 1;
    messages = [
      ...messages,
      {
        role: "assistant",
        content: response,
        tokenCount: estimateTokens(response),
        turnIndex: turnCounter,
      },
    ];
  }
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Preserves high-fidelity detail where it matters most (recent turns) | Requires two LLM summarization calls instead of one (medium + brief) |
| Older context is retained at lower cost rather than discarded entirely | Tier boundaries are fixed -- a critical decision at turn 5 gets the same compression as a tangent at turn 6 |
| Temporal ordering is preserved across tiers | More complex to implement than flat compaction |
| Cascading compression prevents unbounded growth at any tier | Repeated compression (brief-of-medium-of-full) compounds information loss |
| Configurable tier sizes let you tune the resolution gradient | Summarization quality depends on the model; poor summaries at the medium tier cascade into poor brief summaries |

## When to Use

- Long-running sessions (50+ turns) where both recent detail and historical context matter.
- Agent workflows that revisit earlier decisions -- the brief tier keeps them accessible without full-resolution cost.
- Multi-phase projects where the model needs to know "what phase are we in" (brief tier) and "what did we just do" (full tier) simultaneously.
- Systems that need a predictable token budget -- each tier has a known maximum size, making total context size bounded and calculable.

## When NOT to Use

- Short sessions (under 20 turns) where flat compaction or no compression is sufficient.
- When all context is equally important regardless of age (e.g., legal document review where every clause matters).
- When you cannot afford two LLM calls per compression cycle -- use single-pass Conversation Compaction instead.
- Real-time systems where even one summarization call adds unacceptable latency.

## Related Patterns

- **[Conversation Compaction](conversation-compaction.md)** -- Single-tier compression. Hierarchical Summarization extends this with multiple resolution levels. Use Conversation Compaction when a single summary tier is sufficient.
- **[Observation Masking](observation-masking.md)** -- Complementary pattern that compresses tool outputs deterministically. Apply observation masking to tool results within each tier before summarizing.
- **[Lossy Context Distillation](lossy-context-distillation.md)** -- If even the brief tier is too verbose, distillation can extract only task-relevant facts from it.
- **[Token Budget Allocation](token-budget-allocation.md)** -- Use budget allocation to set the token limits for each tier, ensuring the hierarchical context stays within its allocated share of the context window.

## Real-World Examples

1. **Claude Code's automatic compaction with recency awareness** -- When Claude Code compacts context, it preserves recent file modifications and test commands at higher fidelity than older conversational turns, effectively implementing a two-tier hierarchy.

2. **Google Gemini's long-context summarization** -- Gemini models with 1M+ token windows still benefit from hierarchical summarization when sessions exceed even those limits, or when cost optimization requires using fewer tokens despite having the capacity.

3. **Meeting transcription assistants** -- Tools like Otter.ai and Fireflies produce hierarchical summaries: a brief headline, a medium-detail action-items list, and full transcript access. The same principle applies to LLM context management.

4. **Git log as hierarchical history** -- `git log --oneline` (brief), `git log` (medium), and `git show` (full) provide the same information at different resolution levels. Hierarchical Summarization applies this principle to conversation history.
