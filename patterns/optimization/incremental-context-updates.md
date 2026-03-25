# Incremental Context Updates

> Apply diffs to existing context between turns instead of rebuilding from scratch, eliminating redundant computation when 90% of the context is stable between iterations.

## Problem

In multi-turn conversations and agent loops, the context window is rebuilt on every turn. A typical agent loop might:

1. Load system prompt (unchanged) -- 2,000 tokens
2. Load tool definitions (unchanged) -- 5,000 tokens
3. Load retrieval results (mostly unchanged) -- 3,000 tokens
4. Load conversation history (one new message) -- 8,000 tokens
5. Load agent state (minor updates) -- 1,000 tokens

The total context is 19,000 tokens, but only a few hundred tokens actually changed since the last turn. Rebuilding everything from scratch on every iteration wastes compute in several ways:

- **Redundant retrieval**: Re-querying the vector store when the query has not changed.
- **Redundant serialization**: Re-formatting tool schemas and system prompts that are identical.
- **Redundant token counting**: Re-estimating token counts for unchanged sections.
- **Redundant validation**: Re-checking context window limits when the delta is minimal.

In high-frequency agent loops (coding agents, data analysis agents), this overhead multiplies across hundreds of iterations per task.

## Solution

Maintain a versioned context state and apply incremental updates (diffs) between turns. Track each context section independently with a content hash. On each turn, identify which sections changed, update only those sections, and reuse the rest. This turns an O(n) rebuild into an O(delta) patch operation.

The approach has three layers:

1. **Section tracking**: Each context section (system prompt, tools, retrieval, history, state) is independently versioned with a content hash.
2. **Change detection**: On each turn, compute what changed and what stayed the same.
3. **Patch application**: Apply only the changes to the existing context, preserving unchanged sections byte-for-byte (which also improves KV-cache hit rates).

## How It Works

```
Turn N context:
  [System Prompt v1][Tools v3][Retrieval v7][History (50 msgs)][State v12]
  |     2000 tok   | 5000 tok|  3000 tok  |    8000 tok      | 1000 tok |

Between turns:
  - User sends new message            --> History: append 1 message
  - Agent calls a tool                --> State: update tool result
  - System prompt unchanged            --> System Prompt: SKIP
  - Tools unchanged                    --> Tools: SKIP
  - Retrieval unchanged (same query)   --> Retrieval: SKIP

Turn N+1 context (incremental):
  [System Prompt v1][Tools v3][Retrieval v7][History (51 msgs)][State v13]
  |    REUSED      | REUSED  |   REUSED   |    +1 message    |  PATCHED |

  Tokens recomputed: ~300 (new message + state update)
  Tokens reused:     ~18,700
  Savings:           98.4%


Diff tracking:
  Section        | Hash (N)  | Hash (N+1) | Changed? | Action
  ---------------|-----------|------------|----------|--------
  system_prompt  | a1b2c3    | a1b2c3     | No       | Reuse
  tools          | d4e5f6    | d4e5f6     | No       | Reuse
  retrieval      | g7h8i9    | g7h8i9     | No       | Reuse
  history        | j0k1l2    | m3n4o5     | Yes      | Append
  agent_state    | p6q7r8    | s9t0u1     | Yes      | Patch
```

## Implementation

### Python

```python
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChangeType(Enum):
    UNCHANGED = "unchanged"
    APPENDED = "appended"
    REPLACED = "replaced"
    REMOVED = "removed"
    ADDED = "added"


@dataclass(frozen=True)
class SectionVersion:
    """A versioned snapshot of a context section."""
    name: str
    content: str
    content_hash: str
    version: int
    token_estimate: int
    updated_at: float


@dataclass(frozen=True)
class SectionDiff:
    """A diff describing what changed in a section."""
    name: str
    change_type: ChangeType
    old_version: int
    new_version: int
    tokens_changed: int
    details: str = ""


@dataclass(frozen=True)
class ContextSnapshot:
    """A complete context state at a point in time."""
    sections: dict[str, SectionVersion]
    total_tokens: int
    turn_number: int
    timestamp: float

    def to_prompt_sections(self) -> list[tuple[str, str]]:
        """Return ordered (name, content) pairs for prompt assembly."""
        return [
            (name, section.content)
            for name, section in sorted(
                self.sections.items(),
                key=lambda kv: kv[1].version,
            )
        ]


class IncrementalContextUpdater:
    """
    Tracks context state across turns and applies incremental updates.

    Instead of rebuilding context from scratch on each turn, this maintains
    a versioned snapshot and computes minimal diffs. Unchanged sections
    are reused byte-for-byte, which also benefits KV-cache optimization.
    """

    def __init__(self, section_order: list[str] | None = None):
        """
        Args:
            section_order: Optional ordered list of section names.
                Controls the order in which sections appear in the
                assembled context.
        """
        self._sections: dict[str, SectionVersion] = {}
        self._section_order = section_order or []
        self._turn_number = 0
        self._history: list[list[SectionDiff]] = []
        self._total_tokens_saved = 0

    def update_section(self, name: str, content: str) -> SectionDiff:
        """
        Update a single context section. Returns the diff.

        If the content is identical to the current version (same hash),
        no update is performed and the section is marked UNCHANGED.
        """
        new_hash = self._hash_content(content)
        new_tokens = len(content) // 4  # Rough estimate
        now = time.time()

        existing = self._sections.get(name)

        if existing is None:
            # New section
            version = SectionVersion(
                name=name,
                content=content,
                content_hash=new_hash,
                version=1,
                token_estimate=new_tokens,
                updated_at=now,
            )
            self._sections = {**self._sections, name: version}
            if name not in self._section_order:
                self._section_order = [*self._section_order, name]

            return SectionDiff(
                name=name,
                change_type=ChangeType.ADDED,
                old_version=0,
                new_version=1,
                tokens_changed=new_tokens,
            )

        if existing.content_hash == new_hash:
            # No change
            self._total_tokens_saved += existing.token_estimate
            return SectionDiff(
                name=name,
                change_type=ChangeType.UNCHANGED,
                old_version=existing.version,
                new_version=existing.version,
                tokens_changed=0,
            )

        # Content changed -- detect if it is an append or a full replacement
        is_append = content.startswith(existing.content)
        change_type = ChangeType.APPENDED if is_append else ChangeType.REPLACED
        tokens_changed = (
            new_tokens - existing.token_estimate
            if is_append
            else new_tokens
        )

        version = SectionVersion(
            name=name,
            content=content,
            content_hash=new_hash,
            version=existing.version + 1,
            token_estimate=new_tokens,
            updated_at=now,
        )
        self._sections = {**self._sections, name: version}

        if is_append:
            self._total_tokens_saved += existing.token_estimate

        return SectionDiff(
            name=name,
            change_type=change_type,
            old_version=existing.version,
            new_version=version.version,
            tokens_changed=abs(tokens_changed),
            details="appended content" if is_append else "full replacement",
        )

    def remove_section(self, name: str) -> SectionDiff | None:
        """Remove a section from the context."""
        existing = self._sections.get(name)
        if existing is None:
            return None

        self._sections = {
            k: v for k, v in self._sections.items() if k != name
        }
        self._section_order = [
            s for s in self._section_order if s != name
        ]

        return SectionDiff(
            name=name,
            change_type=ChangeType.REMOVED,
            old_version=existing.version,
            new_version=0,
            tokens_changed=existing.token_estimate,
        )

    def apply_turn(
        self, updates: dict[str, str]
    ) -> tuple[ContextSnapshot, list[SectionDiff]]:
        """
        Apply a batch of section updates for a new turn.

        Args:
            updates: Dict mapping section names to new content.
                Sections not included in the dict are left unchanged.

        Returns:
            A tuple of (new snapshot, list of diffs for this turn).
        """
        self._turn_number += 1
        diffs = []

        for name, content in updates.items():
            diff = self.update_section(name, content)
            diffs.append(diff)

        self._history = [*self._history, diffs]

        snapshot = self.snapshot()
        return snapshot, diffs

    def snapshot(self) -> ContextSnapshot:
        """Return the current context state as an immutable snapshot."""
        total_tokens = sum(
            s.token_estimate for s in self._sections.values()
        )
        return ContextSnapshot(
            sections=dict(self._sections),
            total_tokens=total_tokens,
            turn_number=self._turn_number,
            timestamp=time.time(),
        )

    def build_context(self) -> str:
        """Assemble the full context string from current sections."""
        ordered = []
        for name in self._section_order:
            section = self._sections.get(name)
            if section:
                ordered.append(section.content)

        # Include any sections not in the explicit order
        for name, section in self._sections.items():
            if name not in self._section_order:
                ordered.append(section.content)

        return "\n\n".join(ordered)

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens avoided through incremental reuse."""
        return self._total_tokens_saved

    @property
    def turn_number(self) -> int:
        return self._turn_number

    @property
    def section_names(self) -> list[str]:
        return list(self._sections.keys())

    @property
    def change_summary(self) -> dict[str, Any]:
        """Summary of changes across all turns."""
        if not self._history:
            return {"turns": 0, "total_diffs": 0}

        all_diffs = [d for turn_diffs in self._history for d in turn_diffs]
        unchanged = sum(
            1 for d in all_diffs if d.change_type == ChangeType.UNCHANGED
        )
        changed = len(all_diffs) - unchanged

        return {
            "turns": self._turn_number,
            "total_diffs": len(all_diffs),
            "unchanged_sections": unchanged,
            "changed_sections": changed,
            "reuse_rate": (
                (unchanged / len(all_diffs) * 100) if all_diffs else 0
            ),
            "tokens_saved": self._total_tokens_saved,
        }

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### TypeScript

```typescript
import { createHash } from "crypto";

type ChangeType = "unchanged" | "appended" | "replaced" | "removed" | "added";

interface SectionVersion {
  readonly name: string;
  readonly content: string;
  readonly contentHash: string;
  readonly version: number;
  readonly tokenEstimate: number;
  readonly updatedAt: number;
}

interface SectionDiff {
  readonly name: string;
  readonly changeType: ChangeType;
  readonly oldVersion: number;
  readonly newVersion: number;
  readonly tokensChanged: number;
  readonly details: string;
}

interface ContextSnapshot {
  readonly sections: ReadonlyMap<string, SectionVersion>;
  readonly totalTokens: number;
  readonly turnNumber: number;
  readonly timestamp: number;
}

interface ChangeSummary {
  readonly turns: number;
  readonly totalDiffs: number;
  readonly unchangedSections: number;
  readonly changedSections: number;
  readonly reuseRate: number;
  readonly tokensSaved: number;
}

function hashContent(content: string): string {
  return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

class IncrementalContextUpdater {
  private sections: ReadonlyMap<string, SectionVersion> = new Map();
  private sectionOrder: readonly string[] = [];
  private turnNumber = 0;
  private history: readonly (readonly SectionDiff[])[] = [];
  private totalTokensSaved = 0;

  constructor(sectionOrder?: string[]) {
    this.sectionOrder = Object.freeze(sectionOrder ?? []);
  }

  updateSection(name: string, content: string): SectionDiff {
    const newHash = hashContent(content);
    const newTokens = Math.floor(content.length / 4);
    const now = Date.now() / 1000;
    const existing = this.sections.get(name);

    if (!existing) {
      const version: SectionVersion = {
        name,
        content,
        contentHash: newHash,
        version: 1,
        tokenEstimate: newTokens,
        updatedAt: now,
      };
      this.sections = new Map([...this.sections, [name, version]]);
      if (!this.sectionOrder.includes(name)) {
        this.sectionOrder = Object.freeze([...this.sectionOrder, name]);
      }
      return {
        name,
        changeType: "added",
        oldVersion: 0,
        newVersion: 1,
        tokensChanged: newTokens,
        details: "",
      };
    }

    if (existing.contentHash === newHash) {
      this.totalTokensSaved += existing.tokenEstimate;
      return {
        name,
        changeType: "unchanged",
        oldVersion: existing.version,
        newVersion: existing.version,
        tokensChanged: 0,
        details: "",
      };
    }

    const isAppend = content.startsWith(existing.content);
    const changeType: ChangeType = isAppend ? "appended" : "replaced";
    const tokensChanged = isAppend
      ? newTokens - existing.tokenEstimate
      : newTokens;

    const version: SectionVersion = {
      name,
      content,
      contentHash: newHash,
      version: existing.version + 1,
      tokenEstimate: newTokens,
      updatedAt: now,
    };
    this.sections = new Map([...this.sections, [name, version]]);

    if (isAppend) {
      this.totalTokensSaved += existing.tokenEstimate;
    }

    return {
      name,
      changeType,
      oldVersion: existing.version,
      newVersion: version.version,
      tokensChanged: Math.abs(tokensChanged),
      details: isAppend ? "appended content" : "full replacement",
    };
  }

  removeSection(name: string): SectionDiff | null {
    const existing = this.sections.get(name);
    if (!existing) return null;

    const newSections = new Map(this.sections);
    newSections.delete(name);
    this.sections = newSections;
    this.sectionOrder = Object.freeze(
      this.sectionOrder.filter((s) => s !== name)
    );

    return {
      name,
      changeType: "removed",
      oldVersion: existing.version,
      newVersion: 0,
      tokensChanged: existing.tokenEstimate,
      details: "",
    };
  }

  applyTurn(
    updates: Record<string, string>
  ): { snapshot: ContextSnapshot; diffs: readonly SectionDiff[] } {
    this.turnNumber += 1;
    const diffs: SectionDiff[] = [];

    for (const [name, content] of Object.entries(updates)) {
      diffs.push(this.updateSection(name, content));
    }

    this.history = Object.freeze([...this.history, Object.freeze(diffs)]);

    return { snapshot: this.snapshot(), diffs };
  }

  snapshot(): ContextSnapshot {
    let totalTokens = 0;
    for (const section of this.sections.values()) {
      totalTokens += section.tokenEstimate;
    }

    return {
      sections: new Map(this.sections),
      totalTokens,
      turnNumber: this.turnNumber,
      timestamp: Date.now() / 1000,
    };
  }

  buildContext(): string {
    const ordered: string[] = [];

    for (const name of this.sectionOrder) {
      const section = this.sections.get(name);
      if (section) ordered.push(section.content);
    }

    for (const [name, section] of this.sections) {
      if (!this.sectionOrder.includes(name)) {
        ordered.push(section.content);
      }
    }

    return ordered.join("\n\n");
  }

  get tokensSaved(): number {
    return this.totalTokensSaved;
  }

  get currentTurn(): number {
    return this.turnNumber;
  }

  get sectionNames(): readonly string[] {
    return [...this.sections.keys()];
  }

  get changeSummary(): ChangeSummary {
    if (this.history.length === 0) {
      return {
        turns: 0,
        totalDiffs: 0,
        unchangedSections: 0,
        changedSections: 0,
        reuseRate: 0,
        tokensSaved: 0,
      };
    }

    const allDiffs = this.history.flat();
    const unchanged = allDiffs.filter(
      (d) => d.changeType === "unchanged"
    ).length;
    const changed = allDiffs.length - unchanged;

    return {
      turns: this.turnNumber,
      totalDiffs: allDiffs.length,
      unchangedSections: unchanged,
      changedSections: changed,
      reuseRate:
        allDiffs.length > 0 ? (unchanged / allDiffs.length) * 100 : 0,
      tokensSaved: this.totalTokensSaved,
    };
  }
}

export {
  IncrementalContextUpdater,
  SectionVersion,
  SectionDiff,
  ContextSnapshot,
  ChangeSummary,
  ChangeType,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Eliminates redundant computation for unchanged context sections (often 90%+ reuse) | Adds complexity with version tracking and diff computation |
| Naturally improves KV-cache hit rates by preserving byte-identical prefixes | State management bugs can cause stale sections to persist incorrectly |
| Change history provides visibility into context evolution over time | Not beneficial for single-turn or low-frequency interactions |
| Append detection preserves KV-cache validity for growing sections (like history) | Requires all context updates to flow through the updater (no bypassing) |
| Token savings compound over long agent loops (hundreds of iterations) | Hash computation adds a small fixed cost per section per turn |
| Snapshots enable rollback and debugging of context state | Memory usage grows with history tracking (mitigated by capping history) |

## When to Use

- Agent loops with high iteration counts (coding agents, research agents, data analysis)
- Multi-turn conversations where context is large and mostly stable between turns
- Systems where context assembly involves expensive operations (retrieval, API calls)
- Real-time applications where per-turn latency must be minimized
- Any system where you have measured that context is 80%+ stable between turns

## When NOT to Use

- Single-turn interactions with no repeated context
- Systems where context changes dramatically between turns (new topic each message)
- Simple chatbots with small context windows where rebuild cost is negligible
- When context assembly is already fast enough without optimization
- Prototyping and experimentation where simplicity matters more than efficiency

## Related Patterns

- **KV-Cache Optimization** (Optimization): Incremental updates naturally preserve byte-identical prefixes, maximizing KV-cache hit rates. The two patterns reinforce each other.
- **Parallel Context Assembly** (Optimization): For the sections that do change, use parallel assembly to fetch updates concurrently. Incremental updates tell you which sections need re-fetching.
- **Prompt Caching Strategies** (Optimization): Component-level caching (L3) and incremental updates solve the same problem from different angles. Caching stores results externally; incremental updates track state internally.
- **Context Rot Detection** (Evaluation): Version tracking enables staleness detection. If a section has not been updated in many turns, it may contain outdated information.

## Real-World Examples

- **Claude Code**: Maintains conversation context incrementally across tool calls. The system prompt and tool definitions remain stable while tool outputs and user messages are appended. This is why Claude Code sessions feel responsive even with large context windows.
- **Cursor/Copilot**: Code editor AI assistants track file changes incrementally. When you edit line 50 of a file, the assistant does not re-read the entire codebase -- it patches the changed region and reuses everything else.
- **Devin**: Maintains agent state across planning and execution loops, applying incremental updates as tasks progress. The planning context evolves gradually rather than being rebuilt each iteration.
- **React/Redux Pattern**: The broader software engineering pattern of immutable state with diffs (like React's virtual DOM diffing) applies directly to context management. Track state, compute diffs, apply minimal updates.
