# Cross-Session State Sync

> Synchronize agent state across multiple concurrent sessions and deployment instances so that knowledge learned in one session is available in all others.

## Problem

Users interact with AI agents across multiple contexts: different chat windows, different devices, different applications that share the same agent backend. Without state synchronization:

- **Knowledge fragmentation**: The agent learns a user preference in Session A (desktop), but Session B (mobile) does not know about it.
- **Conflicting state**: Two sessions modify the same piece of state concurrently, and one change silently overwrites the other.
- **Stale reads**: A session reads state that was already updated by another session, leading to outdated behavior.
- **Lost progress**: Ongoing task tracking in one session is invisible to other sessions, causing duplicate work or contradictory actions.

This problem intensifies with multi-agent architectures where specialized agents may run on different infrastructure but need a shared understanding of the world.

## Solution

Implement a **state synchronization layer** that sits between agent sessions and the persistent state store. Each session reads from and writes to this layer, which handles:

1. **Optimistic concurrency**: Sessions read the latest state and attach a version number. Writes succeed only if the version has not changed since the read.
2. **Conflict resolution**: When concurrent writes conflict, a deterministic resolution strategy (last-write-wins, merge, or manual) resolves the conflict.
3. **Event-driven propagation**: State changes emit events that other active sessions can subscribe to, enabling near-real-time sync without polling.

## How It Works

```
Session A (Desktop)                    Session B (Mobile)
+-------------------+                  +-------------------+
| Agent instance    |                  | Agent instance    |
| State: v5         |                  | State: v5         |
+--------+----------+                  +--------+----------+
         |                                      |
         | write(preference, v5->v6)             |
         v                                      |
+----------------------------------------------|----------+
| State Sync Layer                              |          |
|                                               |          |
| 1. Validate version: v5 is current?  YES      |          |
| 2. Apply write -> state becomes v6            |          |
| 3. Emit event: {type: "preference_updated",   |          |
|                  version: 6}                   |          |
|                                    event ----->|          |
+------------------------------------------------+---------+
                                                 |
                                     Session B receives event
                                     Updates local state to v6
                                     +-------------------+
                                     | Agent instance    |
                                     | State: v6         |
                                     +-------------------+

Concurrent conflict scenario:
Session A: write(x=10, v5->v6)   Session B: write(x=20, v5->v6)
                     |                         |
                     v                         v
               +-----+-------------------------+-----+
               | Sync Layer                           |
               | A arrives first -> v6, x=10          |
               | B arrives second -> version mismatch |
               |   -> conflict resolution strategy    |
               |   -> merge / reject / last-write-wins|
               +--------------------------------------+
```

## Implementation

### Python

```python
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ConflictStrategy(str, Enum):
    LAST_WRITE_WINS = "last_write_wins"
    MERGE = "merge"
    REJECT = "reject"


@dataclass(frozen=True)
class StateVersion:
    """Versioned state snapshot."""
    version: int
    data: dict[str, Any]
    updated_at: float
    updated_by: str  # session ID that made the change


@dataclass(frozen=True)
class StateChangeEvent:
    """Event emitted when state changes."""
    event_id: str
    key: str
    old_value: Any
    new_value: Any
    version: int
    source_session: str
    timestamp: float


@dataclass(frozen=True)
class ConflictRecord:
    """Record of a write conflict and how it was resolved."""
    key: str
    local_value: Any
    remote_value: Any
    resolved_value: Any
    strategy: ConflictStrategy
    timestamp: float


class StateStore:
    """Thread-safe versioned state store with optimistic concurrency.

    Manages mutable shared state by design -- this is the single source of
    truth that all sessions synchronize against.
    """

    def __init__(self):
        self._state: dict[str, Any] = {}
        self._version: int = 0
        self._lock = asyncio.Lock()
        self._history: list[StateChangeEvent] = []
        self._subscribers: list[Callable[[StateChangeEvent], None]] = []

    @property
    def version(self) -> int:
        return self._version

    async def read(self) -> StateVersion:
        """Read the current state with its version number."""
        async with self._lock:
            return StateVersion(
                version=self._version,
                data=dict(self._state),
                updated_at=time.time(),
                updated_by="",
            )

    async def write(
        self,
        key: str,
        value: Any,
        expected_version: int,
        session_id: str,
    ) -> tuple[bool, StateVersion]:
        """Attempt a versioned write. Fails if version has changed.

        Returns (success, current_state_version).
        """
        async with self._lock:
            if expected_version != self._version:
                # Conflict: version mismatch
                current = StateVersion(
                    version=self._version,
                    data=dict(self._state),
                    updated_at=time.time(),
                    updated_by="",
                )
                return False, current

            old_value = self._state.get(key)
            self._state[key] = value
            self._version += 1

            event = StateChangeEvent(
                event_id=f"evt-{self._version}",
                key=key,
                old_value=old_value,
                new_value=value,
                version=self._version,
                source_session=session_id,
                timestamp=time.time(),
            )
            self._history.append(event)

            # Notify subscribers
            for subscriber in self._subscribers:
                subscriber(event)

            return True, StateVersion(
                version=self._version,
                data=dict(self._state),
                updated_at=time.time(),
                updated_by=session_id,
            )

    def subscribe(self, callback: Callable[[StateChangeEvent], None]) -> None:
        """Subscribe to state change events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StateChangeEvent], None]) -> None:
        """Unsubscribe from state change events."""
        self._subscribers = [s for s in self._subscribers if s != callback]


class SessionStateSync:
    """Per-session state synchronization with conflict resolution.

    Each agent session creates one of these to interact with shared state.
    """

    def __init__(
        self,
        session_id: str,
        store: StateStore,
        conflict_strategy: ConflictStrategy = ConflictStrategy.LAST_WRITE_WINS,
        merge_fn: Callable[[Any, Any], Any] | None = None,
    ):
        self._session_id = session_id
        self._store = store
        self._conflict_strategy = conflict_strategy
        self._merge_fn = merge_fn or self._default_merge
        self._local_version: int = 0
        self._local_cache: dict[str, Any] = {}
        self._conflicts: list[ConflictRecord] = []
        self._pending_events: list[StateChangeEvent] = []

        # Subscribe to remote changes
        self._store.subscribe(self._on_remote_change)

    async def read_state(self) -> dict[str, Any]:
        """Read the latest synchronized state."""
        state_version = await self._store.read()
        self._local_version = state_version.version
        self._local_cache = dict(state_version.data)
        return dict(self._local_cache)

    async def write_state(
        self, key: str, value: Any
    ) -> tuple[bool, dict[str, Any]]:
        """Write a state change with optimistic concurrency.

        If a conflict is detected, applies the configured conflict
        resolution strategy and retries.
        """
        success, state_version = await self._store.write(
            key=key,
            value=value,
            expected_version=self._local_version,
            session_id=self._session_id,
        )

        if success:
            self._local_version = state_version.version
            self._local_cache = dict(state_version.data)
            return True, dict(self._local_cache)

        # Conflict detected -- apply resolution strategy
        remote_value = state_version.data.get(key)
        resolved = self._resolve_conflict(key, value, remote_value)

        # Retry with resolved value and updated version
        self._local_version = state_version.version
        success, state_version = await self._store.write(
            key=key,
            value=resolved,
            expected_version=self._local_version,
            session_id=self._session_id,
        )

        self._local_version = state_version.version
        self._local_cache = dict(state_version.data)
        return success, dict(self._local_cache)

    def get_pending_events(self) -> list[StateChangeEvent]:
        """Get state changes from other sessions since last check."""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    @property
    def conflicts(self) -> list[ConflictRecord]:
        return list(self._conflicts)

    def _on_remote_change(self, event: StateChangeEvent) -> None:
        """Handle state change events from other sessions."""
        if event.source_session == self._session_id:
            return  # Ignore our own changes
        self._pending_events.append(event)
        # Update local cache
        self._local_cache[event.key] = event.new_value
        self._local_version = event.version

    def _resolve_conflict(
        self, key: str, local_value: Any, remote_value: Any
    ) -> Any:
        """Resolve a write conflict using the configured strategy."""
        if self._conflict_strategy == ConflictStrategy.LAST_WRITE_WINS:
            resolved = local_value  # Our write wins
        elif self._conflict_strategy == ConflictStrategy.MERGE:
            resolved = self._merge_fn(local_value, remote_value)
        elif self._conflict_strategy == ConflictStrategy.REJECT:
            resolved = remote_value  # Remote wins, reject ours
        else:
            resolved = local_value

        self._conflicts.append(
            ConflictRecord(
                key=key,
                local_value=local_value,
                remote_value=remote_value,
                resolved_value=resolved,
                strategy=self._conflict_strategy,
                timestamp=time.time(),
            )
        )
        return resolved

    @staticmethod
    def _default_merge(local: Any, remote: Any) -> Any:
        """Default merge: combine dicts, prefer local for scalar conflicts."""
        if isinstance(local, dict) and isinstance(remote, dict):
            return {**remote, **local}
        if isinstance(local, list) and isinstance(remote, list):
            # Deduplicated union
            seen: set[str] = set()
            merged = []
            for item in remote + local:
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            return merged
        return local  # Scalar conflict: local wins

    def close(self) -> None:
        """Unsubscribe from state events."""
        self._store.unsubscribe(self._on_remote_change)


# --- Usage Example ---

async def multi_session_demo():
    """Demonstrate state sync across two concurrent sessions."""
    store = StateStore()

    # Two sessions connected to the same store
    session_a = SessionStateSync("session-desktop", store)
    session_b = SessionStateSync("session-mobile", store)

    # Session A sets a preference
    await session_a.write_state("preferred_language", "python")

    # Session B reads -- sees the change from Session A
    state_b = await session_b.read_state()
    assert state_b["preferred_language"] == "python"

    # Both sessions try to update concurrently
    result_a = await session_a.write_state("editor", "vscode")
    result_b = await session_b.write_state("editor", "neovim")

    # Conflict resolved per strategy; both sessions see consistent state
    final_a = await session_a.read_state()
    final_b = await session_b.read_state()
    assert final_a["editor"] == final_b["editor"]

    # Cleanup
    session_a.close()
    session_b.close()
```

### TypeScript

```typescript
type ConflictStrategy = "last_write_wins" | "merge" | "reject";

interface StateVersion {
  readonly version: number;
  readonly data: Readonly<Record<string, unknown>>;
  readonly updatedAt: number;
  readonly updatedBy: string;
}

interface StateChangeEvent {
  readonly eventId: string;
  readonly key: string;
  readonly oldValue: unknown;
  readonly newValue: unknown;
  readonly version: number;
  readonly sourceSession: string;
  readonly timestamp: number;
}

interface ConflictRecord {
  readonly key: string;
  readonly localValue: unknown;
  readonly remoteValue: unknown;
  readonly resolvedValue: unknown;
  readonly strategy: ConflictStrategy;
  readonly timestamp: number;
}

type SubscribeCallback = (event: StateChangeEvent) => void;
type MergeFn = (local: unknown, remote: unknown) => unknown;

class StateStore {
  private state: Record<string, unknown> = {};
  private currentVersion = 0;
  private readonly subscribers: Set<SubscribeCallback> = new Set();
  private readonly history: StateChangeEvent[] = [];

  get version(): number {
    return this.currentVersion;
  }

  read(): StateVersion {
    return {
      version: this.currentVersion,
      data: { ...this.state },
      updatedAt: Date.now(),
      updatedBy: "",
    };
  }

  write(
    key: string,
    value: unknown,
    expectedVersion: number,
    sessionId: string
  ): { success: boolean; stateVersion: StateVersion } {
    if (expectedVersion !== this.currentVersion) {
      return {
        success: false,
        stateVersion: {
          version: this.currentVersion,
          data: { ...this.state },
          updatedAt: Date.now(),
          updatedBy: "",
        },
      };
    }

    const oldValue = this.state[key];
    this.state[key] = value;
    this.currentVersion += 1;

    const event: StateChangeEvent = {
      eventId: `evt-${this.currentVersion}`,
      key,
      oldValue,
      newValue: value,
      version: this.currentVersion,
      sourceSession: sessionId,
      timestamp: Date.now(),
    };
    this.history.push(event);

    for (const subscriber of this.subscribers) {
      subscriber(event);
    }

    return {
      success: true,
      stateVersion: {
        version: this.currentVersion,
        data: { ...this.state },
        updatedAt: Date.now(),
        updatedBy: sessionId,
      },
    };
  }

  subscribe(callback: SubscribeCallback): void {
    this.subscribers.add(callback);
  }

  unsubscribe(callback: SubscribeCallback): void {
    this.subscribers.delete(callback);
  }
}

class SessionStateSync {
  private readonly sessionId: string;
  private readonly store: StateStore;
  private readonly conflictStrategy: ConflictStrategy;
  private readonly mergeFn: MergeFn;
  private localVersion = 0;
  private localCache: Record<string, unknown> = {};
  private readonly conflictLog: ConflictRecord[] = [];
  private readonly pendingEvents: StateChangeEvent[] = [];
  private readonly eventHandler: SubscribeCallback;

  constructor(
    sessionId: string,
    store: StateStore,
    conflictStrategy: ConflictStrategy = "last_write_wins",
    mergeFn?: MergeFn
  ) {
    this.sessionId = sessionId;
    this.store = store;
    this.conflictStrategy = conflictStrategy;
    this.mergeFn = mergeFn ?? this.defaultMerge;

    this.eventHandler = (event: StateChangeEvent) => {
      if (event.sourceSession === this.sessionId) return;
      this.pendingEvents.push(event);
      this.localCache[event.key] = event.newValue;
      this.localVersion = event.version;
    };
    this.store.subscribe(this.eventHandler);
  }

  readState(): Record<string, unknown> {
    const stateVersion = this.store.read();
    this.localVersion = stateVersion.version;
    this.localCache = { ...stateVersion.data };
    return { ...this.localCache };
  }

  writeState(
    key: string,
    value: unknown
  ): { success: boolean; state: Record<string, unknown> } {
    const { success, stateVersion } = this.store.write(
      key,
      value,
      this.localVersion,
      this.sessionId
    );

    if (success) {
      this.localVersion = stateVersion.version;
      this.localCache = { ...stateVersion.data };
      return { success: true, state: { ...this.localCache } };
    }

    // Conflict: resolve and retry
    const remoteValue = stateVersion.data[key];
    const resolved = this.resolveConflict(key, value, remoteValue);
    this.localVersion = stateVersion.version;

    const retry = this.store.write(
      key,
      resolved,
      this.localVersion,
      this.sessionId
    );
    this.localVersion = retry.stateVersion.version;
    this.localCache = { ...retry.stateVersion.data };
    return { success: retry.success, state: { ...this.localCache } };
  }

  getPendingEvents(): StateChangeEvent[] {
    const events = [...this.pendingEvents];
    this.pendingEvents.length = 0;
    return events;
  }

  get conflicts(): readonly ConflictRecord[] {
    return [...this.conflictLog];
  }

  close(): void {
    this.store.unsubscribe(this.eventHandler);
  }

  private resolveConflict(
    key: string,
    localValue: unknown,
    remoteValue: unknown
  ): unknown {
    let resolved: unknown;

    switch (this.conflictStrategy) {
      case "last_write_wins":
        resolved = localValue;
        break;
      case "merge":
        resolved = this.mergeFn(localValue, remoteValue);
        break;
      case "reject":
        resolved = remoteValue;
        break;
      default:
        resolved = localValue;
    }

    this.conflictLog.push({
      key,
      localValue,
      remoteValue,
      resolvedValue: resolved,
      strategy: this.conflictStrategy,
      timestamp: Date.now(),
    });

    return resolved;
  }

  private defaultMerge(local: unknown, remote: unknown): unknown {
    if (
      typeof local === "object" &&
      typeof remote === "object" &&
      local !== null &&
      remote !== null &&
      !Array.isArray(local) &&
      !Array.isArray(remote)
    ) {
      return { ...(remote as Record<string, unknown>), ...(local as Record<string, unknown>) };
    }
    if (Array.isArray(local) && Array.isArray(remote)) {
      const seen = new Set<string>();
      const merged: unknown[] = [];
      for (const item of [...remote, ...local]) {
        const key = JSON.stringify(item);
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(item);
        }
      }
      return merged;
    }
    return local;
  }
}

// --- Usage Example ---

function multiSessionDemo(): void {
  const store = new StateStore();

  const sessionA = new SessionStateSync("session-desktop", store);
  const sessionB = new SessionStateSync("session-mobile", store);

  // Session A sets a preference
  sessionA.writeState("preferred_language", "python");

  // Session B reads -- sees the change
  const stateB = sessionB.readState();
  console.log(`Session B sees: ${stateB.preferred_language}`); // "python"

  // Concurrent updates
  sessionA.writeState("editor", "vscode");
  sessionB.writeState("editor", "neovim");

  // Both sessions converge to the same state
  const finalA = sessionA.readState();
  const finalB = sessionB.readState();
  console.log(`Consistent: ${finalA.editor === finalB.editor}`); // true

  sessionA.close();
  sessionB.close();
}

export {
  StateStore,
  SessionStateSync,
  StateChangeEvent,
  ConflictStrategy,
  ConflictRecord,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Knowledge learned in any session is available everywhere | Requires shared state infrastructure (database, message bus) |
| Conflict resolution prevents silent data loss | Optimistic concurrency adds retry complexity |
| Event-driven propagation enables near-real-time sync | Network partitions can cause temporary inconsistency |
| Audit trail of all state changes via event history | Merge strategies can produce unexpected results if not carefully designed |
| Clean separation between session logic and sync logic | Additional latency for every state read/write |

## When to Use

- Multi-device agents where users switch between desktop, mobile, and web interfaces.
- Multi-agent systems where specialized agents share common state (user preferences, task progress, established facts).
- Long-running workflows where sessions may be interrupted and resumed on different instances.
- Any system where "the agent should remember what I told it in another conversation" is a requirement.
- Collaborative agents where multiple users interact with a shared agent that must maintain consistent state.

## When NOT to Use

- Single-session, stateless interactions where no state persists between calls.
- When there is only ever one active session per user (no concurrency to synchronize).
- When eventual consistency is unacceptable and you need strict linearizability (use a distributed database directly).
- Simple preference stores where a single key-value write with no concurrency is sufficient.
- When the infrastructure cost of a sync layer exceeds the benefit for your use case.

## Related Patterns

- **[Filesystem-as-Memory](filesystem-as-memory.md)** -- Can serve as the backing store for synchronized state. Files on disk are the persistent layer; the sync layer handles concurrent access.
- **[Episodic Memory](episodic-memory.md)** -- Episodes captured in one session should be queryable from another. Cross-session sync ensures the episodic store is consistent.
- **[Semantic Memory Indexing](semantic-memory-indexing.md)** -- The semantic index must be updated when synchronized state changes, so new memories are searchable from all sessions.
- **[Memory Consolidation](memory-consolidation.md)** -- Consolidation may run in a background session, and its results must be synced to all active sessions.

## Real-World Examples

1. **ChatGPT Memory** -- When ChatGPT learns a fact about you ("I prefer Python"), that memory is available in all subsequent conversations, regardless of device or conversation thread.

2. **Apple Intelligence / Siri** -- User context (preferences, routines, established facts) is synchronized across iPhone, iPad, Mac, and HomePod through iCloud, so Siri behaves consistently everywhere.

3. **Notion AI** -- Workspace knowledge is synchronized across all clients. When one user adds information that Notion AI can reference, all other users benefit from that knowledge immediately.

4. **Multiplayer coding agents** -- In pair programming with AI, both human participants and the AI agent share a synchronized understanding of the codebase state, task progress, and decisions made.
