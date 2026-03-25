# Context Isolation Patterns

Context isolation is the practice of **preventing context pollution between tasks** by giving each unit of work its own bounded context window. Instead of cramming everything into a single growing context, isolation patterns partition work across separate agent instances, each with only the information it needs.

These patterns answer the question: *How do I stop one task's context from degrading another task's performance?*

## Decision Tree

```
Start here: What is your multi-task challenge?
|
|-- "My agent's context gets polluted by exploratory sub-tasks"
|     --> Sub-Agent Delegation
|
|-- "I have multiple specialized agents that need to collaborate"
|     --> Multi-Agent Context Orchestration
|
|-- "I need to execute untrusted code or content safely"
|     --> Sandbox Contexts
|
|-- "I need to test multiple hypotheses without bias"
|     --> Sandbox Contexts (competing sandboxes)
|
|-- "Different phases of my workflow need different context views"
|     --> Role-Based Context Partitioning
|
|-- "My reviewer is biased because it also wrote the code"
|     --> Role-Based Context Partitioning
|
|-- "I need parallel execution of independent research tasks"
|     --> Sub-Agent Delegation (supports concurrent child agents)
|
|-- "I need to define strict context contracts between team members"
|     --> Multi-Agent Context Orchestration
|
|-- "I have a single complex task that involves both research and action"
|     --> Sub-Agent Delegation for research, parent agent for action
|
|-- "I have a pipeline where Agent A's output feeds Agent B's input"
|     --> Multi-Agent Context Orchestration with sequential routing
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [Sub-Agent Delegation](sub-agent-delegation.md) | Spawn child agents with isolated contexts for sub-tasks | Context pollution prevention and parallel execution |
| [Multi-Agent Context Orchestration](multi-agent-context-orchestration.md) | Manage context flow between collaborating agents via contracts | Scalable multi-agent collaboration without context explosion |
| [Sandbox Contexts](sandbox-contexts.md) | Create disposable, sandboxed environments for risky or exploratory operations | Safe execution with zero context leakage |
| [Role-Based Context Partitioning](role-based-context-partitioning.md) | Partition context by the agent's current role, hiding irrelevant information | Elimination of cross-role bias and context bloat |

## How They Compose

These four patterns address isolation at different scales and for different purposes:

- **Sub-Agent Delegation** is a parent-child relationship. A single orchestrator spawns focused workers, gives them minimal context, and collects their results. The parent's context stays clean because exploratory work happens in disposable child contexts.
- **Multi-Agent Context Orchestration** is a peer-to-peer (or pipeline) relationship. Multiple specialized agents collaborate, and an orchestration layer controls what context flows between them, preventing any single agent from accumulating the full context of all agents.
- **Sandbox Contexts** are disposable execution environments optimized for safety. Unlike sub-agents (which return rich results), sandboxes are designed to be tainted, corrupted, or crashed without consequence. They add output sanitization and injection detection on top of basic isolation.
- **Role-Based Context Partitioning** filters context within a single workflow based on the agent's current role. Rather than spawning separate agents, it controls what is *visible* at each phase, preventing information from one role (e.g., implementation rationale) from biasing another role (e.g., code review).

In practice, they nest naturally: a multi-agent orchestrator coordinates specialized agents, each with role-based context partitioning for its internal workflow. Within any role, risky operations can be sandboxed. And any agent may use sub-agent delegation for its own sub-tasks. The orchestration layer defines the macro context flow; the other patterns handle isolation at progressively finer granularity.
