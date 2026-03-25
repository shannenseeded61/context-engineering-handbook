# Role-Based Context Partitioning

> Partition context based on the agent's current role in a workflow, ensuring each role sees only the information relevant to its function and preventing cross-role bias.

## Problem

When an agent performs multiple roles in a workflow -- researcher, implementer, reviewer, tester -- all accumulated context is visible to every subsequent role. This creates several problems:

- **Anchoring bias**: A reviewer who also wrote the code cannot objectively evaluate it because the implementation rationale is fresh in context, making the code seem more reasonable than it might be to a fresh reader.
- **Information leakage**: A "researcher" role exploring options should not see implementation details that anchor it toward a specific solution before research is complete.
- **Context bloat**: Each role accumulates context that is irrelevant to other roles. A reviewer does not need the 50 research links the researcher evaluated.
- **Role confusion**: When all context is visible, the agent struggles to stay in role. A reviewer starts suggesting implementation changes; a researcher starts writing code.

Without partitioning, multi-role workflows degrade into a single undifferentiated agent with too much context and no clear boundaries.

## Solution

Define explicit **roles** with **context visibility rules**. When the agent transitions between roles, a partitioning layer filters the context -- hiding information that belongs to other roles and surfacing only what the current role needs. Role transitions pass a structured handoff document, not the raw context.

Each role definition specifies:
- What context it can see (its own history plus approved cross-role information)
- What context it cannot see (other roles' deliberation, intermediate work)
- What it must produce as output (the handoff contract for downstream roles)

## How It Works

```
Workflow: Feature Implementation
+-------------------------------------------------------------+
|                                                               |
|  RESEARCHER role                                              |
|  Sees: requirements, existing docs, search results            |
|  Hidden: (nothing yet -- first role)                          |
|  Produces: Research Summary (3-5 options with trade-offs)     |
|                                                               |
|       | Handoff: Research Summary only                        |
|       v                                                       |
|  ARCHITECT role                                               |
|  Sees: requirements, Research Summary                         |
|  Hidden: raw search results, dead-end explorations            |
|  Produces: Architecture Decision Record                       |
|                                                               |
|       | Handoff: ADR only                                     |
|       v                                                       |
|  IMPLEMENTER role                                             |
|  Sees: requirements, ADR, relevant source files               |
|  Hidden: research options, architectural deliberation         |
|  Produces: Code changes + implementation notes                |
|                                                               |
|       | Handoff: diff + implementation notes only              |
|       v                                                       |
|  REVIEWER role                                                |
|  Sees: requirements, ADR (summary), diff, test results        |
|  Hidden: implementation deliberation, research, alternatives  |
|  Produces: Review verdict with action items                   |
|                                                               |
+-------------------------------------------------------------+

Each role sees a FILTERED view of the workflow.
No role sees another role's internal reasoning.
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Role(str, Enum):
    RESEARCHER = "researcher"
    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    REVIEWER = "reviewer"
    TESTER = "tester"


@dataclass(frozen=True)
class ContextItem:
    """A single piece of context with role visibility metadata."""
    item_id: str
    content: str
    source_role: Role | None  # None = shared/system context
    category: str             # e.g., "requirement", "research", "code", "decision"
    visibility: frozenset[Role] = field(
        default_factory=lambda: frozenset(Role)
    )

    def visible_to(self, role: Role) -> bool:
        return role in self.visibility


@dataclass(frozen=True)
class RoleDefinition:
    """Defines what a role can see, do, and must produce."""
    role: Role
    system_prompt: str
    visible_categories: frozenset[str]       # Context categories this role can see
    visible_roles: frozenset[Role]           # Can see output from these roles
    required_output: str                     # Description of expected handoff
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleHandoff:
    """The structured output from one role, passed to the next."""
    from_role: Role
    to_role: Role
    summary: str                             # Concise handoff for the next role
    artifacts: dict[str, str] = field(default_factory=dict)


class RoleBasedContextPartitioner:
    """Partitions context based on role definitions and visibility rules.

    Manages context items with role-based visibility, handles transitions
    between roles, and ensures each role sees only approved information.
    """

    def __init__(self, role_definitions: list[RoleDefinition]):
        self._roles: dict[Role, RoleDefinition] = {
            rd.role: rd for rd in role_definitions
        }
        self._context_items: list[ContextItem] = []
        self._handoffs: list[RoleHandoff] = []
        self._current_role: Role | None = None

    def add_shared_context(self, item_id: str, content: str, category: str) -> None:
        """Add context visible to all roles (e.g., requirements)."""
        self._context_items.append(
            ContextItem(
                item_id=item_id,
                content=content,
                source_role=None,
                category=category,
                visibility=frozenset(Role),
            )
        )

    def add_role_context(
        self,
        item_id: str,
        content: str,
        source_role: Role,
        category: str,
        visible_to: frozenset[Role] | None = None,
    ) -> None:
        """Add context produced by a specific role with visibility rules."""
        if visible_to is None:
            # Default: visible only to the source role
            visible_to = frozenset({source_role})

        self._context_items.append(
            ContextItem(
                item_id=item_id,
                content=content,
                source_role=source_role,
                category=category,
                visibility=visible_to,
            )
        )

    def get_context_for_role(self, role: Role) -> list[ContextItem]:
        """Return only the context items visible to the specified role."""
        role_def = self._roles.get(role)
        if role_def is None:
            return []

        visible_items = []
        for item in self._context_items:
            # Check direct visibility
            if not item.visible_to(role):
                continue
            # Check category visibility
            if item.category not in role_def.visible_categories:
                continue
            # Check source role visibility
            if (
                item.source_role is not None
                and item.source_role not in role_def.visible_roles
                and item.source_role != role
            ):
                continue
            visible_items.append(item)

        return visible_items

    def build_role_prompt(self, role: Role) -> str:
        """Build the complete prompt for a role, including filtered context."""
        role_def = self._roles.get(role)
        if role_def is None:
            raise ValueError(f"Unknown role: {role}")

        visible_items = self.get_context_for_role(role)
        relevant_handoffs = [
            h for h in self._handoffs if h.to_role == role
        ]

        sections = [role_def.system_prompt]

        if relevant_handoffs:
            handoff_sections = []
            for h in relevant_handoffs:
                handoff_sections.append(
                    f"### From {h.from_role.value}\n{h.summary}"
                )
            sections.append(
                "## Handoffs from Prior Roles\n\n"
                + "\n\n".join(handoff_sections)
            )

        if visible_items:
            grouped: dict[str, list[str]] = {}
            for item in visible_items:
                grouped.setdefault(item.category, []).append(item.content)

            for category, contents in grouped.items():
                sections.append(
                    f"## {category.title()}\n\n" + "\n\n".join(contents)
                )

        if role_def.constraints:
            constraints_str = "\n".join(f"- {c}" for c in role_def.constraints)
            sections.append(f"## Constraints\n{constraints_str}")

        sections.append(f"## Expected Output\n{role_def.required_output}")

        return "\n\n".join(sections)

    def transition(self, handoff: RoleHandoff) -> str:
        """Transition from one role to another via a structured handoff.

        Returns the built prompt for the new role.
        """
        self._handoffs.append(handoff)
        self._current_role = handoff.to_role
        return self.build_role_prompt(handoff.to_role)


# --- Role Definitions Example ---

FEATURE_WORKFLOW_ROLES = [
    RoleDefinition(
        role=Role.RESEARCHER,
        system_prompt=(
            "You are a technical researcher. Explore options and evaluate "
            "trade-offs. Do NOT write implementation code."
        ),
        visible_categories=frozenset({"requirement", "documentation", "research"}),
        visible_roles=frozenset(),
        required_output=(
            "Produce a Research Summary with 3-5 options, each with "
            "pros/cons and a recommendation."
        ),
        constraints=(
            "Do not write code",
            "Do not make architectural decisions",
            "Focus on gathering and organizing information",
        ),
    ),
    RoleDefinition(
        role=Role.ARCHITECT,
        system_prompt=(
            "You are a software architect. Make design decisions based on "
            "research findings. Produce an Architecture Decision Record."
        ),
        visible_categories=frozenset({"requirement", "documentation", "decision"}),
        visible_roles=frozenset({Role.RESEARCHER}),
        required_output=(
            "Produce an Architecture Decision Record (ADR) with: "
            "decision, rationale, consequences, and alternatives considered."
        ),
        constraints=(
            "Do not implement the solution",
            "Reference research findings, do not re-research",
        ),
    ),
    RoleDefinition(
        role=Role.IMPLEMENTER,
        system_prompt=(
            "You are a software engineer. Implement the architecture as specified. "
            "Follow the ADR exactly."
        ),
        visible_categories=frozenset({"requirement", "decision", "code"}),
        visible_roles=frozenset({Role.ARCHITECT}),
        required_output=(
            "Produce code changes with clear commit messages "
            "and brief implementation notes."
        ),
        constraints=(
            "Follow the ADR -- do not re-architect",
            "Do not evaluate alternatives -- that was the architect's job",
        ),
    ),
    RoleDefinition(
        role=Role.REVIEWER,
        system_prompt=(
            "You are a code reviewer. Evaluate the implementation against "
            "requirements and architectural decisions. Be objective."
        ),
        visible_categories=frozenset({"requirement", "decision", "code", "test"}),
        visible_roles=frozenset({Role.ARCHITECT, Role.IMPLEMENTER}),
        required_output=(
            "Produce a review verdict: APPROVE, REQUEST_CHANGES, or REJECT "
            "with specific action items."
        ),
        constraints=(
            "You did not write this code -- review it as a fresh reader",
            "Do not suggest alternative architectures",
            "Focus on correctness, readability, and test coverage",
        ),
    ),
]


# --- Usage Example ---

async def run_feature_workflow(llm_client, feature_request: str):
    """Run a feature through the role-partitioned workflow."""
    partitioner = RoleBasedContextPartitioner(FEATURE_WORKFLOW_ROLES)

    # Shared context visible to all roles
    partitioner.add_shared_context(
        item_id="req-1",
        content=feature_request,
        category="requirement",
    )

    # Step 1: Researcher -- sees only requirements
    researcher_prompt = partitioner.build_role_prompt(Role.RESEARCHER)
    research_result = await llm_client.complete(
        system_prompt=researcher_prompt,
        messages=[{"role": "user", "content": "Begin research."}],
    )

    # Step 2: Transition to Architect via structured handoff
    architect_prompt = partitioner.transition(
        RoleHandoff(
            from_role=Role.RESEARCHER,
            to_role=Role.ARCHITECT,
            summary=research_result,  # Only the summary, not the raw research
        )
    )
    architect_result = await llm_client.complete(
        system_prompt=architect_prompt,
        messages=[{"role": "user", "content": "Design the architecture."}],
    )

    # Step 3: Transition to Implementer
    implementer_prompt = partitioner.transition(
        RoleHandoff(
            from_role=Role.ARCHITECT,
            to_role=Role.IMPLEMENTER,
            summary=architect_result,
        )
    )
    impl_result = await llm_client.complete(
        system_prompt=implementer_prompt,
        messages=[{"role": "user", "content": "Implement the solution."}],
    )

    # Step 4: Reviewer sees code but NOT the deliberation process
    reviewer_prompt = partitioner.transition(
        RoleHandoff(
            from_role=Role.IMPLEMENTER,
            to_role=Role.REVIEWER,
            summary=impl_result,
        )
    )
    review_result = await llm_client.complete(
        system_prompt=reviewer_prompt,
        messages=[{"role": "user", "content": "Review the implementation."}],
    )

    return review_result
```

### TypeScript

```typescript
type Role = "researcher" | "architect" | "implementer" | "reviewer" | "tester";

interface ContextItem {
  readonly itemId: string;
  readonly content: string;
  readonly sourceRole: Role | null;
  readonly category: string;
  readonly visibility: ReadonlySet<Role>;
}

interface RoleDefinition {
  readonly role: Role;
  readonly systemPrompt: string;
  readonly visibleCategories: ReadonlySet<string>;
  readonly visibleRoles: ReadonlySet<Role>;
  readonly requiredOutput: string;
  readonly constraints: readonly string[];
}

interface RoleHandoff {
  readonly fromRole: Role;
  readonly toRole: Role;
  readonly summary: string;
  readonly artifacts: Readonly<Record<string, string>>;
}

interface LLMClient {
  complete(params: {
    systemPrompt: string;
    messages: readonly { role: string; content: string }[];
  }): Promise<string>;
}

const ALL_ROLES: ReadonlySet<Role> = new Set([
  "researcher",
  "architect",
  "implementer",
  "reviewer",
  "tester",
]);

function createContextItem(
  itemId: string,
  content: string,
  sourceRole: Role | null,
  category: string,
  visibility: ReadonlySet<Role> = ALL_ROLES
): ContextItem {
  return { itemId, content, sourceRole, category, visibility };
}

class RoleBasedContextPartitioner {
  private readonly roles: ReadonlyMap<Role, RoleDefinition>;
  private readonly contextItems: ContextItem[] = [];
  private readonly handoffs: RoleHandoff[] = [];

  constructor(roleDefinitions: readonly RoleDefinition[]) {
    this.roles = new Map(roleDefinitions.map((rd) => [rd.role, rd]));
  }

  addSharedContext(itemId: string, content: string, category: string): void {
    this.contextItems.push(
      createContextItem(itemId, content, null, category, ALL_ROLES)
    );
  }

  addRoleContext(
    itemId: string,
    content: string,
    sourceRole: Role,
    category: string,
    visibleTo?: ReadonlySet<Role>
  ): void {
    this.contextItems.push(
      createContextItem(
        itemId,
        content,
        sourceRole,
        category,
        visibleTo ?? new Set([sourceRole])
      )
    );
  }

  getContextForRole(role: Role): readonly ContextItem[] {
    const roleDef = this.roles.get(role);
    if (!roleDef) return [];

    return this.contextItems.filter((item) => {
      if (!item.visibility.has(role)) return false;
      if (!roleDef.visibleCategories.has(item.category)) return false;
      if (
        item.sourceRole !== null &&
        item.sourceRole !== role &&
        !roleDef.visibleRoles.has(item.sourceRole)
      ) {
        return false;
      }
      return true;
    });
  }

  buildRolePrompt(role: Role): string {
    const roleDef = this.roles.get(role);
    if (!roleDef) throw new Error(`Unknown role: ${role}`);

    const visibleItems = this.getContextForRole(role);
    const relevantHandoffs = this.handoffs.filter((h) => h.toRole === role);

    const sections: string[] = [roleDef.systemPrompt];

    if (relevantHandoffs.length > 0) {
      const handoffSections = relevantHandoffs.map(
        (h) => `### From ${h.fromRole}\n${h.summary}`
      );
      sections.push(
        `## Handoffs from Prior Roles\n\n${handoffSections.join("\n\n")}`
      );
    }

    if (visibleItems.length > 0) {
      const grouped = new Map<string, string[]>();
      for (const item of visibleItems) {
        const existing = grouped.get(item.category) ?? [];
        grouped.set(item.category, [...existing, item.content]);
      }

      for (const [category, contents] of grouped) {
        const title = category.charAt(0).toUpperCase() + category.slice(1);
        sections.push(`## ${title}\n\n${contents.join("\n\n")}`);
      }
    }

    if (roleDef.constraints.length > 0) {
      const constraints = roleDef.constraints.map((c) => `- ${c}`).join("\n");
      sections.push(`## Constraints\n${constraints}`);
    }

    sections.push(`## Expected Output\n${roleDef.requiredOutput}`);

    return sections.join("\n\n");
  }

  transition(handoff: RoleHandoff): string {
    this.handoffs.push(handoff);
    return this.buildRolePrompt(handoff.toRole);
  }
}

// --- Role Definitions Example ---

const FEATURE_WORKFLOW_ROLES: RoleDefinition[] = [
  {
    role: "researcher",
    systemPrompt:
      "You are a technical researcher. Explore options and evaluate trade-offs. Do NOT write implementation code.",
    visibleCategories: new Set(["requirement", "documentation", "research"]),
    visibleRoles: new Set(),
    requiredOutput:
      "Produce a Research Summary with 3-5 options, each with pros/cons and a recommendation.",
    constraints: [
      "Do not write code",
      "Do not make architectural decisions",
      "Focus on gathering and organizing information",
    ],
  },
  {
    role: "architect",
    systemPrompt:
      "You are a software architect. Make design decisions based on research findings.",
    visibleCategories: new Set(["requirement", "documentation", "decision"]),
    visibleRoles: new Set(["researcher"]),
    requiredOutput:
      "Produce an Architecture Decision Record with decision, rationale, consequences, and alternatives.",
    constraints: [
      "Do not implement the solution",
      "Reference research findings, do not re-research",
    ],
  },
  {
    role: "implementer",
    systemPrompt:
      "You are a software engineer. Implement the architecture as specified.",
    visibleCategories: new Set(["requirement", "decision", "code"]),
    visibleRoles: new Set(["architect"]),
    requiredOutput:
      "Produce code changes with clear commit messages and implementation notes.",
    constraints: [
      "Follow the ADR -- do not re-architect",
      "Do not evaluate alternatives",
    ],
  },
  {
    role: "reviewer",
    systemPrompt:
      "You are a code reviewer. Evaluate the implementation objectively.",
    visibleCategories: new Set(["requirement", "decision", "code", "test"]),
    visibleRoles: new Set(["architect", "implementer"]),
    requiredOutput:
      "Produce a review verdict: APPROVE, REQUEST_CHANGES, or REJECT with action items.",
    constraints: [
      "Review as a fresh reader",
      "Do not suggest alternative architectures",
      "Focus on correctness, readability, and test coverage",
    ],
  },
];

// --- Usage Example ---

async function runFeatureWorkflow(
  client: LLMClient,
  featureRequest: string
): Promise<string> {
  const partitioner = new RoleBasedContextPartitioner(FEATURE_WORKFLOW_ROLES);

  partitioner.addSharedContext("req-1", featureRequest, "requirement");

  // Step 1: Researcher
  const researcherPrompt = partitioner.buildRolePrompt("researcher");
  const researchResult = await client.complete({
    systemPrompt: researcherPrompt,
    messages: [{ role: "user", content: "Begin research." }],
  });

  // Step 2: Architect (sees research summary, not raw exploration)
  const architectPrompt = partitioner.transition({
    fromRole: "researcher",
    toRole: "architect",
    summary: researchResult,
    artifacts: {},
  });
  const architectResult = await client.complete({
    systemPrompt: architectPrompt,
    messages: [{ role: "user", content: "Design the architecture." }],
  });

  // Step 3: Implementer (sees ADR, not deliberation)
  const implementerPrompt = partitioner.transition({
    fromRole: "architect",
    toRole: "implementer",
    summary: architectResult,
    artifacts: {},
  });
  const implResult = await client.complete({
    systemPrompt: implementerPrompt,
    messages: [{ role: "user", content: "Implement the solution." }],
  });

  // Step 4: Reviewer (sees code, not process)
  const reviewerPrompt = partitioner.transition({
    fromRole: "implementer",
    toRole: "reviewer",
    summary: implResult,
    artifacts: {},
  });
  return client.complete({
    systemPrompt: reviewerPrompt,
    messages: [{ role: "user", content: "Review the implementation." }],
  });
}

export {
  RoleBasedContextPartitioner,
  RoleDefinition,
  RoleHandoff,
  ContextItem,
  Role,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Eliminates cross-role bias (reviewer does not see implementation rationale) | Handoff summarization can lose nuance from the source role |
| Each role operates with focused, minimal context | Requires upfront definition of roles and visibility rules |
| Prevents context bloat from accumulating irrelevant information | Role transitions add latency (new LLM call per role) |
| Enforces separation of concerns in agent workflows | Rigid role boundaries may miss valuable cross-cutting insights |
| Handoff contracts create clear interfaces between roles | More complex than a single-agent approach for simple tasks |

## When to Use

- Multi-step workflows where different phases require different perspectives (research, design, implement, review).
- When objectivity matters: code review, security audit, or quality assessment should not be biased by the creation process.
- Workflows where different roles have different trust levels or access requirements.
- When context from one phase actively harms performance in another (e.g., seeing too many options paralyzes the implementer).
- Team simulations where you want diverse perspectives, not a single blended viewpoint.

## When NOT to Use

- Simple, single-phase tasks where role separation adds unnecessary overhead.
- When tight feedback loops between roles are needed (e.g., pair programming where researcher and implementer iterate rapidly).
- When the workflow is highly linear and each step trivially follows from the last.
- When the cost of multiple LLM calls per workflow exceeds the benefit of role isolation.
- When the handoff contract cannot capture enough nuance (some tasks require the full deliberation context).

## Related Patterns

- **[Sub-Agent Delegation](sub-agent-delegation.md)** -- Each role can be implemented as a sub-agent. Role-based partitioning defines *what* each agent sees; sub-agent delegation defines *how* they execute.
- **[Sandbox Contexts](sandbox-contexts.md)** -- For risky operations within a role, sandbox the execution without affecting the role's context.
- **[Multi-Agent Context Orchestration](multi-agent-context-orchestration.md)** -- Orchestration manages the flow between agents; role-based partitioning defines the visibility rules within that flow.

## Real-World Examples

1. **Devin's multi-agent pipeline** -- Cognition's Devin separates planning, implementation, and verification into distinct agents with different context views. The verification agent does not see the planning deliberation, only the plan output and the code.

2. **Code review in CI/CD** -- Human code review follows this pattern naturally: reviewers see the diff and PR description, not the author's Slack conversations or abandoned approaches. Automated AI reviewers should be designed the same way.

3. **Legal document review** -- A drafting attorney and a reviewing attorney operate with different context. The reviewer intentionally does not see the negotiation history to catch issues the drafter might rationalize away.

4. **Red team / blue team exercises** -- Security red teams and blue teams operate with strictly partitioned context. The red team does not see defensive measures; the blue team does not see attack plans. This ensures genuine adversarial evaluation.
