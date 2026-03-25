# Dynamic Persona Assembly

> Compose the agent's persona at runtime by combining trait modules based on task, user, or domain.

## Problem

A static system prompt like "You are a helpful assistant" produces generic behavior regardless of who is asking or what they need. A financial analyst asking about tax implications gets the same tone and depth as a teenager asking about homework. You could maintain dozens of separate system prompts for each persona, but that leads to duplication, drift, and a maintenance nightmare. When the same safety rules need updating, you are patching twenty files instead of one.

## Solution

Decompose the persona into independent trait modules -- domain expertise, communication style, knowledge depth, behavioral constraints -- and compose them dynamically at runtime. A classifier inspects the incoming request (task type, user profile, deployment context) and selects the appropriate traits from a registry. The assembled persona feels coherent to the model because the traits are designed to be composable: each one addresses a distinct axis of behavior without contradicting the others.

This is the Strategy pattern applied to prompt identity. The core prompt structure remains stable while the persona adapts. A medical question from a clinician triggers the clinical-expert trait with formal tone; the same question from a patient triggers the health-educator trait with accessible language. Both share the same safety constraints and output format.

## How It Works

```
+------------------+     +---------------------+     +-------------------+
|  Incoming        |---->|  Task Classifier     |---->|  Trait Selector   |
|  Request         |     |                     |     |                   |
|  + User Profile  |     |  - Detect domain    |     |  - Pick expertise |
+------------------+     |  - Detect complexity|     |  - Pick tone      |
                         |  - Read user tier   |     |  - Pick depth     |
                         +---------------------+     |  - Pick constraints|
                                                      +--------+----------+
                                                               |
                                                               v
                                                      +-------------------+
                                                      |  Persona          |
                                                      |  Assembler        |
                                                      |                   |
                                                      |  Combines traits  |
                                                      |  into system      |
                                                      |  prompt string    |
                                                      +-------------------+

Trait axes (independent, composable):
+-----------------------------------------------------------+
| Domain     | finance, medical, legal, engineering, general |
| Tone       | formal, conversational, academic, empathetic  |
| Depth      | beginner, intermediate, expert                |
| Constraints| safety, compliance, brevity, verbosity        |
+-----------------------------------------------------------+
```

## Implementation

### Python

```python
from dataclasses import dataclass
from enum import Enum, auto


class Domain(Enum):
    FINANCE = auto()
    MEDICAL = auto()
    LEGAL = auto()
    ENGINEERING = auto()
    GENERAL = auto()


class Tone(Enum):
    FORMAL = auto()
    CONVERSATIONAL = auto()
    ACADEMIC = auto()
    EMPATHETIC = auto()


class Depth(Enum):
    BEGINNER = auto()
    INTERMEDIATE = auto()
    EXPERT = auto()


@dataclass(frozen=True)
class PersonaTrait:
    """A single composable trait contributing to the assembled persona."""
    axis: str
    key: str
    instruction: str
    priority: int = 0  # Lower = placed earlier in prompt


@dataclass(frozen=True)
class UserProfile:
    """Represents the requesting user's context."""
    expertise_level: Depth = Depth.INTERMEDIATE
    preferred_tone: Tone = Tone.CONVERSATIONAL
    domain: Domain = Domain.GENERAL
    compliance_tags: tuple[str, ...] = ()


# --- Trait Registry ---

DOMAIN_TRAITS: dict[Domain, PersonaTrait] = {
    Domain.FINANCE: PersonaTrait(
        axis="domain",
        key="finance",
        instruction=(
            "You are a financial analysis expert. Use precise financial "
            "terminology. Reference relevant regulations (SEC, GAAP, IFRS) "
            "when applicable. Always note when something is not financial advice."
        ),
        priority=0,
    ),
    Domain.MEDICAL: PersonaTrait(
        axis="domain",
        key="medical",
        instruction=(
            "You are a medical information specialist. Use evidence-based "
            "references. Distinguish between established consensus and emerging "
            "research. Always recommend consulting a healthcare professional "
            "for personal medical decisions."
        ),
        priority=0,
    ),
    Domain.LEGAL: PersonaTrait(
        axis="domain",
        key="legal",
        instruction=(
            "You are a legal research assistant. Cite jurisdictions when "
            "discussing laws. Distinguish between statutes, case law, and "
            "legal opinion. Always note that your output is not legal advice."
        ),
        priority=0,
    ),
    Domain.ENGINEERING: PersonaTrait(
        axis="domain",
        key="engineering",
        instruction=(
            "You are a senior software engineer. Provide production-quality "
            "code with error handling. Explain trade-offs between approaches. "
            "Reference official documentation when suggesting libraries."
        ),
        priority=0,
    ),
    Domain.GENERAL: PersonaTrait(
        axis="domain",
        key="general",
        instruction=(
            "You are a knowledgeable generalist. Provide clear, accurate "
            "information. When a topic requires domain expertise, note the "
            "limitations of general knowledge."
        ),
        priority=0,
    ),
}

TONE_TRAITS: dict[Tone, PersonaTrait] = {
    Tone.FORMAL: PersonaTrait(
        axis="tone",
        key="formal",
        instruction=(
            "Communication style: Use professional, formal language. "
            "Avoid colloquialisms and contractions. Structure responses "
            "with clear headings and numbered points."
        ),
        priority=1,
    ),
    Tone.CONVERSATIONAL: PersonaTrait(
        axis="tone",
        key="conversational",
        instruction=(
            "Communication style: Be friendly and approachable. Use clear, "
            "plain language. It is fine to use contractions and casual phrasing "
            "while maintaining accuracy."
        ),
        priority=1,
    ),
    Tone.ACADEMIC: PersonaTrait(
        axis="tone",
        key="academic",
        instruction=(
            "Communication style: Use academic register with precise "
            "terminology. Structure arguments logically with supporting "
            "evidence. Include caveats and qualifications where appropriate."
        ),
        priority=1,
    ),
    Tone.EMPATHETIC: PersonaTrait(
        axis="tone",
        key="empathetic",
        instruction=(
            "Communication style: Lead with empathy and understanding. "
            "Acknowledge the user's situation before providing information. "
            "Use supportive, encouraging language while remaining accurate."
        ),
        priority=1,
    ),
}

DEPTH_TRAITS: dict[Depth, PersonaTrait] = {
    Depth.BEGINNER: PersonaTrait(
        axis="depth",
        key="beginner",
        instruction=(
            "Knowledge depth: Explain concepts from first principles. "
            "Avoid jargon or define it immediately when used. Use analogies "
            "and simple examples. Do not assume prior knowledge."
        ),
        priority=2,
    ),
    Depth.INTERMEDIATE: PersonaTrait(
        axis="depth",
        key="intermediate",
        instruction=(
            "Knowledge depth: Assume working familiarity with core concepts. "
            "Explain advanced topics but skip basics. Use standard terminology "
            "without over-explaining."
        ),
        priority=2,
    ),
    Depth.EXPERT: PersonaTrait(
        axis="depth",
        key="expert",
        instruction=(
            "Knowledge depth: Assume deep expertise. Use precise technical "
            "language freely. Focus on nuance, edge cases, and trade-offs "
            "rather than introductions. Reference advanced concepts directly."
        ),
        priority=2,
    ),
}


@dataclass(frozen=True)
class AssembledPersona:
    """The final composed persona ready for injection into a system prompt."""
    traits: tuple[PersonaTrait, ...]
    system_prompt: str

    @property
    def trait_keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.traits)


class PersonaAssembler:
    """Composes persona traits into a coherent system prompt section.

    Selects traits based on user profile and task classification,
    then assembles them in priority order.
    """

    def __init__(
        self,
        domain_traits: dict[Domain, PersonaTrait] | None = None,
        tone_traits: dict[Tone, PersonaTrait] | None = None,
        depth_traits: dict[Depth, PersonaTrait] | None = None,
    ) -> None:
        self._domain_traits = domain_traits or DOMAIN_TRAITS
        self._tone_traits = tone_traits or TONE_TRAITS
        self._depth_traits = depth_traits or DEPTH_TRAITS

    def assemble(self, profile: UserProfile) -> AssembledPersona:
        """Build a persona from the user's profile."""
        traits = self._select_traits(profile)
        sorted_traits = sorted(traits, key=lambda t: t.priority)
        prompt = self._compose(sorted_traits)
        return AssembledPersona(traits=tuple(sorted_traits), system_prompt=prompt)

    def _select_traits(self, profile: UserProfile) -> list[PersonaTrait]:
        """Pick one trait per axis based on the user profile."""
        return [
            self._domain_traits[profile.domain],
            self._tone_traits[profile.preferred_tone],
            self._depth_traits[profile.expertise_level],
        ]

    def _compose(self, traits: list[PersonaTrait]) -> str:
        """Join trait instructions into a single system prompt block."""
        sections = [trait.instruction for trait in traits]
        return "\n\n".join(sections)


# --- Usage ---

assembler = PersonaAssembler()

# Clinician asking a medical question
clinician = UserProfile(
    expertise_level=Depth.EXPERT,
    preferred_tone=Tone.FORMAL,
    domain=Domain.MEDICAL,
)
clinician_persona = assembler.assemble(clinician)
print("=== Clinician Persona ===")
print(clinician_persona.system_prompt)
print(f"Traits: {clinician_persona.trait_keys}")

print()

# Student asking an engineering question
student = UserProfile(
    expertise_level=Depth.BEGINNER,
    preferred_tone=Tone.CONVERSATIONAL,
    domain=Domain.ENGINEERING,
)
student_persona = assembler.assemble(student)
print("=== Student Persona ===")
print(student_persona.system_prompt)
print(f"Traits: {student_persona.trait_keys}")
```

### TypeScript

```typescript
// --- Trait types ---

type Domain = "finance" | "medical" | "legal" | "engineering" | "general";
type Tone = "formal" | "conversational" | "academic" | "empathetic";
type Depth = "beginner" | "intermediate" | "expert";

interface PersonaTrait {
  readonly axis: string;
  readonly key: string;
  readonly instruction: string;
  readonly priority: number;
}

interface UserProfile {
  readonly expertiseLevel: Depth;
  readonly preferredTone: Tone;
  readonly domain: Domain;
  readonly complianceTags?: readonly string[];
}

interface AssembledPersona {
  readonly traits: readonly PersonaTrait[];
  readonly systemPrompt: string;
  readonly traitKeys: readonly string[];
}

// --- Trait registries ---

const DOMAIN_TRAITS: Record<Domain, PersonaTrait> = {
  finance: {
    axis: "domain",
    key: "finance",
    instruction:
      "You are a financial analysis expert. Use precise financial " +
      "terminology. Reference relevant regulations (SEC, GAAP, IFRS) " +
      "when applicable. Always note when something is not financial advice.",
    priority: 0,
  },
  medical: {
    axis: "domain",
    key: "medical",
    instruction:
      "You are a medical information specialist. Use evidence-based " +
      "references. Distinguish between established consensus and emerging " +
      "research. Always recommend consulting a healthcare professional " +
      "for personal medical decisions.",
    priority: 0,
  },
  legal: {
    axis: "domain",
    key: "legal",
    instruction:
      "You are a legal research assistant. Cite jurisdictions when " +
      "discussing laws. Distinguish between statutes, case law, and " +
      "legal opinion. Always note that your output is not legal advice.",
    priority: 0,
  },
  engineering: {
    axis: "domain",
    key: "engineering",
    instruction:
      "You are a senior software engineer. Provide production-quality " +
      "code with error handling. Explain trade-offs between approaches. " +
      "Reference official documentation when suggesting libraries.",
    priority: 0,
  },
  general: {
    axis: "domain",
    key: "general",
    instruction:
      "You are a knowledgeable generalist. Provide clear, accurate " +
      "information. When a topic requires domain expertise, note the " +
      "limitations of general knowledge.",
    priority: 0,
  },
};

const TONE_TRAITS: Record<Tone, PersonaTrait> = {
  formal: {
    axis: "tone",
    key: "formal",
    instruction:
      "Communication style: Use professional, formal language. " +
      "Avoid colloquialisms and contractions. Structure responses " +
      "with clear headings and numbered points.",
    priority: 1,
  },
  conversational: {
    axis: "tone",
    key: "conversational",
    instruction:
      "Communication style: Be friendly and approachable. Use clear, " +
      "plain language. It is fine to use contractions and casual phrasing " +
      "while maintaining accuracy.",
    priority: 1,
  },
  academic: {
    axis: "tone",
    key: "academic",
    instruction:
      "Communication style: Use academic register with precise " +
      "terminology. Structure arguments logically with supporting " +
      "evidence. Include caveats and qualifications where appropriate.",
    priority: 1,
  },
  empathetic: {
    axis: "tone",
    key: "empathetic",
    instruction:
      "Communication style: Lead with empathy and understanding. " +
      "Acknowledge the user's situation before providing information. " +
      "Use supportive, encouraging language while remaining accurate.",
    priority: 1,
  },
};

const DEPTH_TRAITS: Record<Depth, PersonaTrait> = {
  beginner: {
    axis: "depth",
    key: "beginner",
    instruction:
      "Knowledge depth: Explain concepts from first principles. " +
      "Avoid jargon or define it immediately when used. Use analogies " +
      "and simple examples. Do not assume prior knowledge.",
    priority: 2,
  },
  intermediate: {
    axis: "depth",
    key: "intermediate",
    instruction:
      "Knowledge depth: Assume working familiarity with core concepts. " +
      "Explain advanced topics but skip basics. Use standard terminology " +
      "without over-explaining.",
    priority: 2,
  },
  expert: {
    axis: "depth",
    key: "expert",
    instruction:
      "Knowledge depth: Assume deep expertise. Use precise technical " +
      "language freely. Focus on nuance, edge cases, and trade-offs " +
      "rather than introductions. Reference advanced concepts directly.",
    priority: 2,
  },
};

// --- Assembler ---

function selectTraits(profile: UserProfile): PersonaTrait[] {
  return [
    DOMAIN_TRAITS[profile.domain],
    TONE_TRAITS[profile.preferredTone],
    DEPTH_TRAITS[profile.expertiseLevel],
  ];
}

function assemblePersona(profile: UserProfile): AssembledPersona {
  const traits = selectTraits(profile).sort(
    (a, b) => a.priority - b.priority
  );

  const systemPrompt = traits
    .map((trait) => trait.instruction)
    .join("\n\n");

  return {
    traits,
    systemPrompt,
    traitKeys: traits.map((t) => t.key),
  };
}

// --- Usage ---

const clinician: UserProfile = {
  expertiseLevel: "expert",
  preferredTone: "formal",
  domain: "medical",
};

const clinicianPersona = assemblePersona(clinician);
console.log("=== Clinician Persona ===");
console.log(clinicianPersona.systemPrompt);
console.log("Traits:", clinicianPersona.traitKeys);

const student: UserProfile = {
  expertiseLevel: "beginner",
  preferredTone: "conversational",
  domain: "engineering",
};

const studentPersona = assemblePersona(student);
console.log("\n=== Student Persona ===");
console.log(studentPersona.systemPrompt);
console.log("Traits:", studentPersona.traitKeys);
```

## Trade-offs

| Pros | Cons |
|------|------|
| Same system supports diverse users without separate prompts | Trait combinations can produce incoherent personas if not tested |
| Adding a new domain is a registry entry, not a prompt rewrite | Combinatorial explosion: N domains x M tones x K depths = many possible personas |
| Traits are independently testable and versionable | Requires a classification step that can misroute requests |
| User preferences can be stored and reused across sessions | More moving parts than a single static system prompt |
| A/B testing individual trait axes is straightforward | Interactions between traits need careful design to avoid contradiction |

## When to Use

- Your application serves multiple user segments (developers, executives, students, clinicians) with different needs
- You want to personalize the agent's behavior without maintaining separate prompt files per persona
- The same agent handles multiple domains and needs to shift expertise dynamically
- You are building a multi-tenant platform where each tenant configures their agent's personality
- You want to A/B test individual persona axes (tone, depth) independently

## When NOT to Use

- Your agent has a single fixed identity (e.g., a brand mascot with a defined voice)
- The user base is homogeneous and one persona fits all
- You are in early prototyping and the persona is still being discovered
- The overhead of classifying the task and selecting traits exceeds the benefit (simple, single-purpose bots)

## Related Patterns

- [System Prompt Architecture](system-prompt-architecture.md) -- provides the structural skeleton into which the assembled persona is injected
- [Constraint Injection](constraint-injection.md) -- the constraints axis of persona assembly can be handled by a dedicated constraint injection system
- [Progressive Disclosure](progressive-disclosure.md) -- persona traits can be revealed progressively as the conversation evolves

## Real-World Examples

- **Character.AI** allows users to create characters with tunable personality traits (tone, expertise, background) that compose into a system prompt governing the character's behavior.
- **Intercom's Fin AI agent** adapts its communication style based on the business's brand voice settings and the customer's tier, effectively assembling a persona per conversation.
- **GitHub Copilot Chat** shifts persona between "explain this code" (educator mode), "fix this bug" (debugger mode), and "write tests" (test engineer mode) based on the user's slash command -- each activating different trait combinations.
- **OpenAI's custom GPTs** let builders configure instructions, conversation style, and knowledge sources, which the platform composes into a runtime persona per session.
