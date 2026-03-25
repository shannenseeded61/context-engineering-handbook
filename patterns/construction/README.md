# Context Construction Patterns

Context construction is the practice of **building the right prompt content** before sending it to a language model. Rather than dumping raw text into a context window, construction patterns help you assemble structured, relevant, and token-efficient prompts that produce consistent results.

These patterns answer the question: *How do I compose what the model sees?*

## Decision Tree

```
Start here: What is your primary challenge?
|
|-- "My system prompt is a tangled mess that's hard to maintain"
|     --> System Prompt Architecture
|
|-- "I'm burning tokens on context the model doesn't need yet"
|     --> Progressive Disclosure
|
|-- "My few-shot examples are static and often irrelevant"
|     --> Few-Shot Curation
|
|-- "Different users or tasks need different agent personalities"
|     --> Dynamic Persona Assembly
|
|-- "The model's output format is inconsistent and breaks my parser"
|     --> Schema-Guided Generation
|
|-- "I'm copy-pasting the same prompt sections everywhere"
|     --> Template Composition
|
|-- "Different environments or user tiers need different safety rules"
|     --> Constraint Injection
|
|-- "I need structured prompts AND dynamic examples"
|     --> System Prompt Architecture + Few-Shot Curation
|
|-- "I need to manage a long multi-turn task with growing context"
|     --> Progressive Disclosure + System Prompt Architecture
|
|-- "I need personalized behavior with enforced compliance rules"
|     --> Dynamic Persona Assembly + Constraint Injection
|
|-- "I need reusable prompt templates with validated output"
|     --> Template Composition + Schema-Guided Generation
```

## Patterns

| Pattern | Summary | Key Benefit |
|---------|---------|-------------|
| [System Prompt Architecture](system-prompt-architecture.md) | Modular, composable system prompt sections | Maintainability and reuse |
| [Progressive Disclosure](progressive-disclosure.md) | Reveal context incrementally as needed | Token efficiency and focus |
| [Few-Shot Curation](few-shot-curation.md) | Dynamically select the best examples per task | Relevance and output quality |
| [Dynamic Persona Assembly](dynamic-persona-assembly.md) | Compose agent persona from trait modules at runtime | Personalization without duplication |
| [Schema-Guided Generation](schema-guided-generation.md) | Inject output schemas to constrain and validate format | Structured, parseable output |
| [Template Composition](template-composition.md) | Build prompts from reusable template fragments | Separation of content and logic |
| [Constraint Injection](constraint-injection.md) | Inject behavioral rules based on deployment context | Context-appropriate safety and compliance |

## How They Compose

These patterns work together naturally:

- **System Prompt Architecture** defines the skeleton of your prompt. It decides *what sections exist*.
- **Progressive Disclosure** decides *when each section is included*. Not every section needs to appear from the start.
- **Few-Shot Curation** populates the examples section dynamically. Instead of hardcoding examples into your system prompt template, you select them at runtime.
- **Dynamic Persona Assembly** populates the persona section based on who is asking and what they need. Different users get different identity blocks.
- **Schema-Guided Generation** fills the output format section with a formal schema, giving the model a precise contract to follow.
- **Template Composition** provides the rendering engine that assembles all of the above into a final string, using template inheritance, partials, and conditionals.
- **Constraint Injection** adds the appropriate behavioral boundaries based on environment, user tier, and compliance requirements.

A mature system often uses several of these together: a template engine (Template Composition) renders a modular prompt skeleton (System Prompt Architecture), where the persona is assembled from traits (Dynamic Persona Assembly), constraints are injected by deployment context (Constraint Injection), sections are conditionally included based on conversation state (Progressive Disclosure), the examples section is populated via embedding similarity (Few-Shot Curation), and the output format includes a formal schema (Schema-Guided Generation).
