# Schema-Guided Generation

> Provide output schemas as context to constrain the model's format and enable automatic validation.

## Problem

When you ask a language model for structured output -- JSON, YAML, a specific data shape -- without a formal schema, the model improvises. Field names drift between responses (`user_name` vs `userName` vs `name`). Optional fields appear inconsistently. Nested structures vary in depth. Downstream code that parses these responses breaks unpredictably. You end up writing brittle regex extractors or defensive parsing code that handles every variation the model might produce, rather than telling the model exactly what shape to produce.

## Solution

Include the output schema directly in the context -- as JSON Schema, a Pydantic model definition, a Zod schema, or a TypeScript interface -- along with one or two examples of valid output. The schema serves dual purpose: it documents the expected format for the model (acting as a precise specification in the prompt), and it provides a validation contract for your code (the same schema that appears in the prompt is used to parse and validate the response). When the model sees a formal schema, it adheres to it far more reliably than when given prose descriptions of the desired format.

Modern APIs (OpenAI's structured outputs, Anthropic's tool use) support schema-constrained decoding natively. This pattern works both with those APIs and with plain text completions where you validate after the fact.

## How It Works

```
+------------------+     +---------------------+     +------------------+
|  Define Schema   |---->|  Build Prompt       |---->|  Call Model      |
|                  |     |                     |     |                  |
|  Pydantic / Zod  |     |  System: schema +   |     |  Response text   |
|  JSON Schema     |     |  instructions       |     |  or structured   |
+------------------+     |  User: actual query  |     +--------+---------+
                         +---------------------+              |
                                                               v
                                                      +------------------+
                                                      |  Validate        |
                                                      |                  |
                                                      |  Parse response  |
                                                      |  against schema  |
                                                      |                  |
                                                      |  Valid? Use it.  |
                                                      |  Invalid? Retry  |
                                                      |  or fallback.    |
                                                      +------------------+

Schema placement in prompt:
+------------------------------------------------------+
| System prompt:                                       |
|   "You must respond with valid JSON matching this    |
|    schema: { ... }                                   |
|    Example valid response: { ... }"                  |
|                                                      |
| User message:                                        |
|   "Analyze this customer review: ..."                |
+------------------------------------------------------+
```

## Implementation

### Python

```python
import json
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError


# --- Define output schemas as Pydantic models ---

class SentimentResult(BaseModel):
    """Schema for sentiment analysis output."""
    sentiment: str = Field(
        description="One of: positive, negative, neutral, mixed"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0 and 1"
    )
    key_phrases: list[str] = Field(
        description="Up to 5 phrases that drove the sentiment classification"
    )
    reasoning: str = Field(
        description="One sentence explaining the classification"
    )


class ExtractionResult(BaseModel):
    """Schema for entity extraction output."""
    entities: list["ExtractedEntity"] = Field(
        description="All entities found in the text"
    )
    summary: str = Field(
        description="One sentence summary of the text"
    )


class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity's canonical name")
    entity_type: str = Field(
        description="One of: person, organization, location, date, monetary_value"
    )
    mentions: int = Field(ge=1, description="Number of times mentioned")


ExtractionResult.model_rebuild()

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class SchemaPromptConfig:
    """Configuration for schema-guided prompt generation."""
    task_instruction: str
    example_input: str
    example_output: dict[str, Any]
    max_retries: int = 2


class SchemaGuidedPrompter:
    """Injects schema definitions into prompts and validates responses.

    The same Pydantic model used for validation is serialized into the
    prompt so the model sees the exact contract it must fulfill.
    """

    def build_system_prompt(
        self,
        schema_class: type[BaseModel],
        config: SchemaPromptConfig,
    ) -> str:
        """Build a system prompt that includes the schema and an example."""
        schema_json = json.dumps(
            schema_class.model_json_schema(), indent=2
        )
        example_json = json.dumps(config.example_output, indent=2)

        return (
            f"{config.task_instruction}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_json}\n```\n\n"
            f"Example:\n"
            f"Input: {config.example_input}\n"
            f"Output:\n```json\n{example_json}\n```\n\n"
            f"Rules:\n"
            f"- Output ONLY the JSON object, no surrounding text.\n"
            f"- Every field in the schema is required.\n"
            f"- Follow the field descriptions and constraints exactly."
        )

    def parse_response(
        self,
        response_text: str,
        schema_class: type[T],
    ) -> T:
        """Parse and validate the model's response against the schema.

        Strips markdown code fences if present, then validates with Pydantic.
        Raises ValidationError if the response does not match.
        """
        cleaned = self._extract_json(response_text)
        parsed = json.loads(cleaned)
        return schema_class.model_validate(parsed)

    def parse_with_retry(
        self,
        call_model: callable,
        user_message: str,
        schema_class: type[T],
        config: SchemaPromptConfig,
    ) -> T:
        """Call the model and retry on validation failure.

        On failure, the validation error is fed back to the model
        as context for the retry attempt.
        """
        system_prompt = self.build_system_prompt(schema_class, config)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            response_text = call_model(messages)
            try:
                return self.parse_response(response_text, schema_class)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                # Feed the error back for the retry
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response failed validation:\n{exc}\n\n"
                        f"Please fix the JSON and respond again."
                    ),
                })

        raise ValueError(
            f"Failed to get valid response after {config.max_retries + 1} "
            f"attempts. Last error: {last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences from a JSON response."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```json) and last line (```)
            json_lines = [
                line for line in lines[1:]
                if not line.strip().startswith("```")
            ]
            return "\n".join(json_lines)
        return stripped


# --- Usage ---

prompter = SchemaGuidedPrompter()

sentiment_config = SchemaPromptConfig(
    task_instruction=(
        "You are a sentiment analysis engine. Analyze the sentiment "
        "of the provided text."
    ),
    example_input="The product arrived on time and works perfectly!",
    example_output={
        "sentiment": "positive",
        "confidence": 0.95,
        "key_phrases": ["arrived on time", "works perfectly"],
        "reasoning": "Strong positive language about delivery and product quality.",
    },
)

# Build the system prompt
system_prompt = prompter.build_system_prompt(SentimentResult, sentiment_config)
print(system_prompt)

# Validate a mock response
mock_response = json.dumps({
    "sentiment": "negative",
    "confidence": 0.87,
    "key_phrases": ["broke after two days", "waste of money"],
    "reasoning": "Negative experience with product durability and value.",
})

result = prompter.parse_response(mock_response, SentimentResult)
print(f"\nParsed: {result}")
print(f"Confidence: {result.confidence}")
```

### TypeScript

```typescript
import { z } from "zod";

// --- Define output schemas with Zod ---

const SentimentResultSchema = z.object({
  sentiment: z.enum(["positive", "negative", "neutral", "mixed"]),
  confidence: z.number().min(0).max(1),
  key_phrases: z
    .array(z.string())
    .max(5)
    .describe("Up to 5 phrases that drove the classification"),
  reasoning: z
    .string()
    .describe("One sentence explaining the classification"),
});

type SentimentResult = z.infer<typeof SentimentResultSchema>;

const ExtractedEntitySchema = z.object({
  name: z.string().describe("The entity's canonical name"),
  entity_type: z.enum([
    "person",
    "organization",
    "location",
    "date",
    "monetary_value",
  ]),
  mentions: z.number().int().min(1),
});

const ExtractionResultSchema = z.object({
  entities: z.array(ExtractedEntitySchema),
  summary: z.string().describe("One sentence summary of the text"),
});

type ExtractionResult = z.infer<typeof ExtractionResultSchema>;

// --- Schema-guided prompter ---

interface SchemaPromptConfig {
  readonly taskInstruction: string;
  readonly exampleInput: string;
  readonly exampleOutput: Record<string, unknown>;
  readonly maxRetries: number;
}

type ModelCaller = (
  messages: Array<{ role: string; content: string }>
) => Promise<string>;

function zodSchemaToJsonDescription(schema: z.ZodTypeAny): string {
  /**
   * Convert a Zod schema to a human-readable JSON Schema string.
   * In production, use zod-to-json-schema for full fidelity.
   * This simplified version extracts the shape for prompt injection.
   */
  if (schema instanceof z.ZodObject) {
    const shape = schema.shape;
    const fields: Record<string, string> = {};
    for (const [key, value] of Object.entries(shape)) {
      const zodValue = value as z.ZodTypeAny;
      fields[key] = zodValue.description ?? zodValue.constructor.name;
    }
    return JSON.stringify(fields, null, 2);
  }
  return "{}";
}

function buildSystemPrompt(
  schema: z.ZodTypeAny,
  config: SchemaPromptConfig
): string {
  const schemaDescription = zodSchemaToJsonDescription(schema);
  const exampleJson = JSON.stringify(config.exampleOutput, null, 2);

  return [
    config.taskInstruction,
    "",
    "You MUST respond with valid JSON matching this schema:",
    "```json",
    schemaDescription,
    "```",
    "",
    "Example:",
    `Input: ${config.exampleInput}`,
    "Output:",
    "```json",
    exampleJson,
    "```",
    "",
    "Rules:",
    "- Output ONLY the JSON object, no surrounding text.",
    "- Every field in the schema is required.",
    "- Follow the field descriptions and constraints exactly.",
  ].join("\n");
}

function extractJson(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith("```")) {
    const lines = trimmed.split("\n");
    const jsonLines = lines
      .slice(1)
      .filter((line) => !line.trim().startsWith("```"));
    return jsonLines.join("\n");
  }
  return trimmed;
}

function parseResponse<T>(responseText: string, schema: z.ZodSchema<T>): T {
  const cleaned = extractJson(responseText);
  const parsed = JSON.parse(cleaned);
  return schema.parse(parsed);
}

async function parseWithRetry<T>(
  callModel: ModelCaller,
  userMessage: string,
  schema: z.ZodSchema<T>,
  zodSchema: z.ZodTypeAny,
  config: SchemaPromptConfig
): Promise<T> {
  const systemPrompt = buildSystemPrompt(zodSchema, config);
  const messages: Array<{ role: string; content: string }> = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userMessage },
  ];

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    const responseText = await callModel(messages);
    try {
      return parseResponse(responseText, schema);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      messages.push({ role: "assistant", content: responseText });
      messages.push({
        role: "user",
        content:
          `Your response failed validation:\n${lastError.message}\n\n` +
          `Please fix the JSON and respond again.`,
      });
    }
  }

  throw new Error(
    `Failed to get valid response after ${config.maxRetries + 1} ` +
      `attempts. Last error: ${lastError?.message}`
  );
}

// --- Usage ---

const sentimentConfig: SchemaPromptConfig = {
  taskInstruction:
    "You are a sentiment analysis engine. Analyze the sentiment " +
    "of the provided text.",
  exampleInput: "The product arrived on time and works perfectly!",
  exampleOutput: {
    sentiment: "positive",
    confidence: 0.95,
    key_phrases: ["arrived on time", "works perfectly"],
    reasoning:
      "Strong positive language about delivery and product quality.",
  },
  maxRetries: 2,
};

// Build the system prompt
const systemPrompt = buildSystemPrompt(
  SentimentResultSchema,
  sentimentConfig
);
console.log(systemPrompt);

// Validate a mock response
const mockResponse = JSON.stringify({
  sentiment: "negative",
  confidence: 0.87,
  key_phrases: ["broke after two days", "waste of money"],
  reasoning: "Negative experience with product durability and value.",
});

const result = parseResponse(mockResponse, SentimentResultSchema);
console.log("\nParsed:", result);
console.log("Confidence:", result.confidence);
```

## Trade-offs

| Pros | Cons |
|------|------|
| Eliminates format drift between responses | Schema definition in the prompt consumes tokens |
| Same schema validates both prompt and response | Complex nested schemas can confuse smaller models |
| Retry loop with error feedback self-corrects most failures | Retry adds latency and cost for invalid responses |
| Type-safe parsed output in application code | Requires schema maintenance as output requirements evolve |
| Compatible with native structured output APIs | Models may hallucinate valid-looking but factually wrong data that passes validation |

## When to Use

- Your application parses model output programmatically (API responses, database records, structured reports)
- You need consistent field names, types, and shapes across thousands of requests
- Downstream systems break on format variations and you need a validation contract
- You are using tool/function calling and need precise argument schemas
- Multiple models or providers must produce the same output format

## When NOT to Use

- The output is free-form text (essays, conversations, creative writing) with no structural requirements
- You are prototyping and the output format changes every iteration
- The schema is trivially simple (a single boolean or number) and does not warrant the overhead
- You are using a native structured output API that handles schema enforcement at the decoding level (the prompt-side injection becomes redundant)

## Related Patterns

- [System Prompt Architecture](system-prompt-architecture.md) -- the schema definition is one section in a modular system prompt
- [Few-Shot Curation](few-shot-curation.md) -- example outputs in the schema prompt are effectively few-shot examples and can be curated dynamically
- [Template Composition](template-composition.md) -- schema definitions can be injected into prompts via template variables

## Real-World Examples

- **OpenAI Structured Outputs** (`response_format: { type: "json_schema", json_schema: ... }`) enforces schema compliance at the decoding level, making schema-guided generation a first-class API feature.
- **Anthropic tool use** requires JSON Schema definitions for tool inputs, which the model must follow exactly -- the same principle applied to function calling.
- **Instructor library** (Python) wraps OpenAI/Anthropic calls with Pydantic models, automatically injecting schemas into prompts and validating responses with retry logic -- a direct implementation of this pattern.
- **LangChain's `with_structured_output()`** accepts Pydantic models or JSON Schema and handles schema injection, parsing, and retry across multiple LLM providers.
- **Marvin** (by Prefect) uses Pydantic models as the interface for AI functions, generating prompts from type annotations and docstrings automatically.
