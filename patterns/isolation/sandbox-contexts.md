# Sandbox Contexts

> Create disposable, isolated context environments for risky or exploratory operations that are destroyed after use, preventing context leakage back to the parent.

## Problem

Agents frequently need to perform risky or exploratory operations: executing untrusted code, browsing unknown websites, testing hypotheses, or evaluating multiple approaches. If these operations happen in the main context, they introduce several problems:

- **Context pollution**: Failed experiments, error tracebacks, and irrelevant exploration artifacts accumulate in the parent context.
- **Security risk**: Executing untrusted code or processing untrusted content in the main context could produce prompt injections or unintended side effects.
- **Irreversibility**: Once information enters the context window, it cannot be removed -- it influences all subsequent reasoning.
- **Hypothesis bias**: When testing multiple approaches, earlier attempts bias the evaluation of later ones because the model has already "seen" them.

Without sandboxing, agents must either avoid risky operations entirely or accept permanent context contamination.

## Solution

Create a **ContextSandbox** -- a disposable execution environment that clones minimal parent context, runs operations in complete isolation, and returns only explicitly approved results. The sandbox has its own context window, can execute tools, and is destroyed after use. Nothing from the sandbox enters the parent context unless it passes through an explicit extraction gate.

The key insight: the sandbox is not just about isolation (sub-agent delegation does that). It is about **disposability and safety**. The sandbox can be tainted, corrupted, or crashed without consequence. It is designed to be thrown away.

## How It Works

```
Parent Agent Context
+--------------------------------------------------+
| System prompt + user request                      |
| "Evaluate this code snippet the user pasted"      |
|                                                   |
| [Creating sandbox...]                             |
|                                                   |
|   Sandbox (disposable, isolated)                  |
|   +------------------------------------------+   |
|   | Cloned subset of parent context:          |   |
|   |   - Task description                      |   |
|   |   - Relevant variables                    |   |
|   |                                           |   |
|   | Executes in isolation:                    |   |
|   |   - Runs untrusted code                   |   |
|   |   - Captures stdout, stderr               |   |  These NEVER leak
|   |   - Encounters error / prompt injection    |   |  to the parent
|   |   - Tries recovery, fails                  |   |
|   |   - 500 lines of messy output             |   |
|   +------------------------------------------+   |
|   | Extraction gate:                          |   |
|   |   - Filters: only approved result types   |   |
|   |   - Sanitizes: strips injection attempts  |   |
|   |   - Summarizes: condenses raw output      |   |
|   +------------------------------------------+   |
|   [Sandbox destroyed]                             |
|                                                   |
| Receives: "Code executes successfully. Output:    |
|   fibonacci(10) = 55. No side effects detected."  |
|                                                   |
| Continues with clean context...                   |
+--------------------------------------------------+
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import asyncio
import uuid


class SandboxStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DESTROYED = "destroyed"


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for a sandbox execution environment."""
    max_tokens: int = 30_000
    timeout_seconds: float = 60.0
    allowed_tools: tuple[str, ...] = ()
    blocked_patterns: tuple[str, ...] = (
        "ignore previous instructions",
        "system prompt",
        "you are now",
    )
    max_output_chars: int = 5_000
    extract_format: str = "summary"  # "summary", "raw", "structured"


@dataclass(frozen=True)
class SandboxResult:
    """The extracted, sanitized result from a sandbox execution."""
    sandbox_id: str
    status: SandboxStatus
    extracted_output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxContext:
    """Minimal context cloned from parent into the sandbox."""
    task_description: str
    variables: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()

    def to_system_prompt(self) -> str:
        sections = [
            "You are operating inside a disposable sandbox environment.",
            "Your output will be filtered before reaching the parent agent.",
            "Do not attempt to communicate outside the sandbox.",
            f"\n## Task\n{self.task_description}",
        ]
        if self.variables:
            vars_str = "\n".join(f"- {k}: {v}" for k, v in self.variables.items())
            sections.append(f"\n## Available Variables\n{vars_str}")
        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
            sections.append(f"\n## Constraints\n{constraints_str}")
        return "\n".join(sections)


class ContextSandbox:
    """A disposable, isolated execution environment.

    Creates a fresh context window, executes operations in isolation,
    and returns only approved, sanitized results. The sandbox is
    destroyed after use regardless of outcome.
    """

    def __init__(
        self,
        llm_client,
        config: SandboxConfig | None = None,
        output_sanitizer: Callable[[str], str] | None = None,
    ):
        self._llm_client = llm_client
        self._config = config or SandboxConfig()
        self._sanitizer = output_sanitizer or self._default_sanitizer
        self._status = SandboxStatus.CREATED

    async def execute(
        self, sandbox_ctx: SandboxContext
    ) -> SandboxResult:
        """Execute a task inside the sandbox and return sanitized results.

        The sandbox is destroyed after this call, regardless of success or failure.
        """
        sandbox_id = uuid.uuid4().hex[:12]
        warnings: list[str] = []

        try:
            self._status = SandboxStatus.RUNNING
            system_prompt = sandbox_ctx.to_system_prompt()

            raw_output = await asyncio.wait_for(
                self._llm_client.complete(
                    system_prompt=system_prompt,
                    messages=[{"role": "user", "content": "Execute the task."}],
                    max_tokens=self._config.max_tokens,
                    tools=list(self._config.allowed_tools),
                ),
                timeout=self._config.timeout_seconds,
            )

            # Check for injection attempts in the output
            injection_detected = self._detect_injection(raw_output)
            if injection_detected:
                warnings.append(
                    "Potential prompt injection detected in sandbox output. "
                    "Output was sanitized."
                )

            # Sanitize and truncate
            sanitized = self._sanitizer(raw_output)
            if len(sanitized) > self._config.max_output_chars:
                sanitized = sanitized[: self._config.max_output_chars] + "\n[truncated]"
                warnings.append("Output exceeded max length and was truncated.")

            # Extract based on configured format
            extracted = self._extract_output(sanitized, self._config.extract_format)

            self._status = SandboxStatus.COMPLETED
            return SandboxResult(
                sandbox_id=sandbox_id,
                status=SandboxStatus.COMPLETED,
                extracted_output=extracted,
                metadata={"tokens_used": len(raw_output.split()) * 2},
                warnings=tuple(warnings),
            )

        except asyncio.TimeoutError:
            self._status = SandboxStatus.FAILED
            return SandboxResult(
                sandbox_id=sandbox_id,
                status=SandboxStatus.FAILED,
                extracted_output="",
                metadata={"error": "Sandbox execution timed out"},
                warnings=("Execution exceeded timeout.",),
            )
        except Exception as e:
            self._status = SandboxStatus.FAILED
            return SandboxResult(
                sandbox_id=sandbox_id,
                status=SandboxStatus.FAILED,
                extracted_output="",
                metadata={"error": str(e)},
                warnings=(f"Sandbox execution failed: {type(e).__name__}",),
            )
        finally:
            # Sandbox is always destroyed -- no state persists
            self._status = SandboxStatus.DESTROYED

    def _detect_injection(self, text: str) -> bool:
        """Check for common prompt injection patterns in sandbox output."""
        lower_text = text.lower()
        return any(
            pattern in lower_text for pattern in self._config.blocked_patterns
        )

    def _default_sanitizer(self, text: str) -> str:
        """Remove potentially dangerous patterns from sandbox output."""
        sanitized = text
        for pattern in self._config.blocked_patterns:
            sanitized = sanitized.replace(pattern, "[FILTERED]")
        return sanitized

    def _extract_output(self, sanitized: str, fmt: str) -> str:
        """Extract the relevant portion of sandbox output."""
        if fmt == "raw":
            return sanitized
        if fmt == "summary":
            # Take the last meaningful section as the conclusion
            lines = [l for l in sanitized.strip().split("\n") if l.strip()]
            if len(lines) <= 5:
                return sanitized.strip()
            return "\n".join(lines[-5:])
        return sanitized


class SandboxManager:
    """Manages multiple sandbox executions with lifecycle tracking."""

    def __init__(self, llm_client):
        self._llm_client = llm_client
        self._history: list[SandboxResult] = []

    async def run_sandboxed(
        self,
        task: str,
        variables: dict[str, Any] | None = None,
        config: SandboxConfig | None = None,
    ) -> SandboxResult:
        """Create a sandbox, execute a task, destroy it, return the result."""
        sandbox = ContextSandbox(self._llm_client, config=config)
        ctx = SandboxContext(
            task_description=task,
            variables=variables or {},
        )
        result = await sandbox.execute(ctx)
        self._history.append(result)
        return result

    async def run_competing_sandboxes(
        self,
        task: str,
        approaches: list[dict[str, Any]],
        config: SandboxConfig | None = None,
    ) -> list[SandboxResult]:
        """Run the same task with different approaches in parallel sandboxes.

        Each sandbox is fully isolated, so earlier approaches cannot
        bias the evaluation of later ones.
        """
        tasks = [
            self.run_sandboxed(
                task=f"{task}\n\nApproach: {approach.get('description', '')}",
                variables=approach.get("variables", {}),
                config=config,
            )
            for approach in approaches
        ]
        return await asyncio.gather(*tasks)

    @property
    def history(self) -> list[SandboxResult]:
        return list(self._history)


# --- Usage Example ---

async def evaluate_untrusted_code(llm_client, code_snippet: str):
    """Evaluate user-provided code in a sandbox without polluting main context."""
    manager = SandboxManager(llm_client)

    result = await manager.run_sandboxed(
        task=(
            "Execute the following code snippet and report:\n"
            "1. Does it run without errors?\n"
            "2. What is the output?\n"
            "3. Are there any security concerns?\n\n"
            f"```python\n{code_snippet}\n```"
        ),
        config=SandboxConfig(
            timeout_seconds=30.0,
            allowed_tools=("code_interpreter",),
            max_output_chars=2_000,
            extract_format="summary",
        ),
    )

    if result.status == SandboxStatus.COMPLETED:
        print(f"Sandbox result: {result.extracted_output}")
    else:
        print(f"Sandbox failed: {result.metadata.get('error')}")

    # The main context never saw the raw code execution output,
    # error tracebacks, or any prompt injection attempts
    if result.warnings:
        print(f"Warnings: {', '.join(result.warnings)}")
```

### TypeScript

```typescript
interface SandboxConfig {
  readonly maxTokens: number;
  readonly timeoutMs: number;
  readonly allowedTools: readonly string[];
  readonly blockedPatterns: readonly string[];
  readonly maxOutputChars: number;
  readonly extractFormat: "summary" | "raw" | "structured";
}

interface SandboxResult {
  readonly sandboxId: string;
  readonly status: "completed" | "failed" | "destroyed";
  readonly extractedOutput: string;
  readonly metadata: Record<string, unknown>;
  readonly warnings: readonly string[];
}

interface SandboxContext {
  readonly taskDescription: string;
  readonly variables: Readonly<Record<string, unknown>>;
  readonly constraints: readonly string[];
}

interface LLMClient {
  complete(params: {
    systemPrompt: string;
    messages: readonly { role: string; content: string }[];
    maxTokens: number;
    tools?: readonly string[];
  }): Promise<string>;
}

const DEFAULT_CONFIG: SandboxConfig = {
  maxTokens: 30_000,
  timeoutMs: 60_000,
  allowedTools: [],
  blockedPatterns: [
    "ignore previous instructions",
    "system prompt",
    "you are now",
  ],
  maxOutputChars: 5_000,
  extractFormat: "summary",
};

function buildSandboxPrompt(ctx: SandboxContext): string {
  const sections = [
    "You are operating inside a disposable sandbox environment.",
    "Your output will be filtered before reaching the parent agent.",
    "Do not attempt to communicate outside the sandbox.",
    `\n## Task\n${ctx.taskDescription}`,
  ];

  const varEntries = Object.entries(ctx.variables);
  if (varEntries.length > 0) {
    const vars = varEntries.map(([k, v]) => `- ${k}: ${v}`).join("\n");
    sections.push(`\n## Available Variables\n${vars}`);
  }

  if (ctx.constraints.length > 0) {
    const constraints = ctx.constraints.map((c) => `- ${c}`).join("\n");
    sections.push(`\n## Constraints\n${constraints}`);
  }

  return sections.join("\n");
}

function detectInjection(
  text: string,
  blockedPatterns: readonly string[]
): boolean {
  const lowerText = text.toLowerCase();
  return blockedPatterns.some((pattern) => lowerText.includes(pattern));
}

function sanitizeOutput(
  text: string,
  blockedPatterns: readonly string[]
): string {
  let sanitized = text;
  for (const pattern of blockedPatterns) {
    sanitized = sanitized.replaceAll(pattern, "[FILTERED]");
  }
  return sanitized;
}

function extractOutput(sanitized: string, format: string): string {
  if (format === "raw") return sanitized;
  if (format === "summary") {
    const lines = sanitized
      .trim()
      .split("\n")
      .filter((l) => l.trim());
    if (lines.length <= 5) return sanitized.trim();
    return lines.slice(-5).join("\n");
  }
  return sanitized;
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Sandbox timed out")), ms);
    promise.then(
      (val) => { clearTimeout(timer); resolve(val); },
      (err) => { clearTimeout(timer); reject(err); }
    );
  });
}

async function executeSandbox(
  client: LLMClient,
  ctx: SandboxContext,
  config: SandboxConfig = DEFAULT_CONFIG
): Promise<SandboxResult> {
  const sandboxId = crypto.randomUUID().slice(0, 12);
  const warnings: string[] = [];

  try {
    const systemPrompt = buildSandboxPrompt(ctx);

    const rawOutput = await withTimeout(
      client.complete({
        systemPrompt,
        messages: [{ role: "user", content: "Execute the task." }],
        maxTokens: config.maxTokens,
        tools: [...config.allowedTools],
      }),
      config.timeoutMs
    );

    if (detectInjection(rawOutput, config.blockedPatterns)) {
      warnings.push(
        "Potential prompt injection detected in sandbox output. Output was sanitized."
      );
    }

    let sanitized = sanitizeOutput(rawOutput, config.blockedPatterns);
    if (sanitized.length > config.maxOutputChars) {
      sanitized = sanitized.slice(0, config.maxOutputChars) + "\n[truncated]";
      warnings.push("Output exceeded max length and was truncated.");
    }

    const extracted = extractOutput(sanitized, config.extractFormat);

    return {
      sandboxId,
      status: "completed",
      extractedOutput: extracted,
      metadata: { tokensUsed: Math.ceil(rawOutput.split(/\s+/).length * 1.5) },
      warnings,
    };
  } catch (err) {
    return {
      sandboxId,
      status: "failed",
      extractedOutput: "",
      metadata: { error: err instanceof Error ? err.message : String(err) },
      warnings: [
        `Sandbox execution failed: ${err instanceof Error ? err.constructor.name : "Unknown"}`,
      ],
    };
  }
  // Sandbox is implicitly destroyed -- no state survives this function
}

async function runCompetingSandboxes(
  client: LLMClient,
  task: string,
  approaches: Array<{
    description: string;
    variables?: Record<string, unknown>;
  }>,
  config?: SandboxConfig
): Promise<readonly SandboxResult[]> {
  return Promise.all(
    approaches.map((approach) =>
      executeSandbox(
        client,
        {
          taskDescription: `${task}\n\nApproach: ${approach.description}`,
          variables: approach.variables ?? {},
          constraints: [],
        },
        config
      )
    )
  );
}

// --- Usage Example ---

async function evaluateUntrustedCode(
  client: LLMClient,
  codeSnippet: string
): Promise<void> {
  const result = await executeSandbox(
    client,
    {
      taskDescription: [
        "Execute the following code snippet and report:",
        "1. Does it run without errors?",
        "2. What is the output?",
        "3. Are there any security concerns?",
        "",
        "```python",
        codeSnippet,
        "```",
      ].join("\n"),
      variables: {},
      constraints: [],
    },
    { ...DEFAULT_CONFIG, timeoutMs: 30_000, extractFormat: "summary" }
  );

  if (result.status === "completed") {
    console.log(`Sandbox result: ${result.extractedOutput}`);
  } else {
    console.log(`Sandbox failed: ${result.metadata.error}`);
  }

  if (result.warnings.length > 0) {
    console.log(`Warnings: ${result.warnings.join(", ")}`);
  }
}

export { executeSandbox, runCompetingSandboxes, SandboxConfig, SandboxResult };
```

## Trade-offs

| Pros | Cons |
|------|------|
| Complete context isolation -- parent is never contaminated | Additional LLM call per sandbox execution (cost) |
| Safe execution of untrusted content and code | Results must pass through extraction gate, losing detail |
| Disposability eliminates cleanup concerns | Sandbox cannot ask clarifying questions of the user |
| Parallel competing sandboxes eliminate hypothesis bias | Cold start penalty: sandbox must rebuild minimal context |
| Failed sandboxes have zero impact on parent | Output sanitization may produce false positives, filtering benign content |

## When to Use

- Executing or evaluating untrusted code snippets from users or external sources.
- Testing multiple competing hypotheses where earlier attempts should not bias later ones.
- Processing content from untrusted URLs or documents that may contain prompt injections.
- Exploratory operations where you want to "try and discard" without consequence.
- Any operation where a failure or unexpected output could corrupt the parent agent's reasoning.

## When NOT to Use

- When the operation is simple and trusted -- the overhead of a sandbox is not justified.
- When the child needs ongoing interaction with the user (sandboxes are fire-and-forget).
- When the full intermediate reasoning is valuable to the parent (use sub-agent delegation with rich result extraction instead).
- When latency is critical and the sandbox round-trip is too slow.
- When you need the sandbox state to persist across calls (sandboxes are single-use by design).

## Related Patterns

- **[Sub-Agent Delegation](sub-agent-delegation.md)** -- Sandboxes are a specialized form of sub-agent delegation, optimized for disposability and safety rather than result richness.
- **[Multi-Agent Context Orchestration](multi-agent-context-orchestration.md)** -- For persistent, collaborating agents rather than disposable execution environments.
- **[Role-Based Context Partitioning](role-based-context-partitioning.md)** -- Partitions context by role within a single workflow, complementary to sandboxing risky operations.

## Real-World Examples

1. **ChatGPT Code Interpreter** -- User code executes in a sandboxed container. The model receives stdout/stderr summaries, never the raw execution environment state. If the container crashes, it is replaced with a fresh one.

2. **GitHub Copilot Workspace** -- When evaluating proposed changes, each approach runs in an isolated environment. Failed builds in one approach do not affect evaluation of others.

3. **Browser-use agents** -- Agents that browse the web create sandboxed contexts for each page visit, extracting only relevant information and discarding the full page content (which could contain prompt injections).

4. **LLM red-teaming frameworks** -- Tools like garak run adversarial prompts in sandboxed contexts, ensuring that successful attacks against one sandbox do not propagate to the evaluation harness.
