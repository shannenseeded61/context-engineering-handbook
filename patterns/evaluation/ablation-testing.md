# Ablation Testing

> Systematically remove context components to measure their individual contribution to output quality, identifying bloated or redundant context that can be safely removed.

## Problem

Context windows fill up over time as teams add more instructions, examples, retrieval results, and tool definitions. But nobody measures whether each component actually helps:

- **Accumulated instructions**: The system prompt has grown to 3,000 tokens over months of incremental additions. Some sections were added for edge cases that no longer occur. Nobody knows which instructions are load-bearing.
- **Few-shot example bloat**: Six examples were added during development, but perhaps two would suffice. Each example costs 200-400 tokens.
- **Retrieval over-fetching**: The RAG system returns 10 documents by default. Maybe 3 would produce the same quality output while saving 70% of context tokens.
- **Redundant tool schemas**: Twenty tool definitions are loaded, but 80% of tasks use only 5 tools. The unused schemas waste 3,000+ tokens.

Without measurement, context grows monotonically. Teams are afraid to remove anything because they do not know what will break. The result is context bloat: a window full of marginally useful tokens that crowd out high-value content and increase costs.

## Solution

Run ablation studies on your context: systematically remove one component at a time (or one group at a time), run the resulting context against a test suite, and measure the impact on output quality. Components that can be removed without significant quality loss are candidates for elimination or conditional loading.

The process mirrors scientific ablation studies:

1. **Baseline**: Run the full context against a test suite and record quality scores.
2. **Single ablation**: Remove each component individually, re-run the test suite, and record the quality delta.
3. **Contribution scoring**: Components whose removal causes large quality drops are high-contribution. Components whose removal causes no quality change are candidates for removal.
4. **Group ablation** (optional): Test removing combinations of low-contribution components together to check for interaction effects.

## How It Works

```
Full context (baseline):
  [System Prompt][Few-Shot x6][Tools x20][Retrieval x10][History]
  Score: 92/100

Ablation runs:
  Run 1: Remove system prompt section "tone guidelines"
         [System(-tone)][Few-Shot x6][Tools x20][Retrieval x10][History]
         Score: 91/100  -->  contribution: 1 point

  Run 2: Remove few-shot examples 4-6
         [System][Few-Shot x3][Tools x20][Retrieval x10][History]
         Score: 90/100  -->  contribution: 2 points

  Run 3: Remove tools 6-20 (rarely used)
         [System][Few-Shot x6][Tools x5][Retrieval x10][History]
         Score: 91/100  -->  contribution: 1 point

  Run 4: Reduce retrieval from 10 to 3 documents
         [System][Few-Shot x6][Tools x20][Retrieval x3][History]
         Score: 88/100  -->  contribution: 4 points

  Run 5: Remove error handling instructions
         [System(-errors)][Few-Shot x6][Tools x20][Retrieval x10][History]
         Score: 78/100  -->  contribution: 14 points  [HIGH]

Contribution ranking:
  1. Error handling instructions  --> 14 pts  (KEEP)
  2. Retrieval docs 4-10          --> 4 pts   (KEEP, consider reducing)
  3. Few-shot examples 4-6        --> 2 pts   (REMOVE)
  4. Tone guidelines              --> 1 pt    (REMOVE)
  5. Tools 6-20                   --> 1 pt    (LOAD CONDITIONALLY)

Optimized context: 40% fewer tokens, 2-point quality drop (92 --> 90)
```

## Implementation

### Python

```python
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass(frozen=True)
class ContextComponent:
    """A named, removable component of the context."""
    name: str
    content: str
    category: str  # e.g., "system", "examples", "tools", "retrieval"
    token_estimate: int = 0

    def __post_init__(self):
        if self.token_estimate == 0:
            object.__setattr__(
                self, "token_estimate", len(self.content) // 4
            )


@dataclass(frozen=True)
class TestCase:
    """A test case for evaluating context quality."""
    name: str
    query: str
    expected_behavior: str
    evaluation_fn: Callable[[str], float]  # Takes response, returns 0-1 score
    weight: float = 1.0


@dataclass(frozen=True)
class AblationResult:
    """Result of ablating (removing) a single component."""
    component_name: str
    baseline_score: float
    ablated_score: float
    contribution: float  # baseline - ablated
    tokens_freed: int
    efficiency: float  # contribution per 1000 tokens
    test_details: dict = field(default_factory=dict)

    @property
    def is_removable(self) -> bool:
        """Component can likely be removed if contribution is minimal."""
        return self.contribution < 0.02  # Less than 2% impact


@dataclass(frozen=True)
class AblationReport:
    """Complete ablation study results."""
    baseline_score: float
    results: tuple[AblationResult, ...]
    total_tokens: int
    timestamp: float

    def ranked_by_contribution(self) -> list[AblationResult]:
        """Return results sorted by contribution (highest first)."""
        return sorted(self.results, key=lambda r: -r.contribution)

    def removable_components(
        self, threshold: float = 0.02
    ) -> list[AblationResult]:
        """Components whose removal causes less than threshold quality loss."""
        return [r for r in self.results if r.contribution < threshold]

    def potential_savings(self, threshold: float = 0.02) -> dict[str, Any]:
        """Calculate tokens and cost savings from removing low-value components."""
        removable = self.removable_components(threshold)
        tokens_freed = sum(r.tokens_freed for r in removable)
        quality_loss = sum(r.contribution for r in removable)

        return {
            "removable_components": [r.component_name for r in removable],
            "tokens_freed": tokens_freed,
            "token_reduction_pct": (
                (tokens_freed / self.total_tokens * 100)
                if self.total_tokens > 0 else 0
            ),
            "estimated_quality_loss": quality_loss,
            "baseline_score": self.baseline_score,
            "projected_score": self.baseline_score - quality_loss,
        }

    def to_summary(self) -> str:
        """Human-readable summary of the ablation study."""
        lines = [
            "# Ablation Study Results",
            f"**Baseline Score**: {self.baseline_score:.1%}",
            f"**Total Tokens**: {self.total_tokens:,}",
            "",
            "## Component Contributions (ranked)",
            "",
            "| Component | Contribution | Tokens | Efficiency | Action |",
            "|-----------|-------------|--------|------------|--------|",
        ]

        for r in self.ranked_by_contribution():
            action = "REMOVE" if r.is_removable else "KEEP"
            lines.append(
                f"| {r.component_name} | {r.contribution:.1%} | "
                f"{r.tokens_freed:,} | {r.efficiency:.4f} | {action} |"
            )

        savings = self.potential_savings()
        lines.extend([
            "",
            "## Optimization Opportunity",
            f"- Removable components: {len(savings['removable_components'])}",
            f"- Tokens freed: {savings['tokens_freed']:,} "
            f"({savings['token_reduction_pct']:.1f}%)",
            f"- Projected score: {savings['projected_score']:.1%} "
            f"(from {savings['baseline_score']:.1%})",
        ])

        return "\n".join(lines)


class AblationTester:
    """
    Runs ablation studies on context components to measure their
    individual contribution to output quality.

    Systematically removes components, runs a test suite against
    each variant, and reports contribution scores.
    """

    def __init__(
        self,
        run_fn: Callable[[str, str], Awaitable[str]],
        test_cases: list[TestCase],
    ):
        """
        Args:
            run_fn: Async function that takes (context, query) and returns
                the LLM response string. This is the function under test.
            test_cases: Test suite to evaluate output quality.
        """
        self._run_fn = run_fn
        self._test_cases = test_cases
        self._report_history: list[AblationReport] = []

    async def run_study(
        self,
        components: list[ContextComponent],
        ablation_groups: dict[str, list[str]] | None = None,
    ) -> AblationReport:
        """
        Run a complete ablation study.

        Args:
            components: All context components to test.
            ablation_groups: Optional dict mapping group names to lists
                of component names. Tests removing entire groups at once.

        Returns:
            AblationReport with per-component contribution scores.
        """
        # Build full context
        full_context = self._assemble_context(components)
        total_tokens = sum(c.token_estimate for c in components)

        # Run baseline
        baseline_score, baseline_details = await self._evaluate(full_context)

        # Run single-component ablations
        results = []
        for component in components:
            remaining = [c for c in components if c.name != component.name]
            ablated_context = self._assemble_context(remaining)

            ablated_score, test_details = await self._evaluate(ablated_context)
            contribution = max(0.0, baseline_score - ablated_score)

            tokens_freed = component.token_estimate
            efficiency = (
                (contribution / (tokens_freed / 1000))
                if tokens_freed > 0 else 0.0
            )

            results.append(AblationResult(
                component_name=component.name,
                baseline_score=baseline_score,
                ablated_score=ablated_score,
                contribution=contribution,
                tokens_freed=tokens_freed,
                efficiency=efficiency,
                test_details=test_details,
            ))

        # Run group ablations if specified
        if ablation_groups:
            for group_name, member_names in ablation_groups.items():
                remaining = [
                    c for c in components if c.name not in member_names
                ]
                ablated_context = self._assemble_context(remaining)

                ablated_score, test_details = await self._evaluate(
                    ablated_context
                )
                contribution = max(0.0, baseline_score - ablated_score)
                tokens_freed = sum(
                    c.token_estimate for c in components
                    if c.name in member_names
                )
                efficiency = (
                    (contribution / (tokens_freed / 1000))
                    if tokens_freed > 0 else 0.0
                )

                results.append(AblationResult(
                    component_name=f"[group] {group_name}",
                    baseline_score=baseline_score,
                    ablated_score=ablated_score,
                    contribution=contribution,
                    tokens_freed=tokens_freed,
                    efficiency=efficiency,
                    test_details=test_details,
                ))

        report = AblationReport(
            baseline_score=baseline_score,
            results=tuple(results),
            total_tokens=total_tokens,
            timestamp=time.time(),
        )

        self._report_history = [*self._report_history, report]
        return report

    async def _evaluate(
        self, context: str
    ) -> tuple[float, dict[str, float]]:
        """Run all test cases against a context variant and return scores."""
        details: dict[str, float] = {}
        total_weight = 0.0
        weighted_score = 0.0

        for test in self._test_cases:
            response = await self._run_fn(context, test.query)
            score = test.evaluation_fn(response)
            details[test.name] = score
            weighted_score += score * test.weight
            total_weight += test.weight

        overall = weighted_score / total_weight if total_weight > 0 else 0.0
        return overall, details

    @staticmethod
    def _assemble_context(components: list[ContextComponent]) -> str:
        """Assemble components into a single context string."""
        return "\n\n".join(c.content for c in components if c.content)

    @property
    def report_history(self) -> list[AblationReport]:
        return list(self._report_history)
```

### TypeScript

```typescript
interface ContextComponent {
  readonly name: string;
  readonly content: string;
  readonly category: string;
  readonly tokenEstimate: number;
}

interface TestCase {
  readonly name: string;
  readonly query: string;
  readonly expectedBehavior: string;
  readonly evaluationFn: (response: string) => number;
  readonly weight: number;
}

interface AblationResult {
  readonly componentName: string;
  readonly baselineScore: number;
  readonly ablatedScore: number;
  readonly contribution: number;
  readonly tokensFreed: number;
  readonly efficiency: number;
  readonly testDetails: Readonly<Record<string, number>>;
}

interface AblationReport {
  readonly baselineScore: number;
  readonly results: readonly AblationResult[];
  readonly totalTokens: number;
  readonly timestamp: number;
}

interface SavingsProjection {
  readonly removableComponents: readonly string[];
  readonly tokensFreed: number;
  readonly tokenReductionPct: number;
  readonly estimatedQualityLoss: number;
  readonly baselineScore: number;
  readonly projectedScore: number;
}

function createComponent(params: {
  name: string;
  content: string;
  category: string;
  tokenEstimate?: number;
}): ContextComponent {
  return {
    name: params.name,
    content: params.content,
    category: params.category,
    tokenEstimate: params.tokenEstimate ?? Math.floor(params.content.length / 4),
  };
}

function createTestCase(params: {
  name: string;
  query: string;
  expectedBehavior: string;
  evaluationFn: (response: string) => number;
  weight?: number;
}): TestCase {
  return {
    name: params.name,
    query: params.query,
    expectedBehavior: params.expectedBehavior,
    evaluationFn: params.evaluationFn,
    weight: params.weight ?? 1.0,
  };
}

function rankedByContribution(report: AblationReport): readonly AblationResult[] {
  return [...report.results].sort((a, b) => b.contribution - a.contribution);
}

function removableComponents(
  report: AblationReport,
  threshold = 0.02
): readonly AblationResult[] {
  return report.results.filter((r) => r.contribution < threshold);
}

function potentialSavings(
  report: AblationReport,
  threshold = 0.02
): SavingsProjection {
  const removable = removableComponents(report, threshold);
  const tokensFreed = removable.reduce((sum, r) => sum + r.tokensFreed, 0);
  const qualityLoss = removable.reduce((sum, r) => sum + r.contribution, 0);

  return {
    removableComponents: Object.freeze(
      removable.map((r) => r.componentName)
    ),
    tokensFreed,
    tokenReductionPct:
      report.totalTokens > 0
        ? (tokensFreed / report.totalTokens) * 100
        : 0,
    estimatedQualityLoss: qualityLoss,
    baselineScore: report.baselineScore,
    projectedScore: report.baselineScore - qualityLoss,
  };
}

function reportToSummary(report: AblationReport): string {
  const lines = [
    "# Ablation Study Results",
    `**Baseline Score**: ${Math.round(report.baselineScore * 100)}%`,
    `**Total Tokens**: ${report.totalTokens.toLocaleString()}`,
    "",
    "## Component Contributions (ranked)",
    "",
    "| Component | Contribution | Tokens | Efficiency | Action |",
    "|-----------|-------------|--------|------------|--------|",
  ];

  for (const r of rankedByContribution(report)) {
    const action = r.contribution < 0.02 ? "REMOVE" : "KEEP";
    lines.push(
      `| ${r.componentName} | ${Math.round(r.contribution * 100)}% | ` +
        `${r.tokensFreed.toLocaleString()} | ${r.efficiency.toFixed(4)} | ${action} |`
    );
  }

  const savings = potentialSavings(report);
  lines.push(
    "",
    "## Optimization Opportunity",
    `- Removable components: ${savings.removableComponents.length}`,
    `- Tokens freed: ${savings.tokensFreed.toLocaleString()} (${savings.tokenReductionPct.toFixed(1)}%)`,
    `- Projected score: ${Math.round(savings.projectedScore * 100)}% (from ${Math.round(savings.baselineScore * 100)}%)`
  );

  return lines.join("\n");
}

type RunFn = (context: string, query: string) => Promise<string>;

class AblationTester {
  private readonly runFn: RunFn;
  private readonly testCases: readonly TestCase[];
  private reportHistory: readonly AblationReport[] = [];

  constructor(runFn: RunFn, testCases: TestCase[]) {
    this.runFn = runFn;
    this.testCases = Object.freeze([...testCases]);
  }

  async runStudy(
    components: ContextComponent[],
    ablationGroups?: Record<string, string[]>
  ): Promise<AblationReport> {
    const fullContext = this.assembleContext(components);
    const totalTokens = components.reduce(
      (sum, c) => sum + c.tokenEstimate,
      0
    );

    // Baseline
    const { overall: baselineScore } = await this.evaluate(fullContext);

    // Single-component ablations
    const results: AblationResult[] = [];

    for (const component of components) {
      const remaining = components.filter((c) => c.name !== component.name);
      const ablatedContext = this.assembleContext(remaining);

      const { overall: ablatedScore, details } =
        await this.evaluate(ablatedContext);
      const contribution = Math.max(0, baselineScore - ablatedScore);
      const tokensFreed = component.tokenEstimate;
      const efficiency =
        tokensFreed > 0 ? contribution / (tokensFreed / 1000) : 0;

      results.push({
        componentName: component.name,
        baselineScore,
        ablatedScore,
        contribution,
        tokensFreed,
        efficiency,
        testDetails: Object.freeze(details),
      });
    }

    // Group ablations
    if (ablationGroups) {
      for (const [groupName, memberNames] of Object.entries(ablationGroups)) {
        const remaining = components.filter(
          (c) => !memberNames.includes(c.name)
        );
        const ablatedContext = this.assembleContext(remaining);

        const { overall: ablatedScore, details } =
          await this.evaluate(ablatedContext);
        const contribution = Math.max(0, baselineScore - ablatedScore);
        const tokensFreed = components
          .filter((c) => memberNames.includes(c.name))
          .reduce((sum, c) => sum + c.tokenEstimate, 0);
        const efficiency =
          tokensFreed > 0 ? contribution / (tokensFreed / 1000) : 0;

        results.push({
          componentName: `[group] ${groupName}`,
          baselineScore,
          ablatedScore,
          contribution,
          tokensFreed,
          efficiency,
          testDetails: Object.freeze(details),
        });
      }
    }

    const report: AblationReport = {
      baselineScore,
      results: Object.freeze(results),
      totalTokens,
      timestamp: Date.now() / 1000,
    };

    this.reportHistory = Object.freeze([...this.reportHistory, report]);
    return report;
  }

  private async evaluate(
    context: string
  ): Promise<{ overall: number; details: Record<string, number> }> {
    const details: Record<string, number> = {};
    let totalWeight = 0;
    let weightedScore = 0;

    for (const test of this.testCases) {
      const response = await this.runFn(context, test.query);
      const score = test.evaluationFn(response);
      details[test.name] = score;
      weightedScore += score * test.weight;
      totalWeight += test.weight;
    }

    return {
      overall: totalWeight > 0 ? weightedScore / totalWeight : 0,
      details,
    };
  }

  private assembleContext(components: ContextComponent[]): string {
    return components
      .filter((c) => c.content)
      .map((c) => c.content)
      .join("\n\n");
  }

  get history(): readonly AblationReport[] {
    return this.reportHistory;
  }
}

export {
  AblationTester,
  AblationReport,
  AblationResult,
  ContextComponent,
  TestCase,
  createComponent,
  createTestCase,
  rankedByContribution,
  removableComponents,
  potentialSavings,
  reportToSummary,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Data-driven decisions about what to keep in context (no more guessing) | Requires a representative test suite -- results are only as good as the tests |
| Identifies significant savings opportunities (30-50% token reduction is common) | Expensive to run: each ablation requires a full test suite execution against the LLM |
| Contribution scores are intuitive and actionable | Single-component ablation misses interaction effects (removing A and B together may be worse than removing either alone) |
| Efficiency metric (contribution per 1000 tokens) enables cost-benefit analysis | Non-deterministic LLM outputs add noise to scores -- multiple runs needed for confidence |
| Group ablation catches synergistic components | Results may not generalize to queries outside the test suite |
| Report format supports both automated and human decision-making | Point-in-time analysis -- context contribution can change as the product evolves |

## When to Use

- Context windows are approaching capacity and you need to free up space
- LLM costs are high and you suspect context is bloated with low-value content
- Before major prompt refactoring to understand which sections are load-bearing
- Periodically (quarterly) as a context hygiene practice
- When adding new context components, to verify they actually improve output quality
- After migrating to a smaller/cheaper model, to identify what context the smaller model needs most

## When NOT to Use

- Context is small and well-understood (under 2,000 tokens)
- You do not have a representative test suite with quality evaluation functions
- When LLM API costs for running the study exceed the potential savings
- During active development when context is changing rapidly (wait for stability)
- For one-off prompts that will not be reused

## Related Patterns

- **Context Coverage Analysis** (Evaluation): Coverage analysis checks if context is sufficient for a specific query. Ablation testing checks if context components are necessary for general quality. They answer opposite questions.
- **KV-Cache Optimization** (Optimization): Ablation results inform which components should be in the stable prefix (high-contribution) vs. conditionally loaded (low-contribution).
- **Context Rot Detection** (Evaluation): Rot detection flags degraded context in real-time. Ablation testing is an offline analysis that informs what to include in context proactively.
- **Prompt Caching Strategies** (Optimization): Components identified as high-contribution and stable are prime candidates for long-TTL caching.

## Real-World Examples

- **Anthropic Prompt Engineering**: Anthropic's documentation recommends testing prompt components individually to understand their contribution. Ablation testing formalizes this into a repeatable process.
- **OpenAI Evals Framework**: The evals framework supports running test suites against prompt variants, which is the infrastructure needed for systematic ablation studies.
- **Production RAG Tuning**: Teams routinely ablation-test the number of retrieved documents (k in top-k retrieval). Common finding: reducing k from 10 to 3-5 maintains quality while saving 50-70% of retrieval context tokens.
- **Few-Shot Example Optimization**: Research consistently shows that carefully selected 2-3 examples often outperform 6-8 random examples. Ablation testing identifies which specific examples are contributing and which are noise.
- **System Prompt Minimization**: Enterprise teams report reducing system prompts by 40-60% after ablation studies reveal that many accumulated instructions have zero measurable impact on output quality.
