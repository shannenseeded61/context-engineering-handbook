# Context Coverage Analysis

> Analyze whether the context window contains all the information needed to answer the current query, identifying gaps before the model is forced to hallucinate or guess.

## Problem

LLMs do not say "I do not have enough information in my context to answer this." Instead, they hallucinate confidently. When critical context is missing, the failure mode is invisible:

- **Missing retrieval results**: The RAG system returned documents about a related but wrong topic. The model answers using the wrong documents rather than flagging the gap.
- **Incomplete tool output**: A tool call returned partial results (paginated API, truncated output). The model treats partial data as complete.
- **Absent domain knowledge**: The query requires information that was never loaded into context (e.g., the user asks about a config file that was not retrieved). The model invents plausible-sounding configuration.
- **Stale references**: The context contains an old version of a document. The model answers based on outdated information without indicating it might be wrong.

The root cause is that standard LLM pipelines have no mechanism to compare *what the query needs* against *what the context provides*. The model receives context and a query, and it always produces an answer -- whether or not the context supports one.

## Solution

Decompose each query into discrete **information requirements** -- the specific pieces of knowledge needed to produce a correct answer. Then check each requirement against the available context. If coverage falls below a threshold, the system can trigger additional retrieval, ask for clarification, or explicitly caveat the response.

This works as a pre-flight check before the main LLM call:

1. **Requirement extraction**: Analyze the query to identify what information is needed (entities, relationships, facts, code references).
2. **Coverage mapping**: For each requirement, search the context for supporting evidence.
3. **Gap identification**: Requirements without supporting context are flagged as gaps.
4. **Action routing**: Based on coverage score, either proceed, trigger retrieval for gaps, or ask the user for clarification.

## How It Works

```
User query: "Update the database migration to add an index on users.email"
                    |
                    v
+--------------------------------------+
| Requirement Extraction               |
| 1. Current migration file content    |
| 2. users table schema                |
| 3. Database type (Postgres/MySQL)    |
| 4. Migration framework conventions   |
| 5. Existing indexes on users table   |
+--------------------------------------+
                    |
                    v
+--------------------------------------+
| Coverage Check Against Context       |
|                                      |
| 1. Migration file    [FOUND]  score: 1.0   -- file loaded in context
| 2. Table schema      [FOUND]  score: 0.8   -- partial schema available
| 3. Database type     [FOUND]  score: 1.0   -- mentioned in config
| 4. Framework conv.   [GAP]    score: 0.1   -- not in context
| 5. Existing indexes  [GAP]    score: 0.0   -- no index info available
+--------------------------------------+
                    |
                    v
+--------------------------------------+
| Coverage Score: 58% (below 80%)      |
| Action: RETRIEVE_MORE               |
|                                      |
| Gaps to fill:                        |
| - Load migration framework docs     |
| - Query existing indexes on users   |
+--------------------------------------+
                    |
          +---------+---------+
          |                   |
          v                   v
   Trigger retrieval    Or: Proceed with
   for missing gaps     explicit caveats
```

## Implementation

### Python

```python
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable


class CoverageLevel(Enum):
    FULL = "full"           # All requirements met
    ADEQUATE = "adequate"   # Most requirements met, gaps are non-critical
    PARTIAL = "partial"     # Significant gaps exist
    INSUFFICIENT = "insufficient"  # Critical gaps, should not proceed


class CoverageAction(Enum):
    PROCEED = "proceed"
    RETRIEVE_MORE = "retrieve_more"
    ASK_CLARIFICATION = "ask_clarification"
    CAVEAT_RESPONSE = "caveat_response"


@dataclass(frozen=True)
class InformationRequirement:
    """A discrete piece of information needed to answer a query."""
    description: str
    category: str       # e.g., "code", "schema", "config", "docs"
    critical: bool      # If True, absence should block the response
    search_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageResult:
    """Coverage check result for a single requirement."""
    requirement: InformationRequirement
    score: float        # 0.0 to 1.0
    evidence: str       # The matching context excerpt, if found
    source: str         # Which context section contained the evidence


@dataclass(frozen=True)
class CoverageReport:
    """Full coverage analysis for a query."""
    query: str
    results: tuple[CoverageResult, ...]
    overall_score: float
    level: CoverageLevel
    action: CoverageAction
    gaps: tuple[InformationRequirement, ...]
    critical_gaps: tuple[InformationRequirement, ...]

    def to_context_block(self) -> str:
        """Format the report for injection into the LLM context."""
        lines = [
            "## Context Coverage Analysis",
            f"**Coverage**: {self.overall_score:.0%} ({self.level.value})",
            f"**Action**: {self.action.value}",
            "",
        ]

        if self.gaps:
            lines.append("**Information gaps** (not found in context):")
            for gap in self.gaps:
                critical_tag = " [CRITICAL]" if gap.critical else ""
                lines.append(f"- {gap.description}{critical_tag}")

        if self.critical_gaps:
            lines.append("")
            lines.append(
                "WARNING: Critical information is missing. "
                "Response may be inaccurate."
            )

        return "\n".join(lines)


class RequirementExtractor:
    """
    Extracts information requirements from a query.

    Uses pattern-based heuristics by default. Can be enhanced with
    an LLM-based extractor for more nuanced requirement detection.
    """

    def __init__(
        self,
        domain_patterns: dict[str, list[str]] | None = None,
        llm_extractor: Callable[[str], Awaitable[list[dict]]] | None = None,
    ):
        """
        Args:
            domain_patterns: Dict mapping categories to regex patterns
                that indicate a requirement in that category.
            llm_extractor: Optional async function that uses an LLM to
                extract requirements. Takes query, returns list of dicts
                with keys: description, category, critical, search_terms.
        """
        self._patterns = domain_patterns or self._default_patterns()
        self._llm_extractor = llm_extractor

    def extract(self, query: str) -> list[InformationRequirement]:
        """Extract information requirements from a query using patterns."""
        requirements = []

        for category, patterns in self._patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    term = match if isinstance(match, str) else match[0]
                    requirements.append(InformationRequirement(
                        description=f"{category}: {term}",
                        category=category,
                        critical=category in ("code", "schema"),
                        search_terms=(term.lower(),),
                    ))

        return requirements

    async def extract_with_llm(self, query: str) -> list[InformationRequirement]:
        """Use an LLM to extract nuanced requirements."""
        if self._llm_extractor is None:
            return self.extract(query)

        raw_requirements = await self._llm_extractor(query)
        return [
            InformationRequirement(
                description=r["description"],
                category=r.get("category", "general"),
                critical=r.get("critical", False),
                search_terms=tuple(r.get("search_terms", [])),
            )
            for r in raw_requirements
        ]

    @staticmethod
    def _default_patterns() -> dict[str, list[str]]:
        return {
            "code": [
                r"(?:file|module|function|class|method)\s+[`\"]?(\w[\w./]+)[`\"]?",
                r"(?:update|modify|change|edit)\s+(?:the\s+)?[`\"]?(\w[\w./]+)[`\"]?",
            ],
            "schema": [
                r"(?:table|column|field|index)\s+[`\"]?(\w+(?:\.\w+)?)[`\"]?",
                r"(?:database|db)\s+(?:schema|structure|migration)",
            ],
            "config": [
                r"(?:config|configuration|setting|env)\s+[`\"]?(\w+)[`\"]?",
                r"(?:\.env|\.yaml|\.json|\.toml)\s+(?:file)?",
            ],
            "docs": [
                r"(?:documentation|docs|readme|guide)\s+(?:for\s+)?[`\"]?(\w+)[`\"]?",
                r"(?:how to|convention|pattern|best practice)",
            ],
        }


class ContextCoverageAnalyzer:
    """
    Analyzes whether the current context covers all information
    requirements for a given query.
    """

    def __init__(
        self,
        extractor: RequirementExtractor,
        full_threshold: float = 0.9,
        adequate_threshold: float = 0.7,
        partial_threshold: float = 0.4,
    ):
        self._extractor = extractor
        self._full_threshold = full_threshold
        self._adequate_threshold = adequate_threshold
        self._partial_threshold = partial_threshold
        self._analysis_history: list[CoverageReport] = []

    def analyze(
        self,
        query: str,
        context_sections: dict[str, str],
        requirements: list[InformationRequirement] | None = None,
    ) -> CoverageReport:
        """
        Analyze context coverage for a query.

        Args:
            query: The user query to analyze.
            context_sections: Dict mapping section names to content.
            requirements: Optional pre-extracted requirements. If None,
                the extractor will derive them from the query.

        Returns:
            A CoverageReport with per-requirement scores and gap analysis.
        """
        if requirements is None:
            requirements = self._extractor.extract(query)

        if not requirements:
            # No specific requirements detected -- assume adequate
            report = CoverageReport(
                query=query,
                results=(),
                overall_score=1.0,
                level=CoverageLevel.FULL,
                action=CoverageAction.PROCEED,
                gaps=(),
                critical_gaps=(),
            )
            self._analysis_history = [*self._analysis_history, report]
            return report

        # Check each requirement against context
        results = []
        for req in requirements:
            score, evidence, source = self._check_requirement(
                req, context_sections
            )
            results.append(CoverageResult(
                requirement=req,
                score=score,
                evidence=evidence,
                source=source,
            ))

        # Compute overall score (critical requirements weighted 2x)
        total_weight = sum(
            2.0 if r.requirement.critical else 1.0 for r in results
        )
        weighted_score = sum(
            r.score * (2.0 if r.requirement.critical else 1.0)
            for r in results
        ) / total_weight if total_weight > 0 else 0.0

        # Identify gaps
        gap_threshold = 0.3
        gaps = tuple(
            r.requirement for r in results if r.score < gap_threshold
        )
        critical_gaps = tuple(
            r.requirement for r in results
            if r.score < gap_threshold and r.requirement.critical
        )

        # Determine coverage level and action
        level, action = self._determine_level_and_action(
            weighted_score, critical_gaps
        )

        report = CoverageReport(
            query=query,
            results=tuple(results),
            overall_score=weighted_score,
            level=level,
            action=action,
            gaps=gaps,
            critical_gaps=critical_gaps,
        )

        self._analysis_history = [*self._analysis_history, report]
        return report

    def _check_requirement(
        self,
        requirement: InformationRequirement,
        context_sections: dict[str, str],
    ) -> tuple[float, str, str]:
        """
        Check a single requirement against all context sections.

        Returns:
            Tuple of (score, evidence_excerpt, source_section_name).
        """
        best_score = 0.0
        best_evidence = ""
        best_source = ""

        for section_name, content in context_sections.items():
            content_lower = content.lower()

            for term in requirement.search_terms:
                if term in content_lower:
                    # Found a match -- score based on specificity
                    occurrences = content_lower.count(term)
                    # More occurrences = higher relevance (capped)
                    score = min(1.0, 0.5 + (occurrences * 0.1))

                    if score > best_score:
                        best_score = score
                        # Extract a snippet around the match
                        idx = content_lower.index(term)
                        start = max(0, idx - 100)
                        end = min(len(content), idx + len(term) + 100)
                        best_evidence = content[start:end].strip()
                        best_source = section_name

        return best_score, best_evidence, best_source

    def _determine_level_and_action(
        self,
        score: float,
        critical_gaps: tuple[InformationRequirement, ...],
    ) -> tuple[CoverageLevel, CoverageAction]:
        """Determine the coverage level and recommended action."""
        if critical_gaps:
            return CoverageLevel.INSUFFICIENT, CoverageAction.RETRIEVE_MORE

        if score >= self._full_threshold:
            return CoverageLevel.FULL, CoverageAction.PROCEED
        if score >= self._adequate_threshold:
            return CoverageLevel.ADEQUATE, CoverageAction.CAVEAT_RESPONSE
        if score >= self._partial_threshold:
            return CoverageLevel.PARTIAL, CoverageAction.RETRIEVE_MORE
        return CoverageLevel.INSUFFICIENT, CoverageAction.ASK_CLARIFICATION

    @property
    def average_coverage(self) -> float:
        if not self._analysis_history:
            return 0.0
        return sum(
            r.overall_score for r in self._analysis_history
        ) / len(self._analysis_history)

    @property
    def gap_frequency(self) -> dict[str, int]:
        """Which categories have gaps most often?"""
        freq: dict[str, int] = {}
        for report in self._analysis_history:
            for gap in report.gaps:
                freq[gap.category] = freq.get(gap.category, 0) + 1
        return freq
```

### TypeScript

```typescript
type CoverageLevel = "full" | "adequate" | "partial" | "insufficient";
type CoverageAction =
  | "proceed"
  | "retrieve_more"
  | "ask_clarification"
  | "caveat_response";

interface InformationRequirement {
  readonly description: string;
  readonly category: string;
  readonly critical: boolean;
  readonly searchTerms: readonly string[];
}

interface CoverageResult {
  readonly requirement: InformationRequirement;
  readonly score: number;
  readonly evidence: string;
  readonly source: string;
}

interface CoverageReport {
  readonly query: string;
  readonly results: readonly CoverageResult[];
  readonly overallScore: number;
  readonly level: CoverageLevel;
  readonly action: CoverageAction;
  readonly gaps: readonly InformationRequirement[];
  readonly criticalGaps: readonly InformationRequirement[];
}

function reportToContextBlock(report: CoverageReport): string {
  const lines = [
    "## Context Coverage Analysis",
    `**Coverage**: ${Math.round(report.overallScore * 100)}% (${report.level})`,
    `**Action**: ${report.action}`,
    "",
  ];

  if (report.gaps.length > 0) {
    lines.push("**Information gaps** (not found in context):");
    for (const gap of report.gaps) {
      const tag = gap.critical ? " [CRITICAL]" : "";
      lines.push(`- ${gap.description}${tag}`);
    }
  }

  if (report.criticalGaps.length > 0) {
    lines.push("");
    lines.push(
      "WARNING: Critical information is missing. Response may be inaccurate."
    );
  }

  return lines.join("\n");
}

interface DomainPatterns {
  readonly [category: string]: readonly string[];
}

const DEFAULT_PATTERNS: DomainPatterns = {
  code: [
    "(?:file|module|function|class|method)\\s+[`\"]?(\\w[\\w./]+)[`\"]?",
    "(?:update|modify|change|edit)\\s+(?:the\\s+)?[`\"]?(\\w[\\w./]+)[`\"]?",
  ],
  schema: [
    "(?:table|column|field|index)\\s+[`\"]?(\\w+(?:\\.\\w+)?)[`\"]?",
    "(?:database|db)\\s+(?:schema|structure|migration)",
  ],
  config: [
    "(?:config|configuration|setting|env)\\s+[`\"]?(\\w+)[`\"]?",
    "(?:\\.env|\\.yaml|\\.json|\\.toml)\\s+(?:file)?",
  ],
  docs: [
    "(?:documentation|docs|readme|guide)\\s+(?:for\\s+)?[`\"]?(\\w+)[`\"]?",
    "(?:how to|convention|pattern|best practice)",
  ],
};

class RequirementExtractor {
  private readonly patterns: DomainPatterns;

  constructor(patterns?: DomainPatterns) {
    this.patterns = patterns ?? DEFAULT_PATTERNS;
  }

  extract(query: string): InformationRequirement[] {
    const requirements: InformationRequirement[] = [];

    for (const [category, patterns] of Object.entries(this.patterns)) {
      for (const pattern of patterns) {
        const regex = new RegExp(pattern, "gi");
        let match: RegExpExecArray | null;

        while ((match = regex.exec(query)) !== null) {
          const term = match[1] ?? match[0];
          requirements.push({
            description: `${category}: ${term}`,
            category,
            critical: category === "code" || category === "schema",
            searchTerms: Object.freeze([term.toLowerCase()]),
          });
        }
      }
    }

    return requirements;
  }
}

class ContextCoverageAnalyzer {
  private readonly extractor: RequirementExtractor;
  private readonly fullThreshold: number;
  private readonly adequateThreshold: number;
  private readonly partialThreshold: number;
  private analysisHistory: readonly CoverageReport[] = [];

  constructor(params: {
    extractor: RequirementExtractor;
    fullThreshold?: number;
    adequateThreshold?: number;
    partialThreshold?: number;
  }) {
    this.extractor = params.extractor;
    this.fullThreshold = params.fullThreshold ?? 0.9;
    this.adequateThreshold = params.adequateThreshold ?? 0.7;
    this.partialThreshold = params.partialThreshold ?? 0.4;
  }

  analyze(
    query: string,
    contextSections: ReadonlyMap<string, string>,
    requirements?: InformationRequirement[]
  ): CoverageReport {
    const reqs = requirements ?? this.extractor.extract(query);

    if (reqs.length === 0) {
      const report: CoverageReport = {
        query,
        results: [],
        overallScore: 1.0,
        level: "full",
        action: "proceed",
        gaps: [],
        criticalGaps: [],
      };
      this.analysisHistory = Object.freeze([
        ...this.analysisHistory,
        report,
      ]);
      return report;
    }

    const results: CoverageResult[] = reqs.map((req) =>
      this.checkRequirement(req, contextSections)
    );

    // Weighted score (critical requirements count 2x)
    let totalWeight = 0;
    let weightedSum = 0;
    for (const r of results) {
      const weight = r.requirement.critical ? 2.0 : 1.0;
      totalWeight += weight;
      weightedSum += r.score * weight;
    }
    const overallScore = totalWeight > 0 ? weightedSum / totalWeight : 0;

    const gapThreshold = 0.3;
    const gaps = results
      .filter((r) => r.score < gapThreshold)
      .map((r) => r.requirement);
    const criticalGaps = gaps.filter((g) => g.critical);

    const { level, action } = this.determineLevelAndAction(
      overallScore,
      criticalGaps
    );

    const report: CoverageReport = {
      query,
      results: Object.freeze(results),
      overallScore,
      level,
      action,
      gaps: Object.freeze(gaps),
      criticalGaps: Object.freeze(criticalGaps),
    };

    this.analysisHistory = Object.freeze([
      ...this.analysisHistory,
      report,
    ]);
    return report;
  }

  private checkRequirement(
    requirement: InformationRequirement,
    contextSections: ReadonlyMap<string, string>
  ): CoverageResult {
    let bestScore = 0;
    let bestEvidence = "";
    let bestSource = "";

    for (const [sectionName, content] of contextSections) {
      const contentLower = content.toLowerCase();

      for (const term of requirement.searchTerms) {
        if (contentLower.includes(term)) {
          const occurrences = contentLower.split(term).length - 1;
          const score = Math.min(1.0, 0.5 + occurrences * 0.1);

          if (score > bestScore) {
            bestScore = score;
            const idx = contentLower.indexOf(term);
            const start = Math.max(0, idx - 100);
            const end = Math.min(content.length, idx + term.length + 100);
            bestEvidence = content.slice(start, end).trim();
            bestSource = sectionName;
          }
        }
      }
    }

    return {
      requirement,
      score: bestScore,
      evidence: bestEvidence,
      source: bestSource,
    };
  }

  private determineLevelAndAction(
    score: number,
    criticalGaps: readonly InformationRequirement[]
  ): { level: CoverageLevel; action: CoverageAction } {
    if (criticalGaps.length > 0) {
      return { level: "insufficient", action: "retrieve_more" };
    }
    if (score >= this.fullThreshold) {
      return { level: "full", action: "proceed" };
    }
    if (score >= this.adequateThreshold) {
      return { level: "adequate", action: "caveat_response" };
    }
    if (score >= this.partialThreshold) {
      return { level: "partial", action: "retrieve_more" };
    }
    return { level: "insufficient", action: "ask_clarification" };
  }

  get averageCoverage(): number {
    if (this.analysisHistory.length === 0) return 0;
    const total = this.analysisHistory.reduce(
      (sum, r) => sum + r.overallScore,
      0
    );
    return total / this.analysisHistory.length;
  }

  get gapFrequency(): ReadonlyMap<string, number> {
    const freq = new Map<string, number>();
    for (const report of this.analysisHistory) {
      for (const gap of report.gaps) {
        freq.set(gap.category, (freq.get(gap.category) ?? 0) + 1);
      }
    }
    return freq;
  }
}

export {
  ContextCoverageAnalyzer,
  RequirementExtractor,
  CoverageReport,
  CoverageResult,
  InformationRequirement,
  reportToContextBlock,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Prevents hallucination by detecting missing context before generation | Requirement extraction is imperfect -- may miss implicit needs or flag false gaps |
| Enables targeted retrieval (fetch only what is missing, not everything) | Adds latency to the pre-generation pipeline (extraction + coverage check) |
| Critical gap detection prevents high-confidence wrong answers | Pattern-based extraction requires domain-specific configuration |
| Gap frequency tracking reveals systematic retrieval weaknesses | LLM-based extraction is more accurate but adds an extra LLM call |
| Coverage reports give the model explicit awareness of its knowledge boundaries | Over-sensitive gap detection can trigger unnecessary retrieval or clarification |
| Actionable routing (proceed / retrieve / ask / caveat) automates the response | Simple keyword matching may miss semantic coverage (document discusses the topic without using the exact term) |

## When to Use

- RAG systems where retrieval quality varies and hallucination is harmful
- Coding agents that need specific files or schemas to produce correct changes
- Customer support systems where answering with wrong information is worse than not answering
- Any system where you can define what "complete context" means for common query types
- Multi-step agent workflows where each step has well-defined information requirements

## When NOT to Use

- Open-ended creative tasks where "complete context" is undefined
- Simple Q&A over a single document (coverage is trivially full or empty)
- When the LLM-based extraction cost exceeds the cost of occasional hallucination
- Low-stakes applications where approximate answers are acceptable
- Real-time systems where the pre-flight check latency is unacceptable

## Related Patterns

- **Parallel Context Assembly** (Optimization): Coverage analysis identifies gaps; parallel assembly fetches the missing context concurrently.
- **Context Rot Detection** (Evaluation): Rot detection monitors context health over time; coverage analysis checks context adequacy per-query. They are complementary.
- **Ablation Testing** (Evaluation): Ablation tests which context components contribute to quality; coverage analysis tests which are necessary for correctness. Ablation is offline; coverage is online.
- **Error Preservation** (Optimization): When errors occur due to missing context, coverage analysis retrospectively identifies which gaps caused the failure.

## Real-World Examples

- **Perplexity**: Performs search-result relevance analysis before generating answers. If search results do not cover the query, additional searches are triggered rather than generating from insufficient context.
- **GitHub Copilot Workspace**: Analyzes which files and symbols are needed to implement a change, retrieves them, and flags when required files are missing from the workspace context.
- **Medical AI Systems**: Safety-critical systems that verify all required patient data (labs, imaging, history) is present before generating diagnostic suggestions. Missing critical data triggers a "request more information" response rather than a guess.
- **Legal Research AI**: Systems that check whether all relevant statutes, case law, and jurisdictional information are present before generating legal analysis. Coverage gaps are explicitly flagged in the output.
