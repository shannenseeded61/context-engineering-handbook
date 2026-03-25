# Temporal Context Selection

> Select context based on temporal relevance -- prioritize recent information, apply time-decay scoring, and resolve versioned content to the correct point in time.

## Problem

Knowledge bases accumulate content over time: API documentation gets updated across versions, company policies change quarterly, pricing models evolve, and troubleshooting guides become obsolete. When multiple versions of the same information exist in the retrieval corpus, standard semantic search has no notion of time -- it returns whichever version best matches the query embedding, which is often an older, more heavily linked document rather than the current one.

This produces three specific failure modes:

1. **Stale answers** -- The system retrieves deprecated API signatures, expired pricing, or superseded policies and presents them as current fact.
2. **Version confusion** -- Results mix information from different versions (v2 configuration syntax with v3 feature descriptions), producing internally inconsistent context.
3. **Recency blindness** -- Recent updates (security advisories, breaking changes, new features) are buried under older, more frequently retrieved content.

Without temporal awareness, a RAG system confidently serves yesterday's truth as today's answer.

## Solution

Temporal Context Selection adds a time dimension to retrieval scoring. Each retrieved chunk is scored not only on semantic relevance but also on **temporal relevance** -- how recent the content is, whether it is the current version, and whether it has been superseded.

The pattern has three components:

1. **Time-decay scoring** -- Apply a decay function (exponential or linear) that reduces the score of older content. Recent documents get a boost; ancient documents get penalized.
2. **Version-aware deduplication** -- When multiple versions of the same logical document exist, identify them as versions of the same entity and select only the most appropriate version (usually the latest, unless the user specifies otherwise).
3. **Freshness signals** -- Incorporate explicit metadata signals: `created_at`, `updated_at`, `deprecated_at`, `version`, `superseded_by`. These signals override pure time-decay when available.

## How It Works

```
Retrieved Chunks (from any first-stage retriever)
         |
         v
+---------------------------+
| Temporal Metadata         |
| Extraction                |
|                           |
| - created_at / updated_at |
| - version identifier      |
| - superseded_by link      |
| - deprecation status      |
+---------------------------+
         |
         v
+---------------------------+
| Time-Decay Scoring        |
|                           |
| decay(age) =              |
|   e^(-lambda * age_days)  |
|                           |
| adjusted_score =          |
|   relevance * decay(age)  |
+---------------------------+
         |
         v
+---------------------------+
| Version-Aware Dedup       |
|                           |
| Group by logical doc ID   |
| Keep best version per     |
| group (latest or          |
| user-specified)           |
+---------------------------+
         |
         v
+---------------------------+
| Freshness Penalties       |
|                           |
| - Deprecated? score *= 0  |
| - Superseded? score *= 0.1|
| - No update in 2yr?       |
|     flag as potentially   |
|     stale                 |
+---------------------------+
         |
         v
  Temporally-ranked results
```

1. **Extract temporal metadata** -- For each chunk, read `created_at`, `updated_at`, `version`, `deprecated`, and `superseded_by` from metadata.
2. **Apply time-decay** -- Multiply each chunk's relevance score by an exponential decay factor based on content age. The decay rate (lambda) is configurable per domain.
3. **Deduplicate by version** -- Group chunks that represent the same logical document (same `doc_family_id`). Within each group, keep only the version with the highest temporal score.
4. **Apply freshness penalties** -- Zero out deprecated content. Heavily penalize superseded content. Flag very old content as potentially stale.
5. **Sort and return** -- The final ranking reflects both semantic relevance and temporal appropriateness.

## Implementation

### Python

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TemporalMetadata:
    """Time-related metadata for a retrieved chunk."""

    created_at: datetime
    updated_at: datetime | None = None
    version: str | None = None
    doc_family_id: str | None = None  # Groups versions of same doc
    deprecated: bool = False
    superseded_by: str | None = None  # doc_id of newer version


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    content: str
    source: str
    relevance_score: float
    temporal: TemporalMetadata
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalResult:
    chunk: RetrievedChunk
    original_score: float
    temporal_score: float
    decay_factor: float
    freshness_penalty: float
    staleness_warning: str | None


class TemporalContextSelector:
    """
    Re-scores and filters retrieved chunks based on temporal relevance.

    Combines three temporal signals:
    1. Time-decay: exponential decay based on content age
    2. Version dedup: keep only the best version of each logical document
    3. Freshness penalties: zero out deprecated, penalize superseded
    """

    def __init__(
        self,
        decay_lambda: float = 0.005,
        staleness_threshold_days: int = 730,
        superseded_penalty: float = 0.1,
        reference_time: datetime | None = None,
        prefer_version: str | None = None,
    ) -> None:
        self._decay_lambda = decay_lambda
        self._staleness_days = staleness_threshold_days
        self._superseded_penalty = superseded_penalty
        self._now = reference_time or datetime.now(timezone.utc)
        self._prefer_version = prefer_version  # e.g., "v2" to pin a version

    def select(
        self, chunks: list[RetrievedChunk], top_k: int = 10
    ) -> list[TemporalResult]:
        """Apply temporal scoring, dedup, and return top-K results."""

        # --- 1. Score each chunk temporally ---
        scored = [self._score_chunk(chunk) for chunk in chunks]

        # --- 2. Version-aware deduplication ---
        scored = self._deduplicate_versions(scored)

        # --- 3. Sort by temporal score descending ---
        scored.sort(key=lambda r: r.temporal_score, reverse=True)

        return scored[:top_k]

    def _score_chunk(self, chunk: RetrievedChunk) -> TemporalResult:
        """Compute the temporally-adjusted score for a single chunk."""

        t = chunk.temporal
        effective_date = t.updated_at or t.created_at
        age_days = (self._now - effective_date).total_seconds() / 86400.0

        # --- Time decay ---
        decay = math.exp(-self._decay_lambda * max(age_days, 0))

        # --- Freshness penalty ---
        penalty = 0.0
        staleness_warning = None

        if t.deprecated:
            penalty = 1.0  # Complete elimination
            staleness_warning = "DEPRECATED: This content has been marked as deprecated."
        elif t.superseded_by is not None:
            penalty = 1.0 - self._superseded_penalty
            staleness_warning = f"SUPERSEDED: Newer version exists ({t.superseded_by})."
        elif age_days > self._staleness_days:
            staleness_warning = (
                f"POTENTIALLY STALE: Last updated {int(age_days)} days ago "
                f"(threshold: {self._staleness_days} days)."
            )

        # --- Combine ---
        adjusted = chunk.relevance_score * decay * (1.0 - penalty)

        return TemporalResult(
            chunk=chunk,
            original_score=chunk.relevance_score,
            temporal_score=adjusted,
            decay_factor=decay,
            freshness_penalty=penalty,
            staleness_warning=staleness_warning,
        )

    def _deduplicate_versions(
        self, results: list[TemporalResult]
    ) -> list[TemporalResult]:
        """Keep only the best version within each document family."""

        # Chunks without a doc_family_id are treated as unique.
        families: dict[str, list[TemporalResult]] = {}
        standalone: list[TemporalResult] = []

        for r in results:
            family_id = r.chunk.temporal.doc_family_id
            if family_id is None:
                standalone.append(r)
            else:
                families.setdefault(family_id, []).append(r)

        deduplicated = list(standalone)

        for family_id, members in families.items():
            # If user pinned a version, prefer that one.
            if self._prefer_version:
                pinned = [
                    m
                    for m in members
                    if m.chunk.temporal.version == self._prefer_version
                ]
                if pinned:
                    deduplicated.append(max(pinned, key=lambda r: r.temporal_score))
                    continue

            # Otherwise, pick the member with the highest temporal score.
            # This naturally favors the most recent, non-deprecated version
            # because time-decay and freshness penalties have already been applied.
            deduplicated.append(max(members, key=lambda r: r.temporal_score))

        return deduplicated


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

def main():
    selector = TemporalContextSelector(
        decay_lambda=0.005,       # Moderate decay (~50% at 139 days)
        staleness_threshold_days=365,
        superseded_penalty=0.1,   # Superseded docs keep 10% of score
    )

    chunks = [
        RetrievedChunk(
            doc_id="pricing-v3",
            content="Acme Pro: $49/mo (effective Jan 2026)",
            source="docs",
            relevance_score=0.92,
            temporal=TemporalMetadata(
                created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
                version="v3",
                doc_family_id="pricing",
            ),
        ),
        RetrievedChunk(
            doc_id="pricing-v2",
            content="Acme Pro: $39/mo (effective Jan 2025)",
            source="docs",
            relevance_score=0.95,  # Higher semantic score (more links/references)
            temporal=TemporalMetadata(
                created_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
                version="v2",
                doc_family_id="pricing",
                superseded_by="pricing-v3",
            ),
        ),
        RetrievedChunk(
            doc_id="pricing-v1",
            content="Acme Pro: $29/mo (effective Jan 2024)",
            source="docs",
            relevance_score=0.93,
            temporal=TemporalMetadata(
                created_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
                version="v1",
                doc_family_id="pricing",
                deprecated=True,
            ),
        ),
        RetrievedChunk(
            doc_id="migration-guide",
            content="Migrating from v2 to v3: update your config...",
            source="docs",
            relevance_score=0.78,
            temporal=TemporalMetadata(
                created_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
            ),
        ),
    ]

    results = selector.select(chunks, top_k=5)

    for r in results:
        print(f"[{r.temporal_score:.4f}] {r.chunk.doc_id} (v{r.chunk.temporal.version or 'N/A'})")
        print(f"  Original: {r.original_score:.3f}, Decay: {r.decay_factor:.3f}, Penalty: {r.freshness_penalty:.3f}")
        if r.staleness_warning:
            print(f"  WARNING: {r.staleness_warning}")
        print(f"  {r.chunk.content}")
        print()

    # Expected output:
    # pricing-v3 wins over pricing-v2 despite v2 having a higher semantic score,
    # because v2 is superseded and v3 is more recent.
    # pricing-v1 is eliminated entirely (deprecated).
    # migration-guide appears as a standalone result.
```

### TypeScript

```typescript
interface TemporalMetadata {
  createdAt: Date;
  updatedAt?: Date;
  version?: string;
  docFamilyId?: string; // Groups versions of same doc
  deprecated?: boolean;
  supersededBy?: string; // docId of newer version
}

interface RetrievedChunk {
  docId: string;
  content: string;
  source: string;
  relevanceScore: number;
  temporal: TemporalMetadata;
  metadata?: Record<string, unknown>;
}

interface TemporalResult {
  chunk: RetrievedChunk;
  originalScore: number;
  temporalScore: number;
  decayFactor: number;
  freshnessPenalty: number;
  stalenessWarning: string | null;
}

interface TemporalSelectorConfig {
  decayLambda?: number; // Decay rate (default: 0.005)
  stalenessThresholdDays?: number; // Flag as stale after N days (default: 730)
  supersededPenalty?: number; // Score multiplier for superseded docs (default: 0.1)
  referenceTime?: Date; // "Now" for scoring (default: Date.now)
  preferVersion?: string; // Pin to a specific version
}

class TemporalContextSelector {
  private readonly decayLambda: number;
  private readonly stalenessDays: number;
  private readonly supersededPenalty: number;
  private readonly now: Date;
  private readonly preferVersion?: string;

  constructor(config: TemporalSelectorConfig = {}) {
    this.decayLambda = config.decayLambda ?? 0.005;
    this.stalenessDays = config.stalenessThresholdDays ?? 730;
    this.supersededPenalty = config.supersededPenalty ?? 0.1;
    this.now = config.referenceTime ?? new Date();
    this.preferVersion = config.preferVersion;
  }

  select(chunks: RetrievedChunk[], topK = 10): TemporalResult[] {
    // 1. Score each chunk temporally
    let scored = chunks.map((chunk) => this.scoreChunk(chunk));

    // 2. Version-aware deduplication
    scored = this.deduplicateVersions(scored);

    // 3. Sort by temporal score descending
    scored.sort((a, b) => b.temporalScore - a.temporalScore);

    return scored.slice(0, topK);
  }

  private scoreChunk(chunk: RetrievedChunk): TemporalResult {
    const { temporal } = chunk;
    const effectiveDate = temporal.updatedAt ?? temporal.createdAt;
    const ageDays =
      (this.now.getTime() - effectiveDate.getTime()) / (1000 * 60 * 60 * 24);

    // Time decay
    const decay = Math.exp(-this.decayLambda * Math.max(ageDays, 0));

    // Freshness penalty
    let penalty = 0;
    let stalenessWarning: string | null = null;

    if (temporal.deprecated) {
      penalty = 1.0;
      stalenessWarning = "DEPRECATED: This content has been marked as deprecated.";
    } else if (temporal.supersededBy) {
      penalty = 1.0 - this.supersededPenalty;
      stalenessWarning = `SUPERSEDED: Newer version exists (${temporal.supersededBy}).`;
    } else if (ageDays > this.stalenessDays) {
      stalenessWarning =
        `POTENTIALLY STALE: Last updated ${Math.floor(ageDays)} days ago ` +
        `(threshold: ${this.stalenessDays} days).`;
    }

    const adjusted = chunk.relevanceScore * decay * (1.0 - penalty);

    return {
      chunk,
      originalScore: chunk.relevanceScore,
      temporalScore: adjusted,
      decayFactor: decay,
      freshnessPenalty: penalty,
      stalenessWarning,
    };
  }

  private deduplicateVersions(results: TemporalResult[]): TemporalResult[] {
    const families = new Map<string, TemporalResult[]>();
    const standalone: TemporalResult[] = [];

    for (const r of results) {
      const familyId = r.chunk.temporal.docFamilyId;
      if (!familyId) {
        standalone.push(r);
      } else {
        const existing = families.get(familyId) ?? [];
        existing.push(r);
        families.set(familyId, existing);
      }
    }

    const deduplicated = [...standalone];

    for (const [, members] of families) {
      // If user pinned a version, prefer it.
      if (this.preferVersion) {
        const pinned = members.filter(
          (m) => m.chunk.temporal.version === this.preferVersion
        );
        if (pinned.length > 0) {
          deduplicated.push(
            pinned.reduce((best, m) =>
              m.temporalScore > best.temporalScore ? m : best
            )
          );
          continue;
        }
      }

      // Otherwise, pick the highest temporal score.
      deduplicated.push(
        members.reduce((best, m) =>
          m.temporalScore > best.temporalScore ? m : best
        )
      );
    }

    return deduplicated;
  }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

const selector = new TemporalContextSelector({
  decayLambda: 0.005,
  stalenessThresholdDays: 365,
  supersededPenalty: 0.1,
});

const chunks: RetrievedChunk[] = [
  {
    docId: "pricing-v3",
    content: "Acme Pro: $49/mo (effective Jan 2026)",
    source: "docs",
    relevanceScore: 0.92,
    temporal: {
      createdAt: new Date("2026-01-15"),
      version: "v3",
      docFamilyId: "pricing",
    },
  },
  {
    docId: "pricing-v2",
    content: "Acme Pro: $39/mo (effective Jan 2025)",
    source: "docs",
    relevanceScore: 0.95, // Higher semantic score
    temporal: {
      createdAt: new Date("2025-01-10"),
      version: "v2",
      docFamilyId: "pricing",
      supersededBy: "pricing-v3",
    },
  },
  {
    docId: "pricing-v1",
    content: "Acme Pro: $29/mo (effective Jan 2024)",
    source: "docs",
    relevanceScore: 0.93,
    temporal: {
      createdAt: new Date("2024-01-05"),
      version: "v1",
      docFamilyId: "pricing",
      deprecated: true,
    },
  },
];

const results = selector.select(chunks, 5);

for (const r of results) {
  console.log(
    `[${r.temporalScore.toFixed(4)}] ${r.chunk.docId} (${r.chunk.temporal.version ?? "N/A"})`
  );
  if (r.stalenessWarning) {
    console.log(`  WARNING: ${r.stalenessWarning}`);
  }
  console.log(`  ${r.chunk.content}`);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Eliminates stale answers by suppressing deprecated and superseded content | Requires temporal metadata on all documents -- ingestion pipelines must capture timestamps and version info |
| Version-aware dedup prevents mixing information from incompatible versions | Decay function tuning is domain-specific: legal docs decay slowly, tech docs decay quickly |
| Staleness warnings let the model (or user) know when results may be outdated | Over-aggressive decay can suppress still-relevant older content (foundational docs, RFCs, specifications) |
| Version pinning supports users who intentionally work with older versions | Maintaining `doc_family_id` and `superseded_by` links requires deliberate data modeling |
| Composable: works as a post-retrieval filter on top of any search backend | Adds a scoring step that increases retrieval pipeline complexity |

## When to Use

- Your corpus contains versioned content: API docs, policies, pricing, legal documents, changelogs.
- Users frequently receive stale answers because older, better-linked documents outrank newer ones in pure semantic search.
- Multiple versions of the same information coexist in the retrieval index.
- Your domain has a clear notion of information freshness (news, tech docs, financial data, security advisories).
- You need to support version-pinned queries ("How does the v2 API handle auth?") alongside latest-version queries.

## When NOT to Use

- Your corpus is static and does not change over time (e.g., a fixed textbook, historical archives).
- All documents are equally valid regardless of age (e.g., mathematical proofs, philosophical texts).
- You have a strict "latest only" policy and simply delete old versions from the index -- temporal selection is unnecessary if old versions do not exist.
- Your index is small enough to manually curate, making automated temporal filtering overkill.

## Related Patterns

- **[RAG Context Assembly](rag-context-assembly.md)** -- After temporal selection narrows results to the correct versions, RAG Context Assembly handles token budgeting and source attribution.
- **[Hybrid Search Fusion](hybrid-search-fusion.md)** -- Temporal selection can be applied as a post-fusion step, adjusting the fused ranking with temporal signals.
- **[Context-Aware Re-ranking](context-aware-reranking.md)** -- Temporal selection and context-aware re-ranking address orthogonal dimensions (time vs. conversation). They compose well: apply temporal selection first to eliminate stale content, then re-rank the survivors with conversation context.
- **[Just-in-Time Retrieval](just-in-time-retrieval.md)** -- JIT retrieval fetches fresh data at the moment of need, which naturally favors recency. Temporal selection adds structure to that recency preference with decay functions and version awareness.

## Real-World Examples

- **Stripe API Docs**: Stripe maintains versioned API documentation. When a developer asks about a specific API version, the retrieval system must return docs for that version, not the latest. Temporal context selection with version pinning solves this.
- **Wikipedia Current Events**: Wikipedia distinguishes between historical articles and current events. Search systems that surface Wikipedia content apply recency boosts to current events while keeping historical articles at their base relevance.
- **Legal Research (Westlaw, LexisNexis)**: Legal research platforms must distinguish between current law and superseded statutes. Temporal metadata (effective dates, repeal dates) determines which version of a law is authoritative for a given date.
- **Security Advisory Databases (CVE, NVD)**: When querying for vulnerabilities, the system must surface the latest advisory with current patch status, not an initial report that may lack mitigation details. Time-decay and supersession tracking ensure current information surfaces first.
- **Confluence/Notion Enterprise Wikis**: Internal knowledge bases accumulate outdated pages over years. Teams that add `last_verified` metadata and apply temporal decay in their RAG systems dramatically reduce the rate of stale internal answers.
