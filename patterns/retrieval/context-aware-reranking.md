# Context-Aware Re-ranking

> Re-rank retrieved results using the full conversation context -- not just the latest query -- to dramatically improve relevance ordering.

## Problem

Standard retrieval systems rank results based on similarity to the current query alone. But in a multi-turn conversation, the current query is often ambiguous, abbreviated, or referentially dependent on earlier turns. "What about the pricing?" means nothing without knowing which product the user has been discussing. A user who has already established they are working with Python 3.12 on AWS Lambda does not need results about Python 2.7 on bare metal.

Query-only re-ranking produces three failure modes:

1. **Referential ambiguity** -- The query uses pronouns or implicit references ("it", "that API", "the error from before") that only make sense in conversation context.
2. **Redundancy** -- Results repeat information the user has already seen or that the model has already provided, wasting context window tokens.
3. **Context drift** -- Results are topically related to the query but irrelevant to the established task state (e.g., returning beginner tutorials when the conversation has established expert-level discussion).

## Solution

Context-Aware Re-ranking adds a second-stage ranking pass after initial retrieval. Instead of scoring each result against the raw query, the re-ranker scores each result against a **context envelope** that includes:

- The full conversation history (or a compressed summary of it)
- Established facts and constraints (extracted entities, declared preferences)
- The current task state (what step the user is on, what has already been answered)

The re-ranker can be a **cross-encoder model** (e.g., `cross-encoder/ms-marco-MiniLM`) that scores query-document relevance with full attention, or an **LLM-as-judge** that rates each chunk's relevance given the conversation context. The cross-encoder approach is faster and cheaper; the LLM-as-judge approach handles nuance better for complex conversations.

## How It Works

```
Conversation History      Current Query      Initial Retrieval
[turn1, turn2, ...]      "What about pricing?"    (top-50 chunks)
         |                      |                      |
         v                      v                      v
  +----------------------------------------------+
  |          Context Envelope Assembly            |
  |                                               |
  |  - Conversation summary                       |
  |  - Extracted entities: {product: "Acme Pro"}  |
  |  - Task state: "comparing plans"              |
  |  - Already-covered topics                     |
  +----------------------------------------------+
                        |
                        v
           +------------------------+
           |  Re-ranker             |
           |  (cross-encoder or     |
           |   LLM-as-judge)        |
           |                        |
           |  For each chunk:       |
           |    score(chunk,        |
           |          envelope)     |
           +------------------------+
                        |
                        v
             +--------------------+
             | Re-ranked results  |
             | (top-K by          |
             |  context relevance)|
             +--------------------+
```

1. **Build the context envelope** -- Summarize conversation history, extract key entities and constraints, and identify the current task state.
2. **Retrieve candidates** -- Use any first-stage retriever (BM25, semantic, hybrid) to get a broad candidate set (e.g., top-50).
3. **Score each candidate** -- Pass each candidate through the re-ranker alongside the context envelope. The re-ranker assigns a relevance score considering the full conversation, not just the latest query.
4. **Penalize redundancy** -- Reduce scores for chunks covering topics already discussed in the conversation.
5. **Return re-ranked top-K** -- The top results are now ordered by true conversational relevance.

## Implementation

### Python

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ConversationTurn:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RerankedResult:
    chunk: RetrievedChunk
    relevance_score: float
    reasoning: str  # Why this ranking was chosen


@dataclass(frozen=True)
class ContextEnvelope:
    """Everything the re-ranker needs to assess relevance."""

    conversation_summary: str
    extracted_entities: dict[str, str]
    current_query: str
    task_state: str
    covered_topics: list[str]


class RerankerBackend(Protocol):
    """Interface for a re-ranking backend."""

    async def score(
        self, envelope: ContextEnvelope, chunks: list[RetrievedChunk]
    ) -> list[tuple[float, str]]:
        """Return (score, reasoning) for each chunk."""
        ...


class CrossEncoderReranker:
    """Re-rank using a cross-encoder model (fast, cost-effective)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    async def score(
        self, envelope: ContextEnvelope, chunks: list[RetrievedChunk]
    ) -> list[tuple[float, str]]:
        # Build the "query" side from the full context envelope, not just
        # the raw query. This is the key insight of this pattern.
        enriched_query = (
            f"Context: {envelope.conversation_summary}\n"
            f"Entities: {envelope.extracted_entities}\n"
            f"Task: {envelope.task_state}\n"
            f"Question: {envelope.current_query}"
        )

        pairs = [(enriched_query, chunk.content) for chunk in chunks]

        # Cross-encoder scoring is CPU-bound; run in executor to avoid
        # blocking the event loop.
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, lambda: self._model.predict(pairs).tolist()
        )

        return [(score, "cross-encoder score") for score in scores]


class LLMJudgeReranker:
    """Re-rank using an LLM as a relevance judge (nuanced, higher cost)."""

    def __init__(self, llm_client, model: str = "gpt-4o-mini") -> None:
        self._client = llm_client
        self._model = model

    async def score(
        self, envelope: ContextEnvelope, chunks: list[RetrievedChunk]
    ) -> list[tuple[float, str]]:

        async def score_one(chunk: RetrievedChunk) -> tuple[float, str]:
            prompt = f"""Rate the relevance of this text chunk to the user's current need.

## Conversation context
{envelope.conversation_summary}

## Known entities
{envelope.extracted_entities}

## Current task
{envelope.task_state}

## Topics already covered (penalize redundancy)
{envelope.covered_topics}

## Current question
{envelope.current_query}

## Chunk to evaluate
{chunk.content}

Respond with a JSON object:
{{"score": <0.0 to 1.0>, "reasoning": "<one sentence>"}}
"""
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            import json

            result = json.loads(response.choices[0].message.content)
            return (float(result["score"]), result["reasoning"])

        # Score all chunks concurrently.
        return await asyncio.gather(*(score_one(chunk) for chunk in chunks))


class ContextAwareReranker:
    """
    Orchestrates context-aware re-ranking:
    1. Builds a context envelope from conversation history.
    2. Scores each retrieved chunk against the envelope.
    3. Applies redundancy penalties.
    4. Returns re-ranked results.
    """

    def __init__(
        self,
        backend: RerankerBackend,
        summarizer=None,
        redundancy_penalty: float = 0.3,
    ) -> None:
        self._backend = backend
        self._summarizer = summarizer
        self._redundancy_penalty = redundancy_penalty

    async def rerank(
        self,
        query: str,
        conversation: list[ConversationTurn],
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RerankedResult]:
        # --- 1. Build context envelope ---
        envelope = await self._build_envelope(query, conversation)

        # --- 2. Score all chunks ---
        scores_and_reasons = await self._backend.score(envelope, chunks)

        # --- 3. Apply redundancy penalty ---
        results: list[RerankedResult] = []
        covered = set(envelope.covered_topics)

        for chunk, (score, reasoning) in zip(chunks, scores_and_reasons):
            # Simple redundancy check: penalize chunks whose content
            # overlaps significantly with already-covered topics.
            penalty = 0.0
            for topic in covered:
                if topic.lower() in chunk.content.lower():
                    penalty = self._redundancy_penalty
                    break

            final_score = max(0.0, score - penalty)
            results.append(RerankedResult(
                chunk=chunk,
                relevance_score=final_score,
                reasoning=reasoning,
            ))

        # --- 4. Sort and return top-K ---
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:top_k]

    async def _build_envelope(
        self, query: str, conversation: list[ConversationTurn]
    ) -> ContextEnvelope:
        """Extract structured context from the conversation."""

        # Build a simple summary from recent turns.
        recent = conversation[-10:]  # Last 10 turns
        summary_text = "\n".join(f"{t.role}: {t.content}" for t in recent)

        # Extract entities mentioned in conversation (simplified).
        entities = self._extract_entities(conversation)

        # Identify topics already covered by assistant responses.
        covered = [
            t.content[:100]
            for t in conversation
            if t.role == "assistant"
        ]

        # Determine task state from the last few turns.
        task_state = self._infer_task_state(recent)

        return ContextEnvelope(
            conversation_summary=summary_text,
            extracted_entities=entities,
            current_query=query,
            task_state=task_state,
            covered_topics=covered,
        )

    @staticmethod
    def _extract_entities(conversation: list[ConversationTurn]) -> dict[str, str]:
        """Simple entity extraction from conversation turns."""
        entities: dict[str, str] = {}
        for turn in conversation:
            content = turn.content.lower()
            # In production, use NER or an LLM for extraction.
            # This demonstrates the pattern with simple heuristics.
            if "python" in content:
                entities["language"] = "Python"
            if "typescript" in content:
                entities["language"] = "TypeScript"
            if "aws" in content:
                entities["cloud"] = "AWS"
        return entities

    @staticmethod
    def _infer_task_state(recent_turns: list[ConversationTurn]) -> str:
        """Infer what the user is currently trying to accomplish."""
        if not recent_turns:
            return "unknown"
        last_user = next(
            (t for t in reversed(recent_turns) if t.role == "user"), None
        )
        if last_user is None:
            return "unknown"
        content = last_user.content.lower()
        if any(w in content for w in ["error", "bug", "fix", "broken"]):
            return "debugging"
        if any(w in content for w in ["how to", "implement", "build", "create"]):
            return "building"
        if any(w in content for w in ["compare", "vs", "difference", "better"]):
            return "comparing"
        return "exploring"


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

async def main():
    # Choose your backend:
    # backend = CrossEncoderReranker()
    # backend = LLMJudgeReranker(openai_client)

    reranker = ContextAwareReranker(
        backend=backend,
        redundancy_penalty=0.3,
    )

    conversation = [
        ConversationTurn("user", "I'm building a FastAPI app on AWS Lambda"),
        ConversationTurn("assistant", "Great choice. FastAPI works well on Lambda via Mangum..."),
        ConversationTurn("user", "What about cold starts?"),
        ConversationTurn("assistant", "Cold starts on Lambda with Python typically take 1-3s..."),
        ConversationTurn("user", "How can I reduce them?"),
    ]

    # Assume these came from a first-stage retriever.
    candidate_chunks = await first_stage_retriever.retrieve(
        "How can I reduce them?", top_k=50
    )

    results = await reranker.rerank(
        query="How can I reduce them?",
        conversation=conversation,
        chunks=candidate_chunks,
        top_k=10,
    )

    for r in results:
        print(f"[{r.relevance_score:.3f}] {r.chunk.doc_id}")
        print(f"  Reason: {r.reasoning}")
        print(f"  {r.chunk.content[:120]}...")
```

### TypeScript

```typescript
interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

interface RetrievedChunk {
  docId: string;
  content: string;
  source: string;
  metadata: Record<string, unknown>;
}

interface RerankedResult {
  chunk: RetrievedChunk;
  relevanceScore: number;
  reasoning: string;
}

interface ContextEnvelope {
  conversationSummary: string;
  extractedEntities: Record<string, string>;
  currentQuery: string;
  taskState: string;
  coveredTopics: string[];
}

interface RerankerBackend {
  score(
    envelope: ContextEnvelope,
    chunks: RetrievedChunk[]
  ): Promise<Array<{ score: number; reasoning: string }>>;
}

class LLMJudgeReranker implements RerankerBackend {
  constructor(
    private readonly client: any, // OpenAI-compatible client
    private readonly model = "gpt-4o-mini"
  ) {}

  async score(
    envelope: ContextEnvelope,
    chunks: RetrievedChunk[]
  ): Promise<Array<{ score: number; reasoning: string }>> {
    const scoreOne = async (
      chunk: RetrievedChunk
    ): Promise<{ score: number; reasoning: string }> => {
      const prompt = `Rate the relevance of this text chunk to the user's current need.

## Conversation context
${envelope.conversationSummary}

## Known entities
${JSON.stringify(envelope.extractedEntities)}

## Current task
${envelope.taskState}

## Topics already covered (penalize redundancy)
${JSON.stringify(envelope.coveredTopics)}

## Current question
${envelope.currentQuery}

## Chunk to evaluate
${chunk.content}

Respond with JSON: {"score": <0.0 to 1.0>, "reasoning": "<one sentence>"}`;

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
        temperature: 0,
      });

      return JSON.parse(response.choices[0].message.content!);
    };

    return Promise.all(chunks.map(scoreOne));
  }
}

class ContextAwareReranker {
  constructor(
    private readonly backend: RerankerBackend,
    private readonly redundancyPenalty = 0.3
  ) {}

  async rerank(
    query: string,
    conversation: ConversationTurn[],
    chunks: RetrievedChunk[],
    topK = 10
  ): Promise<RerankedResult[]> {
    // 1. Build context envelope
    const envelope = this.buildEnvelope(query, conversation);

    // 2. Score all chunks
    const scores = await this.backend.score(envelope, chunks);

    // 3. Apply redundancy penalty and assemble results
    const results: RerankedResult[] = chunks.map((chunk, i) => {
      const { score, reasoning } = scores[i];

      const penalty = envelope.coveredTopics.some((topic) =>
        chunk.content.toLowerCase().includes(topic.toLowerCase())
      )
        ? this.redundancyPenalty
        : 0;

      return {
        chunk,
        relevanceScore: Math.max(0, score - penalty),
        reasoning,
      };
    });

    // 4. Sort and return top-K
    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return results.slice(0, topK);
  }

  private buildEnvelope(
    query: string,
    conversation: ConversationTurn[]
  ): ContextEnvelope {
    const recent = conversation.slice(-10);

    return {
      conversationSummary: recent
        .map((t) => `${t.role}: ${t.content}`)
        .join("\n"),
      extractedEntities: this.extractEntities(conversation),
      currentQuery: query,
      taskState: this.inferTaskState(recent),
      coveredTopics: conversation
        .filter((t) => t.role === "assistant")
        .map((t) => t.content.slice(0, 100)),
    };
  }

  private extractEntities(
    conversation: ConversationTurn[]
  ): Record<string, string> {
    const entities: Record<string, string> = {};
    for (const turn of conversation) {
      const lower = turn.content.toLowerCase();
      if (lower.includes("python")) entities.language = "Python";
      if (lower.includes("typescript")) entities.language = "TypeScript";
      if (lower.includes("aws")) entities.cloud = "AWS";
    }
    return entities;
  }

  private inferTaskState(recent: ConversationTurn[]): string {
    const lastUser = [...recent].reverse().find((t) => t.role === "user");
    if (!lastUser) return "unknown";
    const lower = lastUser.content.toLowerCase();
    if (["error", "bug", "fix", "broken"].some((w) => lower.includes(w)))
      return "debugging";
    if (["how to", "implement", "build", "create"].some((w) => lower.includes(w)))
      return "building";
    if (["compare", "vs", "difference", "better"].some((w) => lower.includes(w)))
      return "comparing";
    return "exploring";
  }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

const reranker = new ContextAwareReranker(
  new LLMJudgeReranker(openaiClient),
  0.3
);

const conversation: ConversationTurn[] = [
  { role: "user", content: "I'm building a FastAPI app on AWS Lambda" },
  { role: "assistant", content: "Great choice. FastAPI works well on Lambda via Mangum..." },
  { role: "user", content: "What about cold starts?" },
  { role: "assistant", content: "Cold starts on Lambda with Python typically take 1-3s..." },
  { role: "user", content: "How can I reduce them?" },
];

const results = await reranker.rerank(
  "How can I reduce them?",
  conversation,
  candidateChunks,
  10
);

for (const r of results) {
  console.log(`[${r.relevanceScore.toFixed(3)}] ${r.chunk.docId}`);
  console.log(`  Reason: ${r.reasoning}`);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Resolves referential ambiguity ("it", "that", "the error") by grounding in conversation context | Adds latency: cross-encoder re-ranking adds 50-200ms, LLM-as-judge adds 500-2000ms |
| Eliminates redundancy by penalizing chunks covering already-discussed topics | LLM-as-judge approach has non-trivial per-query cost for large candidate sets |
| Adapts ranking to task state -- debugging queries surface different results than exploration queries | Requires maintaining and passing conversation state, adding architectural complexity |
| Cross-encoder approach is well-studied and efficient for moderate candidate sets (<100 chunks) | Cross-encoders have a fixed context length that limits how much conversation history fits |
| LLM-as-judge handles nuanced, multi-turn context that cross-encoders cannot | Quality depends heavily on the summarization and entity extraction steps |

## When to Use

- Multi-turn conversations where the current query depends on earlier turns.
- Queries that frequently contain pronouns, abbreviations, or implicit references.
- Systems where users refine their questions iteratively (e.g., support bots, research assistants).
- When first-stage retrieval returns many relevant-looking candidates that need fine-grained ordering.
- When redundancy in retrieved context is a measurable problem (users seeing repeated information).

## When NOT to Use

- Single-turn, self-contained queries (e.g., search engines with no session).
- When first-stage retrieval already returns a small, high-quality set (less than 5 candidates).
- Extreme latency constraints where even 50ms of re-ranking overhead is unacceptable.
- When the conversation context is trivial (1-2 turns, no established entities or task state).

## Related Patterns

- **[Hybrid Search Fusion](hybrid-search-fusion.md)** -- Produces the initial candidate set that context-aware re-ranking refines. These two patterns compose naturally: fusion for broad recall, re-ranking for precision.
- **[RAG Context Assembly](rag-context-assembly.md)** -- After re-ranking, RAG Context Assembly handles token budgeting, deduplication, and source attribution for the final prompt.
- **[Just-in-Time Retrieval](just-in-time-retrieval.md)** -- Determines when retrieval (and therefore re-ranking) should fire in the conversation.

## Real-World Examples

- **Cohere Rerank API**: Cohere's Rerank endpoint takes a query and a list of documents, scoring relevance with a cross-encoder. Users enrich the query with conversation context to achieve context-aware re-ranking.
- **Perplexity AI**: Uses multi-stage retrieval where initial web results are re-ranked against the full conversation context before being fed to the generation model, producing answers that respect the conversation flow.
- **LangChain Contextual Compression**: LangChain's `ContextualCompressionRetriever` wraps a base retriever with a re-ranker that can use an LLM to score and filter documents based on the query context.
- **Amazon Q**: Amazon's enterprise AI assistant re-ranks retrieved internal documents using conversation history and the user's role/permissions, ensuring results are relevant to both the question and the organizational context.
- **Cursor IDE**: The AI code editor re-ranks retrieved code snippets based on the current file, recent edits, and conversation history, ensuring that code suggestions are contextually appropriate.
