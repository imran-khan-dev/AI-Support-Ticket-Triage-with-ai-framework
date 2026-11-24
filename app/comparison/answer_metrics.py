"""Framework-independent answer-quality and faithfulness metrics.

All metrics are deterministic and require no external LLM calls. Scores are
in the range 0.0 to 1.0. A score of None signals that the metric could not
be computed (e.g. missing data), which is distinct from a genuine 0.0 score.

These metrics are identical for the custom and framework editions so that
answer quality can be compared without depending on internal implementation
details.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable, Mapping


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def _words(text: str) -> set[str]:
    return set(_normalize(text).split())


def _fact_coverage(answer: str, facts: Iterable[str]) -> float:
    """Fraction of expected facts found in the answer.

    Returns 0.0 when there are no expected facts.
    """
    facts = list(facts)
    if not facts:
        return 0.0
    answer_words = _words(answer)
    found = 0
    for fact in facts:
        fact_words = _words(fact)
        if fact_words and fact_words.issubset(answer_words):
            found += 1
    return found / len(facts)


def answer_correctness(
    answer: str | None,
    reference: str | None,
    expected_facts: Iterable[str] | None = None,
) -> float:
    """Answer correctness = blend of string similarity to reference and fact coverage.

    - 50% difflib.SequenceMatcher similarity to reference answer
    - 50% fraction of expected facts present in the answer

    Returns 0.0 when no answer is provided or when both reference and facts
    are missing.
    """
    if answer is None:
        return 0.0

    reference = reference or ""
    facts = list(expected_facts or [])

    if not reference and not facts:
        return 0.0

    sim_score = 0.0
    fact_score = 0.0
    has_reference = bool(reference.strip())
    has_facts = bool(facts)

    if has_reference:
        sim_score = difflib.SequenceMatcher(None, answer, reference).ratio()

    if has_facts:
        fact_score = _fact_coverage(answer, facts)

    if has_reference and has_facts:
        return (sim_score + fact_score) / 2.0
    if has_reference:
        return sim_score
    return fact_score


def answer_relevancy(
    answer: str | None,
    question: str | None,
    reference: str | None = None,
) -> float:
    """Answer relevancy = how well the answer addresses the question.

    Computed as the Jaccard similarity between question words and answer words,
    blended with string similarity to the reference answer when available.

    Returns 0.0 when no answer or question is provided.
    """
    if answer is None or not question:
        return 0.0

    question_words = _words(question)
    answer_words = _words(answer)

    if not question_words or not answer_words:
        return 0.0

    overlap = len(question_words & answer_words)
    union = len(question_words | answer_words)
    jaccard = overlap / union if union else 0.0

    if reference:
        sim = difflib.SequenceMatcher(None, answer, reference).ratio()
        return (jaccard + sim) / 2.0

    return jaccard


def faithfulness(
    answer: str | None,
    context_chunks: Iterable[str] | None,
) -> float:
    """Faithfulness = fraction of answer claims supported by the context.

    A claim is treated as supported if its significant words (nouns, verbs,
    numbers) appear in the combined context. Unsupported claims lower the score.

    Returns 0.0 when no answer is provided. Returns 1.0 when there is no
    context to check against (vacuously faithful - the answer makes no claims
    that can be contradicted by missing context). Returns None when context
    is explicitly empty but an answer exists (cannot verify faithfulness).
    """
    if answer is None:
        return 0.0

    chunks = list(context_chunks or [])
    context_text = " ".join(chunk for chunk in chunks if chunk)

    if not context_text.strip():
        # No context available: cannot verify faithfulness. Return None to
        # signal the metric is unavailable, distinct from a genuine score.
        return None  # type: ignore[return-value]

    context_words = _words(context_text)
    answer_words = _words(answer)

    if not answer_words:
        return 1.0

    # Significant words: filter out very short stop-words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "not", "no", "yes", "if", "then", "else", "when", "where",
        "how", "why", "what", "who", "which", "there", "here",
    }
    significant = {w for w in answer_words if len(w) > 2 and w not in stop_words}

    if not significant:
        return 1.0

    supported = sum(1 for w in significant if w in context_words)
    return supported / len(significant)


def context_precision(
    retrieved_chunks: Iterable[str] | None,
    expected_context_chunk_ids: Iterable[str] | None,
    chunk_id_to_text: Mapping[str, str] | None = None,
) -> float:
    """Context precision = fraction of retrieved chunks that are relevant.

    A chunk is relevant if its id is in the expected set. When chunk_id_to_text
    is provided, relevance is determined by matching expected fact keywords
    against chunk text instead.

    Returns 0.0 when no chunks were retrieved.
    """
    retrieved = list(retrieved_chunks or [])
    if not retrieved:
        return 0.0

    if expected_context_chunk_ids is not None:
        expected_ids = set(expected_context_chunk_ids)
        relevant = sum(1 for cid in retrieved if cid in expected_ids)
        return relevant / len(retrieved)

    return 0.0


def context_recall(
    retrieved_chunk_ids: Iterable[str] | None,
    expected_context_chunk_ids: Iterable[str] | None,
) -> float:
    """Context recall = fraction of expected context chunks that were retrieved.

    Returns 0.0 when there are no expected chunks.
    """
    expected = list(expected_context_chunk_ids or [])
    if not expected:
        return 0.0

    retrieved = set(retrieved_chunk_ids or [])
    found = sum(1 for cid in expected if cid in retrieved)
    return found / len(expected)


def unsupported_claims(
    answer: str | None,
    context_chunks: Iterable[str] | None,
) -> list[str]:
    """Return a list of answer sentences that appear unsupported by the context.

    A sentence is flagged as unsupported if none of its significant words
    appear in the combined context. This is a heuristic, not a guarantee.
    """
    if answer is None:
        return []

    chunks = list(context_chunks or [])
    context_text = " ".join(chunk for chunk in chunks if chunk)
    context_words = _words(context_text)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
        "their", "not", "no", "yes", "if", "then", "else", "when", "where",
        "how", "why", "what", "who", "which", "there", "here",
    }

    sentences = re.split(r"[.!?]+", answer)
    unsupported: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = {w for w in _words(sentence) if len(w) > 2 and w not in stop_words}
        if words and not any(w in context_words for w in words):
            unsupported.append(sentence)

    return unsupported


def average(values: list[float]) -> float:
    """Mean of values, ignoring None. Returns 0.0 for empty or all-None."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0.0
