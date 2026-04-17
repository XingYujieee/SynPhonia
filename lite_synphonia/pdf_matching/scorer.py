from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


@dataclass(frozen=True)
class PageScore:
    page_index: int
    score: float
    max_score: float
    mean_top_score: float
    chunk_hits: int
    dense_score: float = 0.0
    lexical_score: float = 0.0


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]*")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DEFAULT_LEXICAL_WEIGHT = 0.22

_LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for", "from",
    "how", "in", "into", "is", "it", "its", "may", "not", "of", "on", "or", "our",
    "that", "the", "their", "there", "these", "this", "to", "today", "was", "we",
    "with", "you",
}


def cosine_similarity_matrix(query_embeddings: np.ndarray, passage_embeddings: np.ndarray) -> np.ndarray:
    if query_embeddings.size == 0 or passage_embeddings.size == 0:
        return np.zeros((len(query_embeddings), len(passage_embeddings)), dtype=np.float32)

    query = np.asarray(query_embeddings, dtype=np.float32)
    passage = np.asarray(passage_embeddings, dtype=np.float32)

    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    passage_norms = np.linalg.norm(passage, axis=1, keepdims=True)
    query_safe = np.divide(
        query,
        np.maximum(query_norms, 1e-8),
        out=np.zeros_like(query),
        where=query_norms > 1e-8,
    )
    passage_safe = np.divide(
        passage,
        np.maximum(passage_norms, 1e-8),
        out=np.zeros_like(passage),
        where=passage_norms > 1e-8,
    )

    return np.matmul(query_safe, passage_safe.T)


def aggregate_page_scores(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_page_indexes: list[int],
    page_count: int,
    *,
    query_texts: list[str] | None = None,
    page_texts: list[str] | None = None,
    lexical_weight: float = _DEFAULT_LEXICAL_WEIGHT,
) -> list[PageScore]:
    if page_count <= 0:
        return []

    if chunk_embeddings.size == 0:
        return [
            PageScore(
                page_index=page_index,
                score=-1.0,
                max_score=-1.0,
                mean_top_score=-1.0,
                chunk_hits=0,
                dense_score=-1.0,
                lexical_score=0.0,
            )
            for page_index in range(page_count)
        ]

    similarities = cosine_similarity_matrix(query_embedding[None, :], chunk_embeddings)[0]
    scores_by_page: dict[int, list[float]] = {page_index: [] for page_index in range(page_count)}
    for index, score in enumerate(similarities):
        scores_by_page[chunk_page_indexes[index]].append(float(score))

    aggregated: list[PageScore] = []
    lexical_scores = _lexical_page_scores(
        query_texts or [],
        page_texts or [],
        page_count,
    )
    for page_index in range(page_count):
        page_scores = sorted(scores_by_page.get(page_index, []), reverse=True)
        if not page_scores:
            aggregated.append(
                PageScore(
                    page_index=page_index,
                    score=-1.0,
                    max_score=-1.0,
                    mean_top_score=-1.0,
                    chunk_hits=0,
                    dense_score=-1.0,
                    lexical_score=0.0,
                )
            )
            continue

        max_score = page_scores[0]
        top_k = page_scores[: min(5, len(page_scores))]
        mean_top_score = sum(top_k) / len(top_k)
        dense_score = max_score * 0.70 + mean_top_score * 0.30
        lexical_score = lexical_scores[page_index]
        aggregated.append(
            PageScore(
                page_index=page_index,
                score=dense_score + lexical_weight * lexical_score,
                max_score=max_score,
                mean_top_score=mean_top_score,
                chunk_hits=len(page_scores),
                dense_score=dense_score,
                lexical_score=lexical_score,
            )
        )

    aggregated.sort(key=lambda item: (item.score, item.max_score), reverse=True)
    return aggregated


def _lexical_page_scores(
    query_texts: list[str],
    page_texts: list[str],
    page_count: int,
) -> list[float]:
    if not query_texts or not page_texts:
        return [0.0] * page_count

    page_token_sets = [set(_extract_lexical_tokens(text)) for text in page_texts[:page_count]]
    if len(page_token_sets) < page_count:
        page_token_sets.extend([set()] * (page_count - len(page_token_sets)))

    doc_frequency: dict[str, int] = {}
    for token_set in page_token_sets:
        for token in token_set:
            doc_frequency[token] = doc_frequency.get(token, 0) + 1

    query_tokens = set()
    for text in query_texts:
        query_tokens.update(_extract_lexical_tokens(text))
    query_tokens = {token for token in query_tokens if token in doc_frequency}
    if not query_tokens:
        return [0.0] * page_count

    weights = {
        token: 1.0 + math.log((1.0 + page_count) / (1.0 + doc_frequency[token]))
        for token in query_tokens
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return [0.0] * page_count

    scores: list[float] = []
    for token_set in page_token_sets:
        overlap_weight = sum(weights[token] for token in query_tokens if token in token_set)
        scores.append(overlap_weight / total_weight)
    return scores


def _extract_lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []

    for raw_token in _TOKEN_PATTERN.findall(text):
        token = raw_token.lower().strip("._-")
        if len(token) < 2:
            continue
        if token in _LEXICAL_STOPWORDS:
            continue
        tokens.append(token)

    cjk_chars = _CJK_PATTERN.findall(text)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    return tokens
