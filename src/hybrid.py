"""Работа с гибридным поиском"""
from __future__ import annotations

import numpy as np


def build_predictions(
    rankings,
    declaration_ids: list[str],
    regulation_ids: list[str],
    validation: dict[str, str],
    model_name: str,
    top_k: int = 10,
) -> list[dict]:
    """Формируем top-k предсказаний модели."""

    rows = []

    for declaration_idx, declaration_id in enumerate(
        declaration_ids
    ):
        if declaration_id not in validation:
            continue

        ranking = rankings[
            declaration_idx
        ]

        relevant_regulation_id = validation[
            declaration_id
        ]

        for rank, regulation_idx in enumerate(
            ranking[:top_k],
            start=1,
        ):
            regulation_id = regulation_ids[
                regulation_idx
            ]

            rows.append(
                {
                    "model": model_name,
                    "declaration_id": declaration_id,
                    "rank": rank,
                    "regulation_id": regulation_id,
                    "is_relevant": (
                        regulation_id
                        == relevant_regulation_id
                    ),
                }
            )

    return rows


def reciprocal_rank_fusion(
    rankings_list: list[np.ndarray],
    k: int = 60,
) -> np.ndarray:
    """
    Reciprocal Rank Fusion.

    Для каждого документа:

        score(d) = sum(1 / (k + rank))

    где rank начинается с 1.

    Возвращает индексы документов
    в порядке убывания RRF score.
    """

    rrf_scores = calculate_rrf_scores(
        rankings_list=rankings_list,
        k=k,
    )

    return np.argsort(
        -rrf_scores,
        axis=1,
    )


def calculate_rrf_scores(
    rankings_list: list[np.ndarray],
    k: int = 60,
) -> np.ndarray:
    """
    Рассчитываем RRF score для каждого документа.

    Возвращает матрицу:

        [n_declarations, n_regulations]

    где scores[i, j] — RRF score
    регуляции j для декларации i.
    """

    if not rankings_list:
        raise ValueError(
            "rankings_list must contain at least one ranking"
        )

    first_ranking = rankings_list[0]

    if first_ranking.ndim != 2:
        raise ValueError(
            "Each ranking must be a 2-dimensional array"
        )

    n_declarations, n_regulations = (
        first_ranking.shape
    )

    for rankings in rankings_list:
        if rankings.shape != first_ranking.shape:
            raise ValueError(
                "All rankings must have the same shape"
            )

    rrf_scores = np.zeros(
        (
            n_declarations,
            n_regulations,
        ),
        dtype=np.float64,
    )

    for rankings in rankings_list:

        for declaration_idx in range(
            n_declarations
        ):
            ranking = rankings[
                declaration_idx
            ]

            for rank, regulation_idx in enumerate(
                ranking,
                start=1,
            ):
                rrf_scores[
                    declaration_idx,
                    regulation_idx,
                ] += 1.0 / (
                    k + rank
                )

    return rrf_scores
