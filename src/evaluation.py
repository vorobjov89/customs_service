"""Вычисление метрик и ранжирование"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def calculate_rankings(
    similarity_matrix: np.ndarray,
) -> np.ndarray:
    """Возвращаем индексы по убыванию cosine similarity."""

    return np.argsort(
        -similarity_matrix,
        axis=1,
    )


def calculate_validation_ranks(
    rankings: np.ndarray,
    declaration_ids: list[str],
    regulation_ids: list[str],
    validation: dict[str, str],
) -> list[int]:
    """Ранжирование на валидации."""

    declaration_index = {
        declaration_id: index
        for index, declaration_id
        in enumerate(declaration_ids)
    }

    regulation_index = {
        regulation_id: index
        for index, regulation_id
        in enumerate(regulation_ids)
    }

    ranks = []

    for declaration_id, relevant_regulation_id in validation.items():

        if declaration_id not in declaration_index:
            raise ValueError(
                f"Unknown declaration_id: {declaration_id}"
            )

        if relevant_regulation_id not in regulation_index:
            raise ValueError(
                f"Unknown regulation_id: {relevant_regulation_id}"
            )

        declaration_idx = declaration_index[
            declaration_id
        ]

        regulation_idx = regulation_index[
            relevant_regulation_id
        ]

        ranking = rankings[declaration_idx]

        positions = np.where(
            ranking == regulation_idx
        )[0]

        if len(positions) == 0:
            rank = len(regulation_ids) + 1
        else:
            rank = int(positions[0]) + 1

        ranks.append(rank)

    return ranks


def calculate_metrics(
    ranks: Iterable[int],
) -> dict[str, float]:
    """
    Считаем метрики ранжирования.
    -------
    dict[str, float]
        Hit@1, Hit@5, Hit@10 and MRR@10.
    """

    ranks = list(ranks)

    if not ranks:
        raise ValueError(
            "Cannot calculate metrics for empty ranks."
        )

    n = len(ranks)

    hit_at_1 = sum(
        rank <= 1
        for rank in ranks
    ) / n

    hit_at_5 = sum(
        rank <= 5
        for rank in ranks
    ) / n

    hit_at_10 = sum(
        rank <= 10
        for rank in ranks
    ) / n

    mrr_at_10 = sum(
        1.0 / rank
        for rank in ranks
        if rank <= 10
    ) / n

    return {
        "Hit@1": hit_at_1,
        "Hit@5": hit_at_5,
        "Hit@10": hit_at_10,
        "MRR@10": mrr_at_10,
    }


def print_metrics(
    metrics: dict[str, float],
) -> None:
    """Вывод метрик в консоль."""

    print(
        f"Hit@1:  {metrics['Hit@1']:.4f}"
    )

    print(
        f"Hit@5:  {metrics['Hit@5']:.4f}"
    )

    print(
        f"Hit@10: {metrics['Hit@10']:.4f}"
    )

    print(
        f"MRR@10: {metrics['MRR@10']:.4f}"
    )


def evaluate_ranking(
    name: str,
    rankings,
    declaration_ids: list[str],
    regulation_ids: list[str],
    validation: dict[str, str],
):
    """
    Возвращаем ранжирование и метрики вместе.
    """

    ranks = calculate_validation_ranks(
        rankings=rankings,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        validation=validation,
    )

    metrics = calculate_metrics(
        ranks
    )

    print()
    print(name)

    print_metrics(
        metrics
    )

    return ranks, metrics
