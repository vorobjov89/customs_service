"""Работа с плотными эмбеддингами"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .evaluation import calculate_rankings


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    prefix: str,
) -> np.ndarray:
    """Кодируем текст эмбеддингами."""

    inputs = [
        f"{prefix}: {text}"
        for text in texts
    ]

    return model.encode(
        inputs,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def calculate_dense_rankings(
    model: SentenceTransformer,
    declaration_texts: list[str],
    regulation_texts: list[str],
) -> np.ndarray:
    """Ранжирование через сходство по убыванию."""

    declaration_embeddings = encode_texts(
        model=model,
        texts=declaration_texts,
        prefix="query",
    )

    regulation_embeddings = encode_texts(
        model=model,
        texts=regulation_texts,
        prefix="passage",
    )

    similarity = (
        declaration_embeddings
        @ regulation_embeddings.T
    )

    return calculate_rankings(similarity)
