"""Работа с разряженными эмбеддингами"""
from __future__ import annotations

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import normalize_text
from .evaluation import calculate_rankings


def calculate_tfidf_rankings(
    declaration_texts: list[str],
    regulation_texts: list[str],
) -> np.ndarray:
    """Получаем ранжирование tf-idf эмбеддиннгов по убыванию сходства"""

    declaration_texts = [
        normalize_text(text)
        for text in declaration_texts
    ]

    regulation_texts = [
        normalize_text(text)
        for text in regulation_texts
    ]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    vectorizer.fit(
        declaration_texts + regulation_texts
    )

    declaration_matrix = vectorizer.transform(
        declaration_texts
    )

    regulation_matrix = vectorizer.transform(
        regulation_texts
    )

    similarity = cosine_similarity(
        declaration_matrix,
        regulation_matrix,
    )

    return calculate_rankings(similarity)
