"""Эксперимет с TF-IDF"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_jsonl
from src.text import (
    build_declaration_text,
    build_regulation_text,
)


def validate_predictions(
    predictions: pd.DataFrame,
    declarations: list[dict],
    regulations: list[dict],
) -> None:
    """Проверяем вывод предсказаний."""

    declaration_ids = {
        declaration["declaration_id"]
        for declaration in declarations
    }

    regulation_ids = {
        regulation["regulation_id"]
        for regulation in regulations
    }

    assert len(predictions) == len(declarations) * 10

    assert set(predictions["declaration_id"]) == declaration_ids

    assert set(predictions["regulation_id"]).issubset(
        regulation_ids
    )

    for _, group in predictions.groupby(
        "declaration_id"
    ):
        assert len(group) == 10

        assert sorted(
            group["rank"].tolist()
        ) == list(range(1, 11))

        assert group["regulation_id"].nunique() == 10

        assert group["score"].notna().all()


def calculate_rankings(
    declaration_texts: list[str],
    regulation_texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Строим для TF-IDF ранжирование по метрике cosine similarity."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
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

    similarities = cosine_similarity(
        declaration_matrix,
        regulation_matrix,
    )

    rankings = np.argsort(
        -similarities,
        axis=1,
    )[:, :10]

    return similarities, rankings


def build_predictions(
    similarities: np.ndarray,
    rankings: np.ndarray,
    declarations: list[dict],
    regulations: list[dict],
) -> pd.DataFrame:
    """Выдаем таблицу топ-10 предсказаний."""

    rows = []

    for declaration_idx, declaration in enumerate(
        declarations
    ):
        for rank, regulation_idx in enumerate(
            rankings[declaration_idx],
            start=1,
        ):
            rows.append(
                {
                    "declaration_id": (
                        declaration["declaration_id"]
                    ),
                    "rank": rank,
                    "regulation_id": (
                        regulations[regulation_idx][
                            "regulation_id"
                        ]
                    ),
                    "score": float(
                        similarities[
                            declaration_idx,
                            regulation_idx,
                        ]
                    ),
                }
            )

    return pd.DataFrame(rows)


def main(
    data_dir: Path,
    output_dir: Path,
) -> None:
    """Запуск TF-IDF baseline."""

    declarations = load_jsonl(
        data_dir / "declarations.jsonl"
    )

    regulations = load_jsonl(
        data_dir / "regulations.jsonl"
    )

    declaration_texts = [
        build_declaration_text(
            declaration
        )
        for declaration in declarations
    ]

    regulation_texts = [
        build_regulation_text(
            regulation,
            variant="description",
        )
        for regulation in regulations
    ]

    similarities, rankings = calculate_rankings(
        declaration_texts=declaration_texts,
        regulation_texts=regulation_texts,
    )

    result = build_predictions(
        similarities=similarities,
        rankings=rankings,
        declarations=declarations,
        regulations=regulations,
    )

    validate_predictions(
        predictions=result,
        declarations=declarations,
        regulations=regulations,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main(
        data_dir=Path("data"),
        output_dir=Path("out"),
    )
