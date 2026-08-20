"""Финальное построение предсказаний."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import load_jsonl
from src.text import (
    build_declaration_text,
    build_regulation_text,
)
from src.sparse import calculate_tfidf_rankings
from src.dense import calculate_dense_rankings
from src.hybrid import calculate_rrf_scores


DENSE_MODEL_NAME = "intfloat/multilingual-e5-large"

RRF_K = 5
TOP_K = 10


def build_predictions(
    rrf_scores: np.ndarray,
    declaration_ids: list[str],
    regulation_ids: list[str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Формируем финальные top-k предсказания."""

    rows = []

    for declaration_idx, declaration_id in enumerate(
        declaration_ids
    ):
        scores = rrf_scores[declaration_idx]

        ranking = np.argsort(
            -scores
        )[:top_k]

        for rank, regulation_idx in enumerate(
            ranking,
            start=1,
        ):
            rows.append(
                {
                    "declaration_id": declaration_id,
                    "rank": rank,
                    "regulation_id": regulation_ids[
                        regulation_idx
                    ],
                    "score": float(
                        scores[regulation_idx]
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "declaration_id",
            "rank",
            "regulation_id",
            "score",
        ],
    )


def validate_predictions(
    predictions: pd.DataFrame,
    declaration_ids: list[str],
    regulation_ids: list[str],
    top_k: int = TOP_K,
) -> None:
    """Проверяем формат финальных предсказаний."""

    required_columns = [
        "declaration_id",
        "rank",
        "regulation_id",
        "score",
    ]

    if list(predictions.columns) != required_columns:
        raise ValueError(
            "Invalid prediction columns: "
            f"{list(predictions.columns)}"
        )

    expected_rows = (
        len(declaration_ids) * top_k
    )

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(predictions)}"
        )

    # Проверяем, что присутствуют все декларации
    prediction_declaration_ids = set(
        predictions["declaration_id"]
    )

    expected_declaration_ids = set(
        declaration_ids
    )

    if prediction_declaration_ids != (
        expected_declaration_ids
    ):
        missing = (
            expected_declaration_ids
            - prediction_declaration_ids
        )

        unknown = (
            prediction_declaration_ids
            - expected_declaration_ids
        )

        raise ValueError(
            "Declaration IDs mismatch. "
            f"Missing: {sorted(missing)}, "
            f"Unknown: {sorted(unknown)}"
        )

    # Проверяем количество предсказаний
    counts = (
        predictions
        .groupby("declaration_id")
        .size()
    )

    if not (counts == top_k).all():
        raise ValueError(
            "Each declaration must have "
            f"exactly {top_k} predictions"
        )

    # Проверяем каждую декларацию
    for declaration_id, group in (
        predictions.groupby("declaration_id")
    ):
        ranks = sorted(
            group["rank"].tolist()
        )

        if ranks != list(
            range(1, top_k + 1)
        ):
            raise ValueError(
                f"Invalid ranks for "
                f"{declaration_id}: {ranks}"
            )

        if (
            group["regulation_id"]
            .nunique()
            != top_k
        ):
            raise ValueError(
                f"Duplicate regulations for "
                f"{declaration_id}"
            )

        # Проверяем порядок score
        scores = group.sort_values(
            "rank"
        )["score"].to_numpy()

        if not np.all(
            scores[:-1] >= scores[1:]
        ):
            raise ValueError(
                f"Scores are not sorted "
                f"descending for {declaration_id}"
            )

    # Проверяем существование regulation_id
    unknown_regulations = (
        set(predictions["regulation_id"])
        - set(regulation_ids)
    )

    if unknown_regulations:
        raise ValueError(
            "Unknown regulation IDs: "
            f"{sorted(unknown_regulations)}"
        )

    # Проверяем score
    if predictions["score"].isna().any():
        raise ValueError(
            "Predictions contain NaN scores"
        )

    if not np.isfinite(
        predictions["score"]
    ).all():
        raise ValueError(
            "Predictions contain "
            "non-finite scores"
        )


def main(
    data_dir: Path,
    out_dir: Path,
) -> None:
    """Запуск финального hybrid retrieval."""

    print("=" * 80)
    print("FINAL HYBRID RETRIEVAL")
    print("=" * 80)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------

    print()
    print("Loading data...")

    declarations = load_jsonl(
        data_dir / "declarations.jsonl"
    )

    regulations = load_jsonl(
        data_dir / "regulations.jsonl"
    )

    print(
        f"Declarations: {len(declarations)}"
    )

    print(
        f"Regulations: {len(regulations)}"
    )

    declaration_ids = [
        declaration["declaration_id"]
        for declaration in declarations
    ]

    regulation_ids = [
        regulation["regulation_id"]
        for regulation in regulations
    ]

    # ------------------------------------------------------------------
    # Подготовка текстов
    # ------------------------------------------------------------------

    print()
    print("Preparing texts...")

    declaration_texts = [
        build_declaration_text(
            declaration
        )
        for declaration in declarations
    ]

    regulation_texts_sparse = [
        build_regulation_text(
            regulation
        )
        for regulation in regulations
    ]

    regulation_texts_dense = [
        build_regulation_text(
            regulation,
            "description_explanation",
        )
        for regulation in regulations
    ]

    # ------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------

    print()
    print("Calculating TF-IDF rankings...")

    sparse_rankings = calculate_tfidf_rankings(
        declaration_texts=declaration_texts,
        regulation_texts=regulation_texts_sparse,
    )

    # ------------------------------------------------------------------
    # E5
    # ------------------------------------------------------------------

    print()
    print(
        f"Loading model: {DENSE_MODEL_NAME}"
    )

    model = SentenceTransformer(
        DENSE_MODEL_NAME,
        local_files_only=True,
    )

    print("Calculating E5 rankings...")

    dense_rankings = calculate_dense_rankings(
        model=model,
        declaration_texts=declaration_texts,
        regulation_texts=regulation_texts_dense,
    )

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    print()
    print(
        f"Calculating RRF (k={RRF_K})..."
    )

    rrf_scores = calculate_rrf_scores(
        rankings_list=[
            sparse_rankings,
            dense_rankings,
        ],
        k=RRF_K,
    )

    # ------------------------------------------------------------------
    # Финальные предсказания
    # ------------------------------------------------------------------

    print()
    print(
        f"Building top-{TOP_K} predictions..."
    )

    predictions = build_predictions(
        rrf_scores=rrf_scores,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        top_k=TOP_K,
    )

    # ------------------------------------------------------------------
    # Проверка
    # ------------------------------------------------------------------

    print()
    print("Validating predictions...")

    validate_predictions(
        predictions=predictions,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        top_k=TOP_K,
    )

    print("Predictions validation: OK")

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------

    output_path = (
        out_dir / "predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Predictions: {len(predictions)}"
    )

    print(
        f"Saved to: {output_path}"
    )

    print()
    print("Preview:")

    print(
        predictions.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build final hybrid predictions"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out"),
    )

    args = parser.parse_args()

    main(
        data_dir=args.data,
        out_dir=args.out,
    )
