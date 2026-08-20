"""Проверка финального файла predictions.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "declaration_id",
    "rank",
    "regulation_id",
    "score",
}


def validate_predictions(
    predictions_path: Path,
) -> None:
    """Проверяет финальный predictions.csv."""

    print("=" * 80)
    print("PREDICTION VALIDATION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    print()
    print("Loading predictions...")

    predictions_path = predictions_path.resolve()

    print(
        f"Path: {predictions_path}"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: "
            f"{predictions_path}"
        )

    predictions = pd.read_csv(
        predictions_path,
        encoding="utf-8-sig",
    )

    print(
        f"Rows: {len(predictions)}"
    )

    # ------------------------------------------------------------------
    # Проверка колонок
    # ------------------------------------------------------------------

    print()
    print("Checking columns...")

    predictions.columns = [
        column.strip()
        for column in predictions.columns
    ]

    print("Columns found:")

    for column in predictions.columns:
        print(
            f"  {column!r}"
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing_columns)}\n"
            f"Actual columns: "
            f"{list(predictions.columns)}\n"
            f"File: "
            f"{predictions_path}"
        )

    print()
    print("Columns: OK")

    # ------------------------------------------------------------------
    # Проверка declaration_id
    # ------------------------------------------------------------------

    print()
    print("Checking declaration_id...")

    if predictions["declaration_id"].isna().any():
        raise ValueError(
            "Column 'declaration_id' contains NaN values"
        )

    print(
        f"Declarations: "
        f"{predictions['declaration_id'].nunique()}"
    )

    print(
        "Declaration IDs: OK"
    )

    # ------------------------------------------------------------------
    # Проверка regulation_id
    # ------------------------------------------------------------------

    print()
    print("Checking regulation_id...")

    if predictions["regulation_id"].isna().any():
        raise ValueError(
            "Column 'regulation_id' contains NaN values"
        )

    print(
        "Regulation IDs: OK"
    )

    # ------------------------------------------------------------------
    # Проверка score
    # ------------------------------------------------------------------

    print()
    print("Checking score...")

    predictions["score"] = pd.to_numeric(
        predictions["score"],
        errors="coerce",
    )

    if predictions["score"].isna().any():
        raise ValueError(
            "Column 'score' contains invalid "
            "or NaN values"
        )

    if not predictions["score"].map(
        lambda value: isinstance(
            value,
            (int, float),
        )
    ).all():
        raise ValueError(
            "Column 'score' contains non-numeric values"
        )

    print(
        "Scores: OK"
    )

    # ------------------------------------------------------------------
    # Проверка rank
    # ------------------------------------------------------------------

    print()
    print("Checking rank...")

    predictions["rank"] = pd.to_numeric(
        predictions["rank"],
        errors="coerce",
    )

    if predictions["rank"].isna().any():
        raise ValueError(
            "Column 'rank' contains invalid "
            "or NaN values"
        )

    if not (
        predictions["rank"]
        == predictions["rank"].astype(int)
    ).all():
        raise ValueError(
            "Rank must contain integer values"
        )

    print(
        "Rank values: OK"
    )

    # ------------------------------------------------------------------
    # Проверяем ровно 10 предсказаний
    # ------------------------------------------------------------------

    print()
    print(
        "Checking number of predictions "
        "per declaration..."
    )

    declaration_counts = (
        predictions
        .groupby("declaration_id")
        .size()
    )

    invalid_counts = declaration_counts[
        declaration_counts != 10
    ]

    if not invalid_counts.empty:
        print(
            "Invalid declaration counts:"
        )

        print(
            invalid_counts.to_string()
        )

        raise ValueError(
            "Each declaration must have exactly "
            "10 predictions"
        )

    print(
        "Exactly 10 predictions per declaration: OK"
    )

    # ------------------------------------------------------------------
    # Проверяем ранги 1-10
    # ------------------------------------------------------------------

    print()
    print(
        "Checking ranks 1-10..."
    )

    expected_ranks = set(
        range(1, 11)
    )

    for declaration_id, group in predictions.groupby(
        "declaration_id"
    ):
        ranks = set(
            group["rank"].astype(int)
        )

        if ranks != expected_ranks:
            raise ValueError(
                f"Invalid ranks for declaration "
                f"{declaration_id}: "
                f"{sorted(ranks)}"
            )

    print(
        "Ranks 1-10 for every declaration: OK"
    )

    # ------------------------------------------------------------------
    # Проверяем уникальность regulation_id
    # ------------------------------------------------------------------

    print()
    print(
        "Checking regulation uniqueness..."
    )

    duplicates = (
        predictions
        .groupby(
            [
                "declaration_id",
                "regulation_id",
            ]
        )
        .size()
    )

    duplicates = duplicates[
        duplicates > 1
    ]

    if not duplicates.empty:
        print(
            "Duplicate declaration/regulation pairs:"
        )

        print(
            duplicates.to_string()
        )

        raise ValueError(
            "Each declaration must contain "
            "10 unique regulations"
        )

    print(
        "Unique regulations per declaration: OK"
    )

    # ------------------------------------------------------------------
    # Проверяем соответствие rank и score
    # ------------------------------------------------------------------

    print()
    print(
        "Checking score ordering..."
    )

    for declaration_id, group in predictions.groupby(
        "declaration_id"
    ):
        group = group.sort_values(
            "rank"
        )

        scores = group[
            "score"
        ].to_numpy()

        # score должен убывать или оставаться
        # одинаковым при увеличении rank.
        if (
            scores[:-1] < scores[1:]
        ).any():
            raise ValueError(
                f"Scores are not sorted in "
                f"descending order for declaration "
                f"{declaration_id}"
            )

    print(
        "Scores sorted by descending relevance: OK"
    )

    # ------------------------------------------------------------------
    # Проверяем отсутствие лишних колонок?
    # ------------------------------------------------------------------
    #
    # Лишние колонки не запрещены заданием,
    # поэтому здесь специально ничего не делаем.
    #
    # Обязательными являются:
    #
    # declaration_id
    # rank
    # regulation_id
    # score
    #
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Итог
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION PASSED")
    print("=" * 80)

    print()
    print(
        f"File: {predictions_path}"
    )

    print(
        f"Declarations: "
        f"{predictions['declaration_id'].nunique()}"
    )

    print(
        f"Predictions: "
        f"{len(predictions)}"
    )

    print(
        "Predictions per declaration: 10"
    )

    print(
        "Ranks: 1-10"
    )

    print(
        "Regulations: unique per declaration"
    )

    print(
        "Score: numeric and sorted descending"
    )


def main():
    """Запуск валидатора."""

    parser = argparse.ArgumentParser(
        description="Validate predictions.csv"
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "./out/predictions.csv"
        ),
    )

    args = parser.parse_args()

    validate_predictions(
        args.predictions
    )


if __name__ == "__main__":
    main()
    