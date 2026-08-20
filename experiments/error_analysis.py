"""Анализ ошибок моделей retrieval."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import (
    load_jsonl,
    load_validation,
)

from src.text import (
    build_declaration_text,
    build_regulation_text,
)


def build_declaration_index(
    declarations: list[dict],
) -> dict[str, str]:
    """Отображение declaration_id -> declaration text."""

    return {
        declaration["declaration_id"]:
        build_declaration_text(declaration)
        for declaration in declarations
    }


def build_regulation_index(
    regulations: list[dict],
) -> dict[str, str]:
    """Отображение regulation_id -> regulation text."""

    return {
        regulation["regulation_id"]:
        build_regulation_text(regulation)
        for regulation in regulations
    }


def determine_case(
    row: pd.Series,
) -> str:
    """
    Классификация результатов для одной декларации.

    Случаи:

        ALL_CORRECT:
            все модели попали на первое место;

        RRF_FIXED_BOTH:
            TF-IDF и E5 ошиблись,
            RRF попал на первое место;

        RRF_FIXED_TFIDF:
            TF-IDF ошибся,
            RRF попал на первое место;

        RRF_WORSE_THAN_TFIDF:
            TF-IDF попал на первое место,
            RRF — нет;

        NO_HIT_AT_1:
            ни одна модель не попала
            на первое место;

        OTHER:
            остальные комбинации.
    """

    tfidf_hit1 = bool(
        row["tfidf_hit1"]
    )

    e5_hit1 = bool(
        row["e5_hit1"]
    )

    rrf_hit1 = bool(
        row["rrf_hit1"]
    )

    if (
        tfidf_hit1
        and e5_hit1
        and rrf_hit1
    ):
        return "ALL_CORRECT"

    if (
        not tfidf_hit1
        and not e5_hit1
        and rrf_hit1
    ):
        return "RRF_FIXED_BOTH"

    if (
        not tfidf_hit1
        and rrf_hit1
    ):
        return "RRF_FIXED_TFIDF"

    if (
        tfidf_hit1
        and not rrf_hit1
    ):
        return "RRF_WORSE_THAN_TFIDF"

    if (
        not tfidf_hit1
        and not e5_hit1
        and not rrf_hit1
    ):
        return "NO_HIT_AT_1"

    return "OTHER"


def load_predictions(
    path: Path,
) -> pd.DataFrame:
    """Загружаем predictions.csv."""

    predictions = pd.read_csv(
        path
    )

    required_columns = {
        "model",
        "declaration_id",
        "rank",
        "regulation_id",
        "is_relevant",
    }

    missing = (
        required_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "Missing columns in predictions: "
            f"{sorted(missing)}"
        )

    return predictions


def validate_predictions(
    predictions: pd.DataFrame,
    validation: dict[str, str],
) -> None:
    """Проверяем корректность predictions."""

    expected_models = {
        "tfidf",
        "e5_large",
        "rrf_k5",
    }

    # Проверяем модели
    actual_models = set(
        predictions["model"].unique()
    )

    if actual_models != expected_models:
        raise ValueError(
            "Unexpected models in predictions: "
            f"{sorted(actual_models)}"
        )

    # Проверяем количество строк
    # на декларацию
    declaration_counts = (
        predictions
        .groupby("declaration_id")
        .size()
    )

    expected_count = (
        len(expected_models) * 10
    )

    invalid_declarations = (
        declaration_counts[
            declaration_counts
            != expected_count
        ]
    )

    if not invalid_declarations.empty:
        raise ValueError(
            "Invalid number of predictions "
            "for declarations:\n"
            f"{invalid_declarations}"
        )

    # Проверяем каждую модель
    for model_name in expected_models:

        model_predictions = predictions[
            predictions["model"]
            == model_name
        ]

        # Ровно 10 результатов
        # на декларацию
        counts = (
            model_predictions
            .groupby("declaration_id")
            .size()
        )

        invalid_counts = counts[
            counts != 10
        ]

        if not invalid_counts.empty:
            raise ValueError(
                f"Model {model_name} does not "
                "have exactly 10 predictions:\n"
                f"{invalid_counts}"
            )

        # Ранги должны быть 1..10
        invalid_ranks = (
            model_predictions
            .groupby("declaration_id")["rank"]
            .apply(
                lambda ranks:
                sorted(ranks.tolist())
                != list(range(1, 11))
            )
        )

        if invalid_ranks.any():

            declarations = (
                invalid_ranks[
                    invalid_ranks
                ]
                .index
                .tolist()
            )

            raise ValueError(
                f"Invalid ranks for model "
                f"{model_name}: "
                f"{declarations}"
            )

        # Regulation ID должны быть
        # уникальными
        unique_regulations = (
            model_predictions
            .groupby("declaration_id")[
                "regulation_id"
            ]
            .nunique()
        )

        invalid_duplicates = (
            unique_regulations[
                unique_regulations != 10
            ]
        )

        if not invalid_duplicates.empty:
            raise ValueError(
                f"Duplicate regulation_id "
                f"for model {model_name}:\n"
                f"{invalid_duplicates}"
            )

    # Проверяем is_relevant
    for (
        declaration_id,
        relevant_regulation_id,
    ) in validation.items():

        declaration_predictions = (
            predictions[
                predictions["declaration_id"]
                == declaration_id
            ]
        )

        if declaration_predictions.empty:
            raise ValueError(
                "Declaration from validation "
                "is missing in predictions: "
                f"{declaration_id}"
            )

        expected_relevant = (
            declaration_predictions[
                "regulation_id"
            ]
            == relevant_regulation_id
        )

        actual_relevant = (
            declaration_predictions[
                "is_relevant"
            ]
            .astype(bool)
        )

        if not expected_relevant.equals(
            actual_relevant
        ):
            raise ValueError(
                "Incorrect is_relevant values "
                f"for declaration "
                f"{declaration_id}"
            )

    print()
    print(
        "Predictions validation: OK"
    )


def build_comparison(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Строим одну строку для каждой декларации.

    Для каждой модели сохраняем:

        - rank релевантной регуляции;
        - Hit@1;
        - Hit@10.
    """

    rows = []

    for (
        declaration_id,
        group,
    ) in predictions.groupby(
        "declaration_id"
    ):

        result = {
            "declaration_id":
                declaration_id,
        }

        for model_name, prefix in [
            ("tfidf", "tfidf"),
            ("e5_large", "e5"),
            ("rrf_k5", "rrf"),
        ]:

            model_predictions = group[
                group["model"]
                == model_name
            ]

            relevant = model_predictions[
                model_predictions["is_relevant"]
            ]

            if relevant.empty:
                rank = None
            else:
                rank = int(
                    relevant.iloc[0]["rank"]
                )

            result[
                f"{prefix}_rank"
            ] = rank

            result[
                f"{prefix}_hit1"
            ] = (
                rank == 1
                if rank is not None
                else False
            )

            result[
                f"{prefix}_hit10"
            ] = (
                rank is not None
                and rank <= 10
            )

        rows.append(result)

    comparison = pd.DataFrame(
        rows
    )

    comparison["category"] = (
        comparison.apply(
            determine_case,
            axis=1,
        )
    )

    return comparison


def get_relevant_regulation(
    predictions: pd.DataFrame,
    declaration_id: str,
) -> str | None:
    """Получаем ID релевантной регуляции."""

    relevant = predictions[
        (
            predictions["declaration_id"]
            == declaration_id
        )
        & predictions["is_relevant"]
    ]

    if relevant.empty:
        return None

    return str(
        relevant.iloc[0]["regulation_id"]
    )


def build_report(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    declaration_texts: dict[str, str],
    regulation_texts: dict[str, str],
) -> str:
    """Создаём отчёт по ошибкам моделей."""

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        "HYBRID RETRIEVAL ERROR ANALYSIS"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    # ------------------------------------------------------------------
    # CASE COUNTS
    # ------------------------------------------------------------------

    lines.append(
        "CASE COUNTS"
    )

    lines.append(
        "-" * 80
    )

    category_counts = (
        comparison["category"]
        .value_counts()
    )

    for category, count in (
        category_counts.items()
    ):
        lines.append(
            f"{category:30} {count:>6}"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # AGGREGATE STATISTICS
    # ------------------------------------------------------------------

    total = len(
        comparison
    )

    lines.append(
        "AGGREGATE STATISTICS"
    )

    lines.append(
        "-" * 80
    )

    for column, name in [
        ("tfidf_hit1", "TF-IDF Hit@1"),
        ("e5_hit1", "E5 Hit@1"),
        ("rrf_hit1", "RRF Hit@1"),
        ("tfidf_hit10", "TF-IDF Hit@10"),
        ("e5_hit10", "E5 Hit@10"),
        ("rrf_hit10", "RRF Hit@10"),
    ]:

        value = (
            comparison[column].sum()
            / total
        )

        lines.append(
            f"{name:30} {value:.4f}"
        )

    lines.append("")

    # ------------------------------------------------------------------
    # INTERESTING CASES
    # ------------------------------------------------------------------

    interesting_categories = [
        "RRF_FIXED_BOTH",
        "RRF_FIXED_TFIDF",
        "RRF_WORSE_THAN_TFIDF",
        "NO_HIT_AT_1",
    ]

    for category in (
        interesting_categories
    ):

        subset = comparison[
            comparison["category"]
            == category
        ]

        if subset.empty:
            continue

        lines.append(
            "=" * 80
        )

        lines.append(
            category
        )

        lines.append(
            "=" * 80
        )

        for _, row in (
            subset.iterrows()
        ):

            declaration_id = str(
                row["declaration_id"]
            )

            relevant_regulation_id = (
                get_relevant_regulation(
                    predictions=predictions,
                    declaration_id=(
                        declaration_id
                    ),
                )
            )

            lines.append("")

            lines.append(
                f"Declaration: "
                f"{declaration_id}"
            )

            # ----------------------------------------------------------
            # RANKS
            # ----------------------------------------------------------

            lines.append("")

            lines.append(
                "Ranks:"
            )

            lines.append(
                f"  TF-IDF: "
                f"{row['tfidf_rank']}"
            )

            lines.append(
                f"  E5:     "
                f"{row['e5_rank']}"
            )

            lines.append(
                f"  RRF:    "
                f"{row['rrf_rank']}"
            )

            # ----------------------------------------------------------
            # DECLARATION
            # ----------------------------------------------------------

            declaration_text = (
                declaration_texts.get(
                    declaration_id,
                    "",
                )
            )

            lines.append("")

            lines.append(
                "Declaration text:"
            )

            lines.append(
                "-" * 80
            )

            lines.append(
                declaration_text
            )

            # ----------------------------------------------------------
            # RELEVANT REGULATION
            # ----------------------------------------------------------

            lines.append("")

            lines.append(
                "Relevant regulation:"
            )

            lines.append(
                "-" * 80
            )

            if (
                relevant_regulation_id
                is None
            ):
                lines.append(
                    "Relevant regulation "
                    "not found."
                )

            else:
                lines.append(
                    f"Regulation ID: "
                    f"{relevant_regulation_id}"
                )

                regulation_text = (
                    regulation_texts.get(
                        relevant_regulation_id,
                        "",
                    )
                )

                lines.append("")

                lines.append(
                    regulation_text
                )

            lines.append("")

            lines.append(
                "-" * 80
            )

    return "\n".join(
        lines
    )


def main() -> None:
    """Запускаем анализ ошибок моделей."""

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid retrieval error analysis"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out/hybrid"),
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "./out/hybrid/predictions.csv"
        ),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "HYBRID ERROR ANALYSIS"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------------

    print()
    print(
        "Loading predictions..."
    )

    predictions = load_predictions(
        args.predictions
    )

    print(
        f"Predictions loaded: "
        f"{len(predictions)}"
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    validation = load_validation(
        args.data / "validation.json"
    )

    validate_predictions(
        predictions=predictions,
        validation=validation,
    )

    # ------------------------------------------------------------------
    # DECLARATIONS
    # ------------------------------------------------------------------

    print()
    print(
        "Loading declarations..."
    )

    declarations = load_jsonl(
        args.data / "declarations.jsonl"
    )

    print(
        f"Declarations loaded: "
        f"{len(declarations)}"
    )

    # ------------------------------------------------------------------
    # REGULATIONS
    # ------------------------------------------------------------------

    print()
    print(
        "Loading regulations..."
    )

    regulations = load_jsonl(
        args.data / "regulations.jsonl"
    )

    print(
        f"Regulations loaded: "
        f"{len(regulations)}"
    )

    # ------------------------------------------------------------------
    # TEXT INDEXES
    # ------------------------------------------------------------------

    declaration_texts = (
        build_declaration_index(
            declarations
        )
    )

    regulation_texts = (
        build_regulation_index(
            regulations
        )
    )

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------

    print()
    print(
        "Building model comparison..."
    )

    comparison = build_comparison(
        predictions
    )

    comparison_path = (
        args.out
        / "error_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print()

    print(
        f"Saved comparison: "
        f"{comparison_path}"
    )

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    print()
    print(
        "Building report..."
    )

    report = build_report(
        comparison=comparison,
        predictions=predictions,
        declaration_texts=(
            declaration_texts
        ),
        regulation_texts=(
            regulation_texts
        ),
    )

    report_path = (
        args.out
        / "error_analysis.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"Saved report: "
        f"{report_path}"
    )

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    print()

    print(
        report
    )


if __name__ == "__main__":
    main()
    