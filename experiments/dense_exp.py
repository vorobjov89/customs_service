"""Эксперименты с плотными эмбеддингами"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import (
    load_jsonl,
    load_validation,
)

from src.evaluation import (
    calculate_metrics,
    print_metrics,
    calculate_rankings,
    calculate_validation_ranks
)

from src.text import (
    build_declaration_text,
    build_regulation_text,
)

from src.dense import (
    encode_texts
)


MODEL_NAME = "intfloat/multilingual-e5-large"


def run_experiment(
    name: str,
    model: SentenceTransformer,
    declaration_embeddings: np.ndarray,
    regulation_texts: list[str],
    declaration_ids: list[str],
    regulation_ids: list[str],
    validation: dict[str, str],
) -> dict[str, float]:
    "Запускаем эксперимент с перебором разных вариантов плотных эмбеддингов."

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    # Эмбеддинги регуляций
    print()
    print("Encoding regulations...")

    regulation_embeddings = encode_texts(
        model=model,
        texts=regulation_texts,
        prefix="passage",
    )

    print(
        "Regulation embeddings:",
        regulation_embeddings.shape,
    )

    # Считаем метреку сходства
    print()
    print("Calculating similarities...")

    similarity_matrix = (
        declaration_embeddings
        @ regulation_embeddings.T
    )

    print(
        "Similarity matrix:",
        similarity_matrix.shape,
    )

    # Ранжирование
    print()
    print("Calculating rankings...")

    rankings = calculate_rankings(
        similarity_matrix
    )

    # Ранжирование на валидации
    print()
    print(
        "Finding relevant regulation ranks..."
    )

    ranks = calculate_validation_ranks(
        rankings=rankings,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        validation=validation,
    )

    # Считаем метрики
    metrics = calculate_metrics(
        ranks
    )

    print()
    print_metrics(
        metrics
    )

    return metrics


def main():
    """Запускаем эксперименты."""

    parser = argparse.ArgumentParser(
        description="Dense retrieval experiments"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out/dense"),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Dense retrieval experiment")
    print("=" * 70)

    print()
    print(
        f"Model: {MODEL_NAME}"
    )

    # Загружаем данные
    print()
    print("Loading data...")

    declarations = load_jsonl(
        args.data / "declarations.jsonl"
    )

    regulations = load_jsonl(
        args.data / "regulations.jsonl"
    )

    validation = load_validation(
        args.data / "validation.json"
    )

    print(
        f"Declarations loaded: "
        f"{len(declarations)}"
    )

    print(
        f"Regulations loaded: "
        f"{len(regulations)}"
    )

    print(
        f"Validation size: "
        f"{len(validation)}"
    )

    # IDs
    declaration_ids = [
        item["declaration_id"]
        for item in declarations
    ]

    regulation_ids = [
        item["regulation_id"]
        for item in regulations
    ]

    # Тексты деклараций
    print()
    print("Preparing declaration texts...")

    declaration_texts = [
        build_declaration_text(
            item
        )
        for item in declarations
    ]

    # Загружаем модель
    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Model loaded.")

    # Эмбеддинги деклараций
    print()
    print("Encoding declarations...")

    declaration_embeddings = encode_texts(
        model=model,
        texts=declaration_texts,
        prefix="query",
    )

    print(
        "Declaration embeddings:",
        declaration_embeddings.shape,
    )

    # Эксперименты
    experiments = [
        (
            "e5_description",
            "description",
        ),
        (
            "e5_description_explanation",
            "description_explanation",
        ),
        (
            "e5_description_notes",
            "description_notes",
        ),
        (
            "e5_all",
            "all",
        ),
    ]

    results = []

    for name, variant in experiments:

        print()
        print(
            f"Preparing regulations: {name}"
        )

        regulation_texts = [
            build_regulation_text(
                item,
                variant=variant,
            )
            for item in regulations
        ]

        metrics = run_experiment(
            name=name,
            model=model,
            declaration_embeddings=(
                declaration_embeddings
            ),
            regulation_texts=regulation_texts,
            declaration_ids=declaration_ids,
            regulation_ids=regulation_ids,
            validation=validation,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

    # Сохраняем метрики
    results_df = pd.DataFrame(
        results
    )

    output_path = (
        args.out / "metrics.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # Выводим итог
    print()
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print(
        f"Saved metrics to {output_path}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
