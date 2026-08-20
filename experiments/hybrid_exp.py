"""Эксперименты с гибридным поиском"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data import (
    load_jsonl,
    load_validation
)

from src.text import (
    build_declaration_text,
    build_regulation_text,
)

from src.sparse import (
    calculate_tfidf_rankings,
)

from src.dense import (
    encode_texts,
    calculate_rankings
)

from src.evaluation import (
    evaluate_ranking
)

from src.hybrid import (
    reciprocal_rank_fusion,
    build_predictions
)


DENSE_MODEL_NAME = "intfloat/multilingual-e5-large"


def main():
    """Запускаем эксперименты"""

    print("=" * 80)
    print("HYBRID RETRIEVAL")
    print("TF-IDF + E5-large + RRF")
    print("=" * 80)

    # Загружаем данные
    print()
    print("Loading data...")

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
        default=Path("./out/hybrid"),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        f"Declarations: {len(declarations)}"
    )

    print(
        f"Regulations: {len(regulations)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    # IDs
    declaration_ids = [
        declaration["declaration_id"]
        for declaration in declarations
    ]

    regulation_ids = [
        regulation["regulation_id"]
        for regulation in regulations
    ]

    # Тексты
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
            "description_explanation"
        )
        for regulation in regulations
    ]

    # Sparse retrieval
    print()
    print("=" * 80)
    print("SPARSE RETRIEVAL")
    print("=" * 80)

    sparse_rankings = calculate_tfidf_rankings(
        declaration_texts=declaration_texts,
        regulation_texts=regulation_texts_sparse,
    )

    # Dense retrieval
    print()
    print("=" * 80)
    print("DENSE RETRIEVAL")
    print("=" * 80)

    print(
        f"Loading model: {DENSE_MODEL_NAME}"
    )

    model = SentenceTransformer(
        DENSE_MODEL_NAME
    )

    # Эмбеддинги деклараций
    declaration_embeddings = encode_texts(
        model=model,
        texts=declaration_texts,
        prefix="query",
    )

    # Эмбеддинги регуляций
    regulation_embeddings = encode_texts(
        model=model,
        texts=regulation_texts_dense,
        prefix="passage",
    )

    # Считаем метрики сходства
    dense_similarity = (
        declaration_embeddings
        @ regulation_embeddings.T
    )

    dense_rankings = calculate_rankings(
        dense_similarity
    )

    # Оцениваем модели по отдельности
    print()
    print("=" * 80)
    print("INDIVIDUAL MODELS")
    print("=" * 80)

    _, sparse_metrics = evaluate_ranking(
        name="TF-IDF",
        rankings=sparse_rankings,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        validation=validation,
    )

    _, dense_metrics = evaluate_ranking(
        name="E5-large",
        rankings=dense_rankings,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        validation=validation,
    )

    # RRF
    rrf_k = 5

    print()
    print("=" * 80)
    print(f"RRF (k={rrf_k})")
    print("=" * 80)

    rrf_rankings = reciprocal_rank_fusion(
        rankings_list=[
            sparse_rankings,
            dense_rankings,
        ],
        k=rrf_k,
    )

    all_predictions = []

    all_predictions.extend(
        build_predictions(
            rankings=sparse_rankings,
            declaration_ids=declaration_ids,
            regulation_ids=regulation_ids,
            validation=validation,
            model_name="tfidf",
        )
    )

    all_predictions.extend(
        build_predictions(
            rankings=dense_rankings,
            declaration_ids=declaration_ids,
            regulation_ids=regulation_ids,
            validation=validation,
            model_name="e5_large",
        )
    )

    all_predictions.extend(
        build_predictions(
            rankings=rrf_rankings,
            declaration_ids=declaration_ids,
            regulation_ids=regulation_ids,
            validation=validation,
            model_name=f"rrf_k{rrf_k}",
        )
    )

    _, rrf_metrics = evaluate_ranking(
        name=f"RRF (k={rrf_k})",
        rankings=rrf_rankings,
        declaration_ids=declaration_ids,
        regulation_ids=regulation_ids,
        validation=validation,
    )

    # Выводим итог
    results = [
        {
            "model": "TF-IDF",
            **sparse_metrics,
        },
        {
            "model": "E5-large",
            **dense_metrics,
        },
        {
            "model": f"RRF_k{rrf_k}",
            **rrf_metrics,
        },
    ]

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # Сохраняем

    predictions_df = pd.DataFrame(
        all_predictions
    )

    predictions_path = (
        args.out / "predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    print()
    print(
        f"Saved predictions to "
        f"{predictions_path}"
    )

    output_path = (
        args.out / "metrics.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved metrics to {output_path}"
    )


if __name__ == "__main__":
    main()
