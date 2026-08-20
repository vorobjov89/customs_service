"""Эксперименты с подбором лучшего разряженного эмбеддинга"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data import (
    load_jsonl,
    load_validation,
)

from src.evaluation import (
    calculate_metrics,
)

from src.text import (
    build_declaration_text,
    build_regulation_text,
    normalize_text,
)


def run_experiment(
    name: str,
    vectorizer: TfidfVectorizer,
    declaration_texts: list[str],
    regulation_texts: list[str],
    declaration_ids: list[str],
    regulation_ids: list[str],
    validation: dict[str, str],
    normalize: bool = False,
):
    """Run one TF-IDF retrieval experiment."""

    print()
    print(f"Running: {name}")

    # Нормализуем текст
    if normalize:
        declaration_texts = [
            normalize_text(text)
            for text in declaration_texts
        ]

        regulation_texts = [
            normalize_text(text)
            for text in regulation_texts
        ]

    # TF-IDF
    vectorizer.fit(
        declaration_texts + regulation_texts
    )

    declaration_matrix = vectorizer.transform(
        declaration_texts
    )

    regulation_matrix = vectorizer.transform(
        regulation_texts
    )

    print(
        "Declaration matrix:",
        declaration_matrix.shape,
    )

    print(
        "Regulation matrix:",
        regulation_matrix.shape,
    )

    # Считаем метрику сходства
    similarity = cosine_similarity(
        declaration_matrix,
        regulation_matrix,
    )

    # Строим индексы
    declaration_index = {
        declaration_id: i
        for i, declaration_id
        in enumerate(declaration_ids)
    }

    regulation_index = {
        regulation_id: i
        for i, regulation_id
        in enumerate(regulation_ids)
    }

    # Ранжирование
    rows = []
    ranks = []

    for declaration_id in validation:

        declaration_idx = declaration_index[
            declaration_id
        ]

        relevant_regulation_id = validation[
            declaration_id
        ]

        relevant_index = regulation_index[
            relevant_regulation_id
        ]

        scores = similarity[
            declaration_idx
        ]

        ranked_indices = np.argsort(
            -scores
        )

        # Ранжирование релевантных регуляций
        position = np.where(
            ranked_indices == relevant_index
        )[0]

        if len(position) == 0:
            rank = None
        else:
            rank = int(position[0]) + 1

        ranks.append(rank)

        # Top-10 предсказаний
        top_10 = ranked_indices[:10]

        for rank_number, regulation_idx in enumerate(
            top_10,
            start=1,
        ):
            regulation_id = regulation_ids[
                regulation_idx
            ]

            rows.append(
                {
                    "model": name,
                    "declaration_id": declaration_id,
                    "rank": rank_number,
                    "regulation_id": regulation_id,
                    "score": float(
                        scores[regulation_idx]
                    )
                }
            )

    # Метрики
    metrics = calculate_metrics(
        ranks
    )

    metrics = {
        "model": name,
        **metrics,
    }

    print()
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

    # Ранжирование рядов
    rank_rows = []

    for declaration_id, rank in zip(
        validation.keys(),
        ranks,
    ):
        rank_rows.append(
            {
                "model": name,
                "declaration_id": declaration_id,
                "relevant_regulation_id": validation[
                    declaration_id
                ],
                "rank": rank,
            }
        )

    return (
        metrics,
        rows,
        rank_rows,
    )


def main():
    """Запуск экспериментов с разряженными эмбеддингами"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("./data"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./out/sparse"),
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Загружаем данные
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

    # Экспериемнты
    experiments = [
        (
            "tfidf_word_1gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                ngram_range=(1, 1),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_word_1_2gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_word_1_3gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="word",
                ngram_range=(1, 3),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_3_5gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_3_7gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char",
                ngram_range=(3, 7),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_2_6gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char",
                ngram_range=(2, 6),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_wb_2_6gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(2, 6),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_wb_3_5gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_wb_3_7gram",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 7),
                min_df=1,
                sublinear_tf=True,
            ),
            False,
        ),

        (
            "tfidf_char_wb_3_5gram_normalized",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                sublinear_tf=True,
            ),
            True,
        ),
    ]

    # Запуск
    all_metrics = []
    all_predictions = []
    all_ranks = []

    for (
        name,
        vectorizer,
        normalize,
    ) in experiments:

        metrics, predictions, ranks = run_experiment(
            name=name,
            vectorizer=vectorizer,
            declaration_texts=declaration_texts,
            regulation_texts=regulation_texts,
            declaration_ids=declaration_ids,
            regulation_ids=regulation_ids,
            validation=validation,
            normalize=normalize,
        )

        all_metrics.append(metrics)
        all_predictions.extend(predictions)
        all_ranks.extend(ranks)

    # Сохраняем значения метрик
    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_path = (
        args.out / "metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # Сохраняем предсказания
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

    # Сохраняем ранжирование
    ranks_df = pd.DataFrame(
        all_ranks
    )

    ranks_path = (
        args.out / "ranks.csv"
    )

    ranks_df.to_csv(
        ranks_path,
        index=False,
    )

    # Выводим итог
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {ranks_path}")


if __name__ == "__main__":
    main()
