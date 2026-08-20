"""Эксперимент с BM25"""
from pathlib import Path

import bm25s
import pandas as pd

from src.data import load_jsonl
from src.text import (
    build_declaration_text,
    build_regulation_text,
)


def build_predictions(
    declarations: list[dict],
    regulations: list[dict],
    retriever: bm25s.BM25,
) -> pd.DataFrame:
    """Run BM25 retrieval and build prediction table."""

    rows = []

    for declaration in declarations:
        declaration_id = declaration["declaration_id"]

        declaration_text = build_declaration_text(
            declaration
        )

        query_tokens = bm25s.tokenize(
            [declaration_text]
        )

        results, scores = retriever.retrieve(
            query_tokens,
            k=10,
        )

        for rank, (index, score) in enumerate(
            zip(results[0], scores[0]),
            start=1,
        ):
            regulation = regulations[int(index)]

            rows.append(
                {
                    "declaration_id": declaration_id,
                    "rank": rank,
                    "regulation_id": regulation[
                        "regulation_id"
                    ],
                    "score": float(score),
                }
            )

    return pd.DataFrame(rows)


def validate_predictions(
    predictions: pd.DataFrame,
    declarations: list[dict],
) -> None:
    """Validate prediction output format."""

    assert len(predictions) == (
        len(declarations) * 10
    )

    for _, group in predictions.groupby(
        "declaration_id"
    ):
        assert len(group) == 10

        assert group["rank"].tolist() == list(
            range(1, 11)
        )

        assert group["regulation_id"].nunique() == 10

        assert group["score"].notna().all()


def main(
    data_dir: Path,
    out_dir: Path,
) -> None:
    """Запускаем BM25 baseline."""

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Загружаем данные
    declarations = load_jsonl(
        data_dir / "declarations.jsonl"
    )

    regulations = load_jsonl(
        data_dir / "regulations.jsonl"
    )

    # Подготавливаем корпус регуляций
    regulation_texts = [
        build_regulation_text(
            regulation,
            variant="description",
        )
        for regulation in regulations
    ]

    corpus_tokens = bm25s.tokenize(
        regulation_texts
    )

    # Строим BM25 индекс
    retriever = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
    )

    retriever.index(corpus_tokens)

    # Извлекаем предсказания
    predictions = build_predictions(
        declarations=declarations,
        regulations=regulations,
        retriever=retriever,
    )

    # Валидация
    validate_predictions(
        predictions=predictions,
        declarations=declarations,
    )

    # Сохраняем вывод
    output_path = out_dir / "predictions.csv"

    predictions.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved predictions to {output_path}"
    )


if __name__ == "__main__":
    main(
        data_dir=Path("data"),
        out_dir=Path("out/bm25"),
    )
