"""Загрузка данных"""
from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Загрузка записей из JSON."""
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_validation(path: Path) -> dict[str, str]:
    """Декларация -> регуляция валидационное соответствие."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        return {
            item["declaration_id"]: item["regulation_id"]
            for item in data
        }

    raise ValueError(
        "Unsupported validation.json format"
    )
