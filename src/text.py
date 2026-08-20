"""Работа с текстом"""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Нормализуем текст."""
    text = text.lower()
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_declaration_text(
    declaration: dict,
) -> str:
    """Получаем текстовые представления для деклараций."""
    parts = []

    description = declaration.get("G31_1")
    extension = declaration.get("desc_extention")

    if description:
        parts.append(str(description))

    if extension:
        parts.append(str(extension))

    return " ".join(parts)


def build_regulation_text(
    regulation: dict,
    variant: str = "description",
) -> str:
    """
    Получаем текстовые представления для регуляций.
    Возможны 4 варианта:
        "description": только описание
        "description_explanation": описание + объяснение
        "description_notes": описание + замечание
        "all": описание + объяснение + замечание

    """

    description = str(
        regulation.get("description") or ""
    )

    notes = str(
        regulation.get("notes") or ""
    )

    explanation = str(
        regulation.get("explanation") or ""
    )

    variants = {
        "description": [description],
        "description_explanation": [
            description,
            explanation,
        ],
        "description_notes": [
            description,
            notes,
        ],
        "all": [
            description,
            notes,
            explanation,
        ],
    }

    if variant not in variants:
        raise ValueError(
            f"Unknown regulation variant: {variant}"
        )

    return " ".join(
        part for part in variants[variant]
        if part
    )
