from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "dataset" / "house_prices.arff"
RESULT_DIR = ROOT / "docs" / "result"

TARGET = "SalePrice"
CLASSES = ["budget", "standard", "premium"]
DEFAULT_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
    "FullBath",
    "LotArea",
    "1stFlrSF",
    "2ndFlrSF",
]


@dataclass(frozen=True)
class Attribute:
    name: str
    kind: str


@dataclass(frozen=True)
class HouseDataset:
    attributes: list[Attribute]
    rows: list[dict[str, Any]]
    target: str = TARGET


def load_house_prices(path: str | Path = DEFAULT_DATASET_PATH) -> HouseDataset:
    attributes: list[Attribute] = []
    rows: list[dict[str, Any]] = []
    in_data = False

    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    raise ValueError(f"Invalid ARFF attribute line: {line}")
                attributes.append(Attribute(parts[1].strip("'\""), parts[2].strip().upper()))
                continue

            if lower.startswith("@data"):
                in_data = True
                continue

            if in_data:
                values = next(csv.reader([line], quotechar="'", skipinitialspace=False))
                if len(values) != len(attributes):
                    raise ValueError(
                        f"Expected {len(attributes)} values, got {len(values)} in row: {line[:120]}"
                    )
                rows.append(
                    {
                        attribute.name: _convert_arff_value(value, attribute.kind)
                        for attribute, value in zip(attributes, values)
                    }
                )

    if not attributes or not rows:
        raise ValueError(f"No ARFF data found in {path}")
    return HouseDataset(attributes=attributes, rows=rows)


def compute_price_thresholds(rows: list[dict[str, Any]]) -> tuple[float, float]:
    prices = sorted(float(row[TARGET]) for row in rows if row.get(TARGET) is not None)
    if len(prices) < 3:
        raise ValueError("At least three priced rows are required to build price bands")

    low_index = len(prices) // 3
    high_index = (len(prices) * 2) // 3
    return prices[low_index], prices[high_index]


def price_band(price: float | int, thresholds: tuple[float, float] | list[float]) -> str:
    low, high = thresholds
    value = float(price)
    if value <= low:
        return "budget"
    if value <= high:
        return "standard"
    return "premium"


def labels_for_rows(
    rows: list[dict[str, Any]], thresholds: tuple[float, float] | list[float]
) -> list[str]:
    return [price_band(row[TARGET], thresholds) for row in rows]


def extract_features(
    rows: list[dict[str, Any]],
    features: list[str] = DEFAULT_FEATURES,
    medians: dict[str, float] | None = None,
) -> tuple[list[list[float]], dict[str, float]]:
    resolved_medians = dict(medians) if medians is not None else _feature_medians(rows, features)
    matrix: list[list[float]] = []

    for row in rows:
        values: list[float] = []
        for feature in features:
            value = _as_float(row.get(feature))
            values.append(value if value is not None else resolved_medians.get(feature, 0.0))
        matrix.append(values)

    return matrix, resolved_medians


def split_rows(
    rows: list[dict[str, Any]], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, int(len(shuffled) * test_ratio))
    return shuffled[test_size:], shuffled[:test_size]


def selected_feature_snapshot(row: dict[str, Any], features: list[str] = DEFAULT_FEATURES) -> dict[str, Any]:
    return {feature: row.get(feature) for feature in features}


def accuracy(expected: list[str], predicted: list[str]) -> float:
    if not expected:
        return 0.0
    correct = sum(1 for left, right in zip(expected, predicted) if left == right)
    return correct / len(expected)


def softmax(scores: dict[str, float]) -> dict[str, float]:
    peak = max(scores.values())
    exps = {label: math.exp(score - peak) for label, score in scores.items()}
    total = sum(exps.values())
    probabilities: dict[str, float] = {}
    running = 0.0
    for label in CLASSES[:-1]:
        probabilities[label] = exps[label] / total
        running += probabilities[label]
    probabilities[CLASSES[-1]] = max(0.0, 1.0 - running)
    return probabilities


def measure_latency_ms(action: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = action()
    elapsed = (time.perf_counter() - start) * 1000
    return result, round(elapsed, 4)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _convert_arff_value(value: str, kind: str) -> Any:
    if value == "?":
        return None
    if kind in {"INTEGER", "REAL", "NUMERIC"}:
        number = float(value)
        return int(number) if kind == "INTEGER" and number.is_integer() else number
    return value


def _feature_medians(rows: list[dict[str, Any]], features: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for feature in features:
        values = sorted(
            value for row in rows if (value := _as_float(row.get(feature))) is not None
        )
        medians[feature] = _median(values) if values else 0.0
    return medians


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
