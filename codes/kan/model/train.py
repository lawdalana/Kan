from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.common import (
    CLASSES,
    DEFAULT_DATASET_PATH,
    DEFAULT_FEATURES,
    RESULT_DIR,
    accuracy,
    compute_price_thresholds,
    extract_features,
    labels_for_rows,
    load_house_prices,
    measure_latency_ms,
    selected_feature_snapshot,
    softmax,
    split_rows,
    write_json,
    read_json,
)


MODEL_PATH = Path(__file__).with_name("kan_model.json")
RESULT_PATH = RESULT_DIR / "kan_result.json"


@dataclass(frozen=True)
class KanClassifier:
    features: list[str] | None = None
    bins: int = 5
    classes: list[str] | None = None
    price_thresholds: list[float] | None = None
    medians: dict[str, float] | None = None
    priors: dict[str, float] | None = None
    terms: dict[str, dict[str, Any]] | None = None

    def fit(self, rows: list[dict[str, Any]]) -> "KanClassifier":
        features = list(self.features or DEFAULT_FEATURES)
        classes = list(self.classes or CLASSES)
        price_thresholds = list(compute_price_thresholds(rows))
        labels = labels_for_rows(rows, price_thresholds)
        matrix, medians = extract_features(rows, features)

        label_counts = {label: labels.count(label) for label in classes}
        total = len(labels) + len(classes)
        priors = {
            label: math.log((label_counts[label] + 1) / total) for label in classes
        }
        terms: dict[str, dict[str, Any]] = {}

        for column, feature in enumerate(features):
            values = [row[column] for row in matrix]
            breakpoints = _quantile_breakpoints(values, self.bins)
            bin_counts = [
                {label: 0 for label in classes} for _ in range(len(breakpoints) + 1)
            ]
            for value, label in zip(values, labels):
                bin_counts[_bin_index(value, breakpoints)][label] += 1

            weights: list[dict[str, float]] = []
            for counts in bin_counts:
                bin_total = sum(counts.values()) + len(classes)
                weights.append(
                    {
                        label: round(
                            math.log((counts[label] + 1) / bin_total) - priors[label],
                            8,
                        )
                        for label in classes
                    }
                )

            terms[feature] = {"breakpoints": breakpoints, "weights": weights}
        return KanClassifier(
            features=features,
            bins=self.bins,
            classes=classes,
            price_thresholds=price_thresholds,
            medians=medians,
            priors=priors,
            terms=terms,
        )

    def predict_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not self.features or not self.medians or not self.priors or not self.terms:
            raise ValueError("KAN model has not been fitted")

        matrix, _ = extract_features([sample], self.features, self.medians)
        values = matrix[0]
        scores = dict(self.priors)
        for feature, value in zip(self.features, values):
            term = self.terms[feature]
            index = _bin_index(value, term["breakpoints"])
            for label, weight in term["weights"][index].items():
                scores[label] += weight

        probabilities = softmax(scores)
        label = max(probabilities, key=probabilities.get)
        return {
            "label": label,
            "probabilities": {name: round(probabilities[name], 6) for name in self.classes},
        }

    def predict_many(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(sample) for sample in samples]

    def to_dict(self) -> dict[str, Any]:
        if not self.features or not self.medians or not self.priors or not self.terms:
            raise ValueError("KAN model has not been fitted")
        return {
            "model": "KAN",
            "features": self.features,
            "bins": self.bins,
            "classes": self.classes or CLASSES,
            "price_thresholds": self.price_thresholds,
            "medians": self.medians,
            "priors": self.priors,
            "terms": self.terms,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KanClassifier":
        return cls(
            features=list(payload["features"]),
            bins=int(payload["bins"]),
            classes=list(payload["classes"]),
            price_thresholds=list(payload["price_thresholds"]),
            medians={key: float(value) for key, value in payload["medians"].items()},
            priors={key: float(value) for key, value in payload["priors"].items()},
            terms=payload["terms"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "KanClassifier":
        return cls.from_dict(read_json(path))


def train_kan_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    result_path: str | Path = RESULT_PATH,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    dataset = load_house_prices(dataset_path)
    rows = dataset.rows[:limit_rows] if limit_rows else dataset.rows
    train_rows, test_rows = split_rows(rows)
    model = KanClassifier(features=DEFAULT_FEATURES, bins=5).fit(train_rows)
    model.save(model_path)

    expected = labels_for_rows(test_rows, model.price_thresholds or [])
    predictions, latency_ms = measure_latency_ms(lambda: model.predict_many(test_rows[:5]))
    predicted_labels = [prediction["label"] for prediction in model.predict_many(test_rows)]
    result = {
        "model": "KAN",
        "description": "Additive univariate piecewise classifier inspired by KAN feature functions.",
        "features": DEFAULT_FEATURES,
        "classes": CLASSES,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "accuracy": round(accuracy(expected, predicted_labels), 4),
        "latency_ms": latency_ms,
        "sample_predictions": _sample_predictions(test_rows[:5], expected[:5], predictions),
    }
    write_json(result_path, result)
    return result


def _sample_predictions(
    rows: list[dict[str, Any]], expected: list[str], predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "input": selected_feature_snapshot(row),
            "actual": actual,
            "predicted": prediction["label"],
            "probabilities": prediction["probabilities"],
        }
        for row, actual, prediction in zip(rows, expected, predictions)
    ]


def _quantile_breakpoints(values: list[float], bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ordered = sorted(values)
    breakpoints: list[float] = []
    for index in range(1, bins):
        position = min(len(ordered) - 1, (len(ordered) * index) // bins)
        value = round(float(ordered[position]), 6)
        if not breakpoints or value > breakpoints[-1]:
            breakpoints.append(value)
    return breakpoints


def _bin_index(value: float, breakpoints: list[float]) -> int:
    for index, breakpoint in enumerate(breakpoints):
        if value <= breakpoint:
            return index
    return len(breakpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the KAN house price classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    result = train_kan_model(args.dataset, args.model, args.result)
    print(f"KAN accuracy={result['accuracy']} latency_ms={result['latency_ms']}")


if __name__ == "__main__":
    main()
