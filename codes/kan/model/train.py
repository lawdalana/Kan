from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from kan import KAN

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
    read_json,
    selected_feature_snapshot,
    split_rows,
    write_json,
)


MODEL_PATH = Path(__file__).with_name("kan_model.json")
RESULT_PATH = RESULT_DIR / "kan_result.json"


@dataclass(frozen=True)
class KanClassifier:
    features: list[str] | None = None
    hidden_width: int = 4
    grid: int = 5
    k: int = 3
    train_steps: int = 120
    learning_rate: float = 0.03
    batch_size: int = 256
    seed: int = 42
    classes: list[str] | None = None
    price_thresholds: list[float] | None = None
    medians: dict[str, float] | None = None
    minimums: dict[str, float] | None = None
    maximums: dict[str, float] | None = None
    state_dict: dict[str, Any] | None = None
    _runtime_model: Any = field(default=None, compare=False, repr=False)

    def fit(self, rows: list[dict[str, Any]]) -> "KanClassifier":
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.seed)
        features = list(self.features or DEFAULT_FEATURES)
        classes = list(self.classes or CLASSES)
        price_thresholds = list(compute_price_thresholds(rows))
        labels = labels_for_rows(rows, price_thresholds)
        matrix, medians = extract_features(rows, features)
        minimums, maximums = _feature_bounds(matrix, features)

        train_input = torch.tensor(
            _scale_matrix(matrix, features, minimums, maximums), dtype=torch.float32
        )
        label_to_index = {label: index for index, label in enumerate(classes)}
        train_label = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
        dataset = {
            "train_input": train_input,
            "train_label": train_label,
            "test_input": train_input,
            "test_label": train_label,
        }

        model = _build_pykan_model(
            input_width=len(features),
            hidden_width=self.hidden_width,
            output_width=len(classes),
            grid=self.grid,
            k=self.k,
            seed=self.seed,
        )
        _fit_pykan_model(
            model=model,
            dataset=dataset,
            train_steps=self.train_steps,
            learning_rate=self.learning_rate,
            batch_size=min(self.batch_size, len(rows)),
        )

        artifact = KanClassifier(
            features=features,
            hidden_width=self.hidden_width,
            grid=self.grid,
            k=self.k,
            train_steps=self.train_steps,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            seed=self.seed,
            classes=classes,
            price_thresholds=price_thresholds,
            medians=medians,
            minimums=minimums,
            maximums=maximums,
            state_dict=_state_dict_to_json(model.state_dict()),
        )
        object.__setattr__(artifact, "_runtime_model", model)
        return artifact

    def predict_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([sample])[0]

    def predict_many(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.features or not self.medians or not self.minimums or not self.maximums:
            raise ValueError("KAN model has not been fitted")

        matrix, _ = extract_features(samples, self.features, self.medians)
        scaled = _scale_matrix(matrix, self.features, self.minimums, self.maximums)
        inputs = torch.tensor(scaled, dtype=torch.float32)
        model = self._model()

        with torch.no_grad():
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()
            labels = torch.argmax(logits, dim=1).detach().cpu().tolist()

        classes = self.classes or CLASSES
        output: list[dict[str, Any]] = []
        for label_index, probability_row in zip(labels, probabilities):
            probability_map = _probability_map(classes, probability_row)
            output.append({"label": classes[int(label_index)], "probabilities": probability_map})
        return output

    def to_dict(self) -> dict[str, Any]:
        if not self.features or not self.medians or not self.minimums or not self.maximums:
            raise ValueError("KAN model has not been fitted")
        if self.state_dict is None:
            raise ValueError("KAN model state is missing")
        return {
            "backend": "pykan",
            "model": "KAN",
            "description": "pykan KAN classifier with cubic B-spline edge activations.",
            "features": self.features,
            "hidden_width": self.hidden_width,
            "width": [len(self.features), self.hidden_width, len(self.classes or CLASSES)],
            "grid": self.grid,
            "k": self.k,
            "train_steps": self.train_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "classes": self.classes or CLASSES,
            "price_thresholds": self.price_thresholds,
            "medians": self.medians,
            "minimums": self.minimums,
            "maximums": self.maximums,
            "state_dict": self.state_dict,
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KanClassifier":
        return cls(
            features=list(payload["features"]),
            hidden_width=int(payload["hidden_width"]),
            grid=int(payload["grid"]),
            k=int(payload["k"]),
            train_steps=int(payload.get("train_steps", 120)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            batch_size=int(payload.get("batch_size", 256)),
            seed=int(payload.get("seed", 42)),
            classes=list(payload["classes"]),
            price_thresholds=list(payload["price_thresholds"]),
            medians={key: float(value) for key, value in payload["medians"].items()},
            minimums={key: float(value) for key, value in payload["minimums"].items()},
            maximums={key: float(value) for key, value in payload["maximums"].items()},
            state_dict=payload["state_dict"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "KanClassifier":
        return cls.from_dict(read_json(path))

    def _model(self) -> Any:
        if self._runtime_model is not None:
            return self._runtime_model
        if not self.features or not self.classes or self.state_dict is None:
            raise ValueError("KAN model state is missing")
        model = _build_pykan_model(
            input_width=len(self.features),
            hidden_width=self.hidden_width,
            output_width=len(self.classes),
            grid=self.grid,
            k=self.k,
            seed=self.seed,
        )
        model.load_state_dict(_state_dict_from_json(self.state_dict))
        model.eval()
        object.__setattr__(self, "_runtime_model", model)
        return model


def train_kan_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    result_path: str | Path = RESULT_PATH,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    dataset = load_house_prices(dataset_path)
    rows = dataset.rows[:limit_rows] if limit_rows else dataset.rows
    train_rows, test_rows = split_rows(rows)
    model = KanClassifier(features=DEFAULT_FEATURES).fit(train_rows)
    model.save(model_path)

    expected = labels_for_rows(test_rows, model.price_thresholds or [])
    predictions, latency_ms = measure_latency_ms(
        lambda: model.predict_many(test_rows[:5]), repeats=100, warmups=10
    )
    predicted_labels = [prediction["label"] for prediction in model.predict_many(test_rows)]
    result = {
        "model": "KAN",
        "description": "pykan KAN classifier with cubic B-spline edge activations.",
        "backend": "pykan",
        "features": DEFAULT_FEATURES,
        "classes": CLASSES,
        "grid": model.grid,
        "k": model.k,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "accuracy": round(accuracy(expected, predicted_labels), 4),
        "latency_ms": latency_ms,
        "sample_predictions": _sample_predictions(test_rows[:5], expected[:5], predictions),
    }
    write_json(result_path, result)
    return result


def _build_pykan_model(
    input_width: int,
    hidden_width: int,
    output_width: int,
    grid: int,
    k: int,
    seed: int,
    symbolic_enabled: bool = False,
    speed_mode: bool = True,
) -> Any:
    model = KAN(
        width=[input_width, hidden_width, output_width],
        grid=grid,
        k=k,
        seed=seed,
        device="cpu",
        auto_save=False,
        symbolic_enabled=symbolic_enabled,
    )
    if speed_mode:
        model.speed()
    model.eval()
    return model


def _fit_pykan_model(
    model: Any,
    dataset: dict[str, torch.Tensor],
    train_steps: int,
    learning_rate: float,
    batch_size: int,
) -> None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(
            dataset,
            opt="Adam",
            steps=train_steps,
            lr=learning_rate,
            loss_fn=nn.CrossEntropyLoss(),
            batch=batch_size,
            log=train_steps + 1,
            update_grid=False,
        )
    model.eval()


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


def _feature_bounds(
    matrix: list[list[float]], features: list[str]
) -> tuple[dict[str, float], dict[str, float]]:
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = {}
    columns = list(zip(*matrix))
    for feature, values in zip(features, columns):
        minimums[feature] = float(min(values))
        maximums[feature] = float(max(values))
    return minimums, maximums


def _scale_matrix(
    matrix: list[list[float]],
    features: list[str],
    minimums: dict[str, float],
    maximums: dict[str, float],
) -> list[list[float]]:
    scaled: list[list[float]] = []
    for row in matrix:
        scaled_row: list[float] = []
        for feature, value in zip(features, row):
            low = minimums[feature]
            high = maximums[feature]
            if math.isclose(high, low):
                scaled_row.append(0.0)
                continue
            scaled_row.append(max(-1.0, min(1.0, 2 * (value - low) / (high - low) - 1)))
        scaled.append(scaled_row)
    return scaled


def _state_dict_to_json(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {key: value.detach().cpu().tolist() for key, value in state_dict.items()}


def _state_dict_from_json(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in payload.items()}


def _probability_map(classes: list[str], probability_row: list[float]) -> dict[str, float]:
    values: dict[str, float] = {}
    running = 0.0
    for index, class_name in enumerate(classes[:-1]):
        value = round(float(probability_row[index]), 6)
        values[class_name] = value
        running += value
    values[classes[-1]] = round(max(0.0, 1.0 - running), 6)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the pykan house price classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    result = train_kan_model(args.dataset, args.model, args.result)
    print(f"KAN accuracy={result['accuracy']} latency_ms={result['latency_ms']}")


if __name__ == "__main__":
    main()
