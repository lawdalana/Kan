from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.common import (
    CLASSES,
    DEFAULT_DATASET_PATH,
    RESULT_DIR,
    accuracy,
    extract_features,
    labels_for_rows,
    load_house_prices,
    measure_latency_ms,
    read_json,
    selected_feature_snapshot,
    softmax,
    split_rows,
    write_json,
)
from codes.kan.model.train import MODEL_PATH as KAN_MODEL_PATH
from codes.kan.model.train import KanClassifier, train_kan_model


EQUATION_PATH = Path(__file__).with_name("kan_equation.json")
EQUATION_TEXT_PATH = Path(__file__).with_name("kan_equation.txt")
RESULT_PATH = RESULT_DIR / "kan_equation_result.json"


class EquationPredictor:
    def __init__(self, equation: dict[str, Any]) -> None:
        self.equation = equation

    @classmethod
    def from_file(cls, path: str | Path) -> "EquationPredictor":
        return cls(read_json(path))

    def predict_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        features = self.equation["features"]
        matrix, _ = extract_features([sample], features, self.equation["medians"])
        values = matrix[0]
        scores = dict(self.equation["priors"])

        for feature, value in zip(features, values):
            term = self.equation["terms"][feature]
            index = _bin_index(value, term["breakpoints"])
            for label, weight in term["weights"][index].items():
                scores[label] += float(weight)

        probabilities = softmax(scores)
        label = max(probabilities, key=probabilities.get)
        return {
            "label": label,
            "probabilities": {name: round(probabilities[name], 6) for name in self.equation["classes"]},
        }

    def predict_many(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.predict_one(sample) for sample in samples]


def decode_kan_model(
    kan_model_path: str | Path = KAN_MODEL_PATH,
    equation_path: str | Path = EQUATION_PATH,
    text_path: str | Path = EQUATION_TEXT_PATH,
) -> dict[str, Any]:
    kan = KanClassifier.load(kan_model_path)
    equation = kan.to_dict()
    equation["model"] = "KAN-equation"
    equation["description"] = "Decoded piecewise additive equation exported from the KAN artifact."
    write_json(equation_path, equation)
    Path(text_path).write_text(render_equation_text(equation), encoding="utf-8")
    return equation


def evaluate_equation_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    equation_path: str | Path = EQUATION_PATH,
    result_path: str | Path = RESULT_PATH,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    equation = read_json(equation_path)
    predictor = EquationPredictor(equation)
    dataset = load_house_prices(dataset_path)
    rows = dataset.rows[:limit_rows] if limit_rows else dataset.rows
    _, test_rows = split_rows(rows)
    expected = labels_for_rows(test_rows, equation["price_thresholds"])

    predictions, latency_ms = measure_latency_ms(lambda: predictor.predict_many(test_rows[:5]))
    predicted_labels = [prediction["label"] for prediction in predictor.predict_many(test_rows)]
    result = {
        "model": "KAN-equation",
        "description": "Pure Python equation evaluator decoded from the KAN piecewise functions.",
        "features": equation["features"],
        "classes": CLASSES,
        "train_rows": len(rows) - len(test_rows),
        "test_rows": len(test_rows),
        "accuracy": round(accuracy(expected, predicted_labels), 4),
        "latency_ms": latency_ms,
        "sample_predictions": _sample_predictions(test_rows[:5], expected[:5], predictions),
    }
    write_json(result_path, result)
    return result


def render_equation_text(equation: dict[str, Any]) -> str:
    lines = [
        "KAN-equation classifier",
        "score[class] = prior[class] + sum(piecewise_feature_weight[class])",
        "",
    ]
    for label, value in equation["priors"].items():
        lines.append(f"prior[{label}] = {value:.8f}")
    lines.append("")

    for feature in equation["features"]:
        term = equation["terms"][feature]
        breakpoints = term["breakpoints"]
        for index, weights in enumerate(term["weights"]):
            condition = _condition_text(feature, index, breakpoints)
            weight_text = ", ".join(
                f"{label}: {weights[label]:.8f}" for label in equation["classes"]
            )
            lines.append(f"if {condition} add {{{weight_text}}}")
    return "\n".join(lines) + "\n"


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


def _condition_text(feature: str, index: int, breakpoints: list[float]) -> str:
    if not breakpoints:
        return f"{feature} is any value"
    if index == 0:
        return f"{feature} <= {breakpoints[0]}"
    if index == len(breakpoints):
        return f"{feature} > {breakpoints[-1]}"
    return f"{breakpoints[index - 1]} < {feature} <= {breakpoints[index]}"


def _bin_index(value: float, breakpoints: list[float]) -> int:
    for index, breakpoint in enumerate(breakpoints):
        if value <= breakpoint:
            return index
    return len(breakpoints)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and evaluate a KAN model as an equation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--kan-model", type=Path, default=KAN_MODEL_PATH)
    parser.add_argument("--equation", type=Path, default=EQUATION_PATH)
    parser.add_argument("--text", type=Path, default=EQUATION_TEXT_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    if not args.kan_model.exists():
        train_kan_model(args.dataset, args.kan_model)
    decode_kan_model(args.kan_model, args.equation, args.text)
    result = evaluate_equation_model(args.dataset, args.equation, args.result)
    print(f"KAN-equation accuracy={result['accuracy']} latency_ms={result['latency_ms']}")


if __name__ == "__main__":
    main()
