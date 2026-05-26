from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy
import torch
from kan.utils import ex_round


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
    split_rows,
    write_json,
)
from codes.kan.model.train import MODEL_PATH as KAN_MODEL_PATH
from codes.kan.model.train import (
    KanClassifier,
    _build_pykan_model,
    _state_dict_from_json,
    train_kan_model,
)


EQUATION_PATH = Path(__file__).with_name("kan_equation.json")
EQUATION_TEXT_PATH = Path(__file__).with_name("kan_equation.txt")
RESULT_PATH = RESULT_DIR / "kan_equation_result.json"

BACKEND = "pykan-symbolic-formula"
DESCRIPTION = "pykan symbolic_formula exported with ex_round(..., 4)."
FORMULA_DIGITS = 4
SYMBOLIC_WEIGHT_SIMPLE = 0.0
SYMBOLIC_LIBRARY = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs"]


class EquationPredictor:
    def __init__(self, equation: dict[str, Any]) -> None:
        self.equation = equation
        self.features = list(equation["features"])
        self.classes = list(equation["classes"])
        self.medians = {key: float(value) for key, value in equation["medians"].items()}
        self.minimums = {key: float(value) for key, value in equation["minimums"].items()}
        self.maximums = {key: float(value) for key, value in equation["maximums"].items()}
        self.formula_strings = list(equation["formula_strings"])
        self._symbols = _formula_symbols(len(self.features))
        self._formula_functions = [
            _compile_formula(formula_string, self._symbols)
            for formula_string in self.formula_strings
        ]

    @classmethod
    def from_file(cls, path: str | Path) -> "EquationPredictor":
        return cls(read_json(path))

    def predict_one(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.predict_many([sample])[0]

    def predict_many(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not samples:
            return []

        matrix, _ = extract_features(samples, self.features, self.medians)
        activations = _scale_matrix(
            np.asarray(matrix, dtype=np.float64),
            self.features,
            self.minimums,
            self.maximums,
        )
        logits = _evaluate_formula_logits(self._formula_functions, activations)
        probabilities = _softmax_matrix(logits)
        label_indices = np.argmax(probabilities, axis=1).tolist()
        return [
            {
                "label": self.classes[int(label_index)],
                "probabilities": _probability_map(self.classes, probability_row.tolist()),
            }
            for label_index, probability_row in zip(label_indices, probabilities)
        ]

    def predict_many_scalar(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.predict_many(samples)


def decode_kan_model(
    kan_model_path: str | Path = KAN_MODEL_PATH,
    equation_path: str | Path = EQUATION_PATH,
    text_path: str | Path = EQUATION_TEXT_PATH,
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    calibration_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    kan = KanClassifier.load(kan_model_path)
    artifact = kan.to_dict()
    raw_formulas, rounded_formulas = _export_symbolic_formulas(
        artifact=artifact,
        dataset_path=dataset_path,
        calibration_rows=calibration_rows,
    )
    equation = {
        "backend": BACKEND,
        "model": "KAN-equation",
        "description": DESCRIPTION,
        "features": artifact["features"],
        "classes": artifact["classes"],
        "price_thresholds": artifact["price_thresholds"],
        "medians": artifact["medians"],
        "minimums": artifact["minimums"],
        "maximums": artifact["maximums"],
        "grid": artifact["grid"],
        "k": artifact["k"],
        "parameter_count": 0,
        "source_parameter_count": kan.parameter_count(),
        "symbolic_library": SYMBOLIC_LIBRARY,
        "symbolic_weight_simple": SYMBOLIC_WEIGHT_SIMPLE,
        "formula_digits": FORMULA_DIGITS,
        "raw_formula_strings": [str(formula) for formula in raw_formulas],
        "formula_strings": [str(formula) for formula in rounded_formulas],
    }
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

    predictions, latency_ms = measure_latency_ms(
        lambda: predictor.predict_many(test_rows[:5]), repeats=100, warmups=10
    )
    predicted_labels = [prediction["label"] for prediction in predictor.predict_many(test_rows)]
    result = {
        "model": "KAN-equation",
        "description": equation.get("description", DESCRIPTION),
        "backend": equation.get("backend", BACKEND),
        "features": equation["features"],
        "classes": equation.get("classes", CLASSES),
        "grid": equation["grid"],
        "k": equation["k"],
        "parameter_count": int(equation.get("parameter_count", 0)),
        "source_parameter_count": int(equation.get("source_parameter_count", 0)),
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
        "KAN-equation classifier exported from pykan symbolic_formula",
        "Export call: ex_round(model.symbolic_formula()[0][0], 4) for the first class logit, repeated per class.",
        "Formulas use scaled feature variables after min/max scaling to [-1, 1].",
        "Symbolic library: " + ", ".join(equation.get("symbolic_library", SYMBOLIC_LIBRARY)),
        "",
    ]
    for index, feature in enumerate(equation["features"], start=1):
        lines.append(f"x_{index} = scaled {feature}")

    lines.append("")
    for class_name, formula in zip(equation["classes"], equation["formula_strings"]):
        lines.append(f"{class_name} logit = {formula}")
    return "\n".join(lines) + "\n"


def _export_symbolic_formulas(
    artifact: dict[str, Any],
    dataset_path: str | Path | None,
    calibration_rows: list[dict[str, Any]] | None,
) -> tuple[list[Any], list[Any]]:
    model = _build_pykan_model(
        input_width=len(artifact["features"]),
        hidden_width=int(artifact["hidden_width"]),
        output_width=len(artifact["classes"]),
        grid=int(artifact["grid"]),
        k=int(artifact["k"]),
        seed=int(artifact["seed"]),
        symbolic_enabled=True,
        speed_mode=False,
    )
    model.load_state_dict(_state_dict_from_json(artifact["state_dict"]))
    model.eval()

    calibration_input = _calibration_input_tensor(artifact, dataset_path, calibration_rows)
    with torch.no_grad():
        model(calibration_input)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.auto_symbolic(
            lib=SYMBOLIC_LIBRARY,
            verbose=0,
            weight_simple=SYMBOLIC_WEIGHT_SIMPLE,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting a tensor with requires_grad=True to a scalar.*",
            category=UserWarning,
        )
        raw_formulas = list(model.symbolic_formula()[0])
    rounded_formulas = [ex_round(formula, FORMULA_DIGITS) for formula in raw_formulas]
    return raw_formulas, rounded_formulas


def _calibration_input_tensor(
    artifact: dict[str, Any],
    dataset_path: str | Path | None,
    calibration_rows: list[dict[str, Any]] | None,
) -> torch.Tensor:
    rows = calibration_rows
    if rows is None:
        if dataset_path is None:
            rows = _synthetic_calibration_rows(artifact)
        else:
            dataset = load_house_prices(dataset_path)
            rows, _ = split_rows(dataset.rows)

    matrix, _ = extract_features(rows, artifact["features"], artifact["medians"])
    scaled = _scale_matrix(
        np.asarray(matrix, dtype=np.float64),
        artifact["features"],
        artifact["minimums"],
        artifact["maximums"],
    )
    return torch.tensor(scaled, dtype=torch.float32)


def _synthetic_calibration_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position in np.linspace(0.0, 1.0, num=32):
        row: dict[str, Any] = {}
        for feature in artifact["features"]:
            low = float(artifact["minimums"][feature])
            high = float(artifact["maximums"][feature])
            row[feature] = low + (high - low) * float(position)
        rows.append(row)
    return rows


def _formula_symbols(width: int) -> tuple[sympy.Symbol, ...]:
    return tuple(sympy.Symbol(f"x_{index}") for index in range(1, width + 1))


def _compile_formula(
    formula_string: str, symbols: tuple[sympy.Symbol, ...]
) -> Callable[..., Any]:
    local_symbols = {str(symbol): symbol for symbol in symbols}
    expression = sympy.sympify(formula_string, locals=local_symbols)
    return sympy.lambdify(symbols, expression, modules="numpy")


def _evaluate_formula_logits(
    formula_functions: list[Callable[..., Any]], activations: np.ndarray
) -> np.ndarray:
    inputs = [activations[:, index] for index in range(activations.shape[1])]
    outputs: list[np.ndarray] = []
    with np.errstate(all="ignore"):
        for formula_function in formula_functions:
            value = formula_function(*inputs)
            if np.isscalar(value):
                value = np.full(activations.shape[0], float(value), dtype=np.float64)
            outputs.append(np.asarray(value, dtype=np.float64))

    logits = np.column_stack(outputs)
    return np.nan_to_num(logits, nan=0.0, posinf=1_000_000.0, neginf=-1_000_000.0)


def _scale_matrix(
    matrix: np.ndarray,
    features: list[str],
    minimums: dict[str, float],
    maximums: dict[str, float],
) -> np.ndarray:
    lows = np.asarray([minimums[feature] for feature in features], dtype=np.float64)
    highs = np.asarray([maximums[feature] for feature in features], dtype=np.float64)
    denominators = highs - lows
    scaled = np.divide(
        2 * (matrix - lows),
        denominators,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=denominators != 0.0,
    ) - 1
    return np.clip(scaled, -1.0, 1.0)


def _softmax_matrix(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exps = np.exp(shifted)
    totals = np.sum(exps, axis=1, keepdims=True)
    return np.divide(exps, totals, out=np.zeros_like(exps), where=totals != 0.0)


def _probability_map(classes: list[str], probabilities: list[float]) -> dict[str, float]:
    values: dict[str, float] = {}
    running = 0.0
    for index, class_name in enumerate(classes[:-1]):
        value = round(float(probabilities[index]), 6)
        values[class_name] = value
        running += value
    values[classes[-1]] = round(max(0.0, 1.0 - running), 6)
    return values


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and evaluate a pykan model as equations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--kan-model", type=Path, default=KAN_MODEL_PATH)
    parser.add_argument("--equation", type=Path, default=EQUATION_PATH)
    parser.add_argument("--text", type=Path, default=EQUATION_TEXT_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    if not args.kan_model.exists():
        train_kan_model(args.dataset, args.kan_model)
    decode_kan_model(args.kan_model, args.equation, args.text, args.dataset)
    result = evaluate_equation_model(args.dataset, args.equation, args.result)
    print(f"KAN-equation accuracy={result['accuracy']} latency_ms={result['latency_ms']}")


if __name__ == "__main__":
    main()
