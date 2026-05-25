from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    split_rows,
    write_json,
)


MODEL_PATH = Path(__file__).with_name("dnn_model.joblib")
METADATA_PATH = Path(__file__).with_name("dnn_metadata.json")
RESULT_PATH = RESULT_DIR / "dnn_result.json"


def train_original_model(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = METADATA_PATH,
    result_path: str | Path = RESULT_PATH,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    dataset = load_house_prices(dataset_path)
    rows = dataset.rows[:limit_rows] if limit_rows else dataset.rows
    train_rows, test_rows = split_rows(rows)
    thresholds = compute_price_thresholds(train_rows)
    x_train, medians = extract_features(train_rows, DEFAULT_FEATURES)
    x_test, _ = extract_features(test_rows, DEFAULT_FEATURES, medians)
    y_train = labels_for_rows(train_rows, thresholds)
    y_test = labels_for_rows(test_rows, thresholds)
    label_to_index = {label: index for index, label in enumerate(CLASSES)}

    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.01,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42,
        ),
    )
    pipeline.fit(x_train, [label_to_index[label] for label in y_train])

    artifact = {
        "pipeline": pipeline,
        "features": DEFAULT_FEATURES,
        "medians": medians,
        "thresholds": thresholds,
        "classes": CLASSES,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, model_path)

    predictions, latency_ms = measure_latency_ms(lambda: predict_with_artifact(artifact, test_rows[:5]))
    predicted_labels = [prediction["label"] for prediction in predict_with_artifact(artifact, test_rows)]
    result = {
        "model": "DNN",
        "description": "MLPClassifier baseline over selected numeric house features.",
        "features": DEFAULT_FEATURES,
        "classes": CLASSES,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "accuracy": round(accuracy(y_test, predicted_labels), 4),
        "latency_ms": latency_ms,
        "sample_predictions": _sample_predictions(test_rows[:5], y_test[:5], predictions),
    }

    metadata = {
        "model": "DNN",
        "target": dataset.target,
        "features": DEFAULT_FEATURES,
        "classes": CLASSES,
        "price_thresholds": list(thresholds),
        "medians": medians,
    }
    write_json(metadata_path, metadata)
    write_json(result_path, result)
    return result


def load_original_artifact(model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    return load(Path(model_path))


def predict_with_artifact(
    artifact: dict[str, Any], samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    matrix, _ = extract_features(samples, artifact["features"], artifact["medians"])
    pipeline = artifact["pipeline"]
    labels = pipeline.predict(matrix)
    probabilities = pipeline.predict_proba(matrix)
    model_classes = list(pipeline.named_steps["mlpclassifier"].classes_)

    output: list[dict[str, Any]] = []
    for label, probability_row in zip(labels, probabilities):
        label_name = _label_name(label, artifact["classes"])
        probability_map = {
            class_name: _probability_for_class(probability_row, model_classes, class_index, class_name)
            for class_index, class_name in enumerate(CLASSES)
        }
        output.append({"label": label_name, "probabilities": probability_map})
    return output


def _label_name(label: Any, classes: list[str]) -> str:
    if isinstance(label, str) and label in classes:
        return label
    try:
        return classes[int(label)]
    except (TypeError, ValueError, IndexError):
        return str(label)


def _probability_for_class(
    probability_row: Any, model_classes: list[Any], class_index: int, class_name: str
) -> float:
    lookup = class_index if class_index in model_classes else class_name
    if lookup not in model_classes:
        return 0.0
    return round(float(probability_row[model_classes.index(lookup)]), 6)


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
    parser = argparse.ArgumentParser(description="Train and evaluate the DNN house price classifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    result = train_original_model(args.dataset, args.model, args.metadata, args.result)
    print(f"DNN accuracy={result['accuracy']} latency_ms={result['latency_ms']}")


if __name__ == "__main__":
    main()
