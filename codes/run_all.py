from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.common import DEFAULT_DATASET_PATH, RESULT_DIR, load_house_prices, read_json, write_json
from codes.kan.model.train import MODEL_PATH as KAN_MODEL_PATH
from codes.kan.model.train import train_kan_model
from codes.original.model.train import train_original_model


DECODE_PATH = ROOT / "codes" / "kan-decode" / "model" / "decode_equation.py"
spec = importlib.util.spec_from_file_location("decode_equation", DECODE_PATH)
decode_equation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_equation)

WEB_COMPARISON_PATH = ROOT / "codes" / "web-showcase" / "comparison.json"
WEB_DATA_PATH = ROOT / "codes" / "web-showcase" / "comparison-data.js"
DOCS_COMPARISON_PATH = RESULT_DIR / "comparison.json"


def run_all(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    dnn_result = train_original_model(dataset_path=dataset_path)
    kan_result = train_kan_model(dataset_path=dataset_path)
    decode_equation.decode_kan_model(KAN_MODEL_PATH)
    equation_result = decode_equation.evaluate_equation_model(dataset_path=dataset_path)

    comparison = build_comparison([dnn_result, kan_result, equation_result], dataset_path)
    write_json(DOCS_COMPARISON_PATH, comparison)
    write_json(WEB_COMPARISON_PATH, comparison)
    WEB_DATA_PATH.write_text(
        "window.COMPARISON_DATA = "
        + json.dumps(comparison, indent=2, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )
    return comparison


def build_comparison(results: list[dict[str, Any]], dataset_path: str | Path) -> dict[str, Any]:
    dataset = load_house_prices(dataset_path)
    models = [
        {
            "name": result["model"],
            "description": result["description"],
            "backend": result.get("backend", "sklearn"),
            "grid": result.get("grid"),
            "k": result.get("k"),
            "parameter_count": result.get("parameter_count"),
            "source_parameter_count": result.get("source_parameter_count"),
            "accuracy": result["accuracy"],
            "latency_ms": result["latency_ms"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
        }
        for result in results
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "OpenML 42165 house_prices",
            "rows": len(dataset.rows),
            "attributes": len(dataset.attributes),
            "target": dataset.target,
            "task": "SalePrice classified into budget, standard, and premium bands",
        },
        "models": models,
        "samples": [
            {
                "model": result["model"],
                "predictions": result["sample_predictions"],
            }
            for result in results
        ],
    }


def main() -> None:
    comparison = run_all()
    print(f"Wrote {DOCS_COMPARISON_PATH}")
    print(f"Wrote {WEB_COMPARISON_PATH}")
    print(f"Wrote {WEB_DATA_PATH}")
    for model in comparison["models"]:
        print(
            f"{model['name']}: accuracy={model['accuracy']} latency_ms={model['latency_ms']}"
        )


if __name__ == "__main__":
    main()
