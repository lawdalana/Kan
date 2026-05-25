from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.api_common import run_prediction_server
from codes.original.model.train import MODEL_PATH, load_original_artifact, predict_with_artifact, train_original_model


def create_predictor() -> Any:
    if not MODEL_PATH.exists():
        train_original_model()
    artifact = load_original_artifact(MODEL_PATH)
    return lambda samples: predict_with_artifact(artifact, samples)


def main() -> None:
    run_prediction_server("DNN", create_predictor(), default_port=8001)


if __name__ == "__main__":
    main()
