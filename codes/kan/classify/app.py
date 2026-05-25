from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.api_common import run_prediction_server
from codes.kan.model.train import MODEL_PATH, KanClassifier, train_kan_model


def create_predictor() -> KanClassifier:
    if not MODEL_PATH.exists():
        train_kan_model()
    return KanClassifier.load(MODEL_PATH)


def main() -> None:
    predictor = create_predictor()
    run_prediction_server("KAN", predictor.predict_many, default_port=8002)


if __name__ == "__main__":
    main()
