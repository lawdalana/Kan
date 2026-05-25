from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.api_common import run_prediction_server
from codes.kan.model.train import MODEL_PATH as KAN_MODEL_PATH
from codes.kan.model.train import train_kan_model


DECODE_PATH = ROOT / "codes" / "kan-decode" / "model" / "decode_equation.py"
spec = importlib.util.spec_from_file_location("decode_equation", DECODE_PATH)
decode_equation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_equation)


def create_predictor():
    if not KAN_MODEL_PATH.exists():
        train_kan_model()
    if not decode_equation.EQUATION_PATH.exists():
        decode_equation.decode_kan_model(KAN_MODEL_PATH)
    return decode_equation.EquationPredictor.from_file(decode_equation.EQUATION_PATH)


def main() -> None:
    predictor = create_predictor()
    run_prediction_server("KAN-equation", predictor.predict_many, default_port=8003)


if __name__ == "__main__":
    main()
