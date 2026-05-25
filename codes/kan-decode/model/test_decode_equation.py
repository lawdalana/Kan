import importlib.util
from pathlib import Path

from codes.kan.model.train import KanClassifier


DECODE_PATH = Path(__file__).with_name("decode_equation.py")
spec = importlib.util.spec_from_file_location("decode_equation", DECODE_PATH)
decode_equation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_equation)


def test_decoded_equation_matches_kan_predictions(tmp_path):
    rows = [
        {"OverallQual": 3, "GrLivArea": 900, "SalePrice": 90000},
        {"OverallQual": 4, "GrLivArea": 1100, "SalePrice": 120000},
        {"OverallQual": 6, "GrLivArea": 1500, "SalePrice": 180000},
        {"OverallQual": 7, "GrLivArea": 1900, "SalePrice": 230000},
        {"OverallQual": 9, "GrLivArea": 2800, "SalePrice": 420000},
        {"OverallQual": 10, "GrLivArea": 3200, "SalePrice": 600000},
    ]
    kan_path = tmp_path / "kan_model.json"
    equation_path = tmp_path / "kan_equation.json"
    text_path = tmp_path / "kan_equation.txt"
    KanClassifier(features=["OverallQual", "GrLivArea"], bins=3).fit(rows).save(kan_path)

    equation = decode_equation.decode_kan_model(kan_path, equation_path, text_path)

    predictor = decode_equation.EquationPredictor.from_file(equation_path)
    for sample in rows:
        assert predictor.predict_one(sample)["label"] == KanClassifier.load(kan_path).predict_one(sample)["label"]
    assert equation["model"] == "KAN-equation"
    assert "if OverallQual" in text_path.read_text()
