import importlib.util
from pathlib import Path

from codes.kan.model.train import KanClassifier


DECODE_PATH = Path(__file__).with_name("decode_equation.py")
spec = importlib.util.spec_from_file_location("decode_equation", DECODE_PATH)
decode_equation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decode_equation)


def test_symbolic_equation_exports_ex_rounded_pykan_formulas(tmp_path):
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
    KanClassifier(
        features=["OverallQual", "GrLivArea"],
        hidden_width=2,
        grid=3,
        k=3,
        train_steps=2,
    ).fit(rows).save(kan_path)

    equation = decode_equation.decode_kan_model(
        kan_path, equation_path, text_path, calibration_rows=rows
    )

    predictor = decode_equation.EquationPredictor.from_file(equation_path)
    predictions = [predictor.predict_one(sample) for sample in rows]
    assert equation["model"] == "KAN-equation"
    assert equation["backend"] == "pykan-symbolic-formula"
    assert equation["formula_digits"] == 4
    assert equation["symbolic_weight_simple"] == 0.0
    assert len(equation["formula_strings"]) == len(equation["classes"])
    assert {prediction["label"] for prediction in predictions}.issubset(set(equation["classes"]))
    text = text_path.read_text()
    assert "ex_round(model.symbolic_formula()[0][0], 4)" in text
    assert "x_1 = scaled OverallQual" in text


def test_equation_predictor_evaluates_symbolic_formula_logits():
    equation = {
        "features": ["quality", "area"],
        "classes": ["budget", "standard", "premium"],
        "medians": {"quality": 5.0, "area": 1500.0},
        "minimums": {"quality": 0.0, "area": 1000.0},
        "maximums": {"quality": 10.0, "area": 2000.0},
        "formula_strings": ["-x_1", "x_2", "x_1 - x_2"],
    }

    predictor = decode_equation.EquationPredictor(equation)
    prediction = predictor.predict_one({"quality": 10, "area": 1000})

    assert prediction["label"] == "premium"
    assert set(prediction["probabilities"]) == {"budget", "standard", "premium"}
