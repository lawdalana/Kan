from pathlib import Path

from codes.common import DEFAULT_FEATURES, load_house_prices
from codes.kan.model.train import KanClassifier, train_kan_model


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "house_prices.arff"


def test_kan_classifier_uses_pykan_spline_backend():
    rows = [
        {"OverallQual": 3, "GrLivArea": 900, "SalePrice": 90000},
        {"OverallQual": 4, "GrLivArea": 1100, "SalePrice": 120000},
        {"OverallQual": 6, "GrLivArea": 1500, "SalePrice": 180000},
        {"OverallQual": 7, "GrLivArea": 1900, "SalePrice": 230000},
        {"OverallQual": 9, "GrLivArea": 2800, "SalePrice": 420000},
        {"OverallQual": 10, "GrLivArea": 3200, "SalePrice": 600000},
    ]
    model = KanClassifier(
        features=["OverallQual", "GrLivArea"],
        hidden_width=2,
        grid=3,
        k=3,
        train_steps=2,
    ).fit(rows)

    low = model.predict_one({"OverallQual": 3, "GrLivArea": 950})
    high = model.predict_one({"OverallQual": 10, "GrLivArea": 3100})
    artifact = model.to_dict()

    assert artifact["backend"] == "pykan"
    assert artifact["grid"] == 3
    assert artifact["k"] == 3
    assert "act_fun.0.coef" in artifact["state_dict"]
    assert low["label"] in ["budget", "standard", "premium"]
    assert high["label"] in ["budget", "standard", "premium"]
    assert sum(high["probabilities"].values()) == 1.0


def test_train_kan_model_writes_json_artifact_and_result(tmp_path):
    model_path = tmp_path / "kan_model.json"
    result_path = tmp_path / "kan_result.json"

    result = train_kan_model(
        dataset_path=DATASET,
        model_path=model_path,
        result_path=result_path,
        limit_rows=240,
    )

    assert model_path.exists()
    assert result_path.exists()
    assert result["model"] == "KAN"
    assert result["features"] == DEFAULT_FEATURES
    assert 0.0 <= result["accuracy"] <= 1.0

    reloaded = KanClassifier.load(model_path)
    assert reloaded.train_steps > 100
    sample = load_house_prices(DATASET).rows[0]
    assert reloaded.predict_one(sample)["label"] in ["budget", "standard", "premium"]
