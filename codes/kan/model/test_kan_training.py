from pathlib import Path

from codes.common import DEFAULT_FEATURES, load_house_prices
from codes.kan.model.train import KanClassifier, train_kan_model


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "house_prices.arff"


def test_kan_classifier_learns_piecewise_feature_logits():
    rows = [
        {"OverallQual": 3, "GrLivArea": 900, "SalePrice": 90000},
        {"OverallQual": 4, "GrLivArea": 1100, "SalePrice": 120000},
        {"OverallQual": 6, "GrLivArea": 1500, "SalePrice": 180000},
        {"OverallQual": 7, "GrLivArea": 1900, "SalePrice": 230000},
        {"OverallQual": 9, "GrLivArea": 2800, "SalePrice": 420000},
        {"OverallQual": 10, "GrLivArea": 3200, "SalePrice": 600000},
    ]
    model = KanClassifier(features=["OverallQual", "GrLivArea"], bins=3).fit(rows)

    low = model.predict_one({"OverallQual": 3, "GrLivArea": 950})
    high = model.predict_one({"OverallQual": 10, "GrLivArea": 3100})

    assert low["label"] == "budget"
    assert high["label"] == "premium"
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
    sample = load_house_prices(DATASET).rows[0]
    assert reloaded.predict_one(sample)["label"] in ["budget", "standard", "premium"]
