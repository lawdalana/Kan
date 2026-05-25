from pathlib import Path

from codes.common import (
    DEFAULT_FEATURES,
    compute_price_thresholds,
    extract_features,
    load_house_prices,
    price_band,
)


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "house_prices.arff"


def test_load_house_prices_reads_arff_rows_and_schema():
    dataset = load_house_prices(DATASET)

    assert dataset.target == "SalePrice"
    assert len(dataset.attributes) == 81
    assert len(dataset.rows) == 1460
    assert dataset.rows[0]["OverallQual"] == 7
    assert dataset.rows[0]["SalePrice"] == 208500


def test_price_band_uses_training_thresholds():
    thresholds = compute_price_thresholds(
        [
            {"SalePrice": 100000},
            {"SalePrice": 150000},
            {"SalePrice": 200000},
            {"SalePrice": 300000},
            {"SalePrice": 450000},
            {"SalePrice": 700000},
        ]
    )

    assert thresholds == (200000.0, 450000.0)
    assert price_band(150000, thresholds) == "budget"
    assert price_band(300000, thresholds) == "standard"
    assert price_band(700000, thresholds) == "premium"


def test_extract_features_imputes_missing_numeric_values():
    rows = [
        {"OverallQual": 5, "GrLivArea": 1200, "GarageCars": None},
        {"OverallQual": 9, "GrLivArea": None, "GarageCars": 3},
        {"OverallQual": None, "GrLivArea": 2400, "GarageCars": 1},
    ]

    matrix, medians = extract_features(rows, DEFAULT_FEATURES)

    assert matrix == [
        [5.0, 1200.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [9.0, 1800.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [7.0, 2400.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert medians["OverallQual"] == 7.0
    assert medians["GrLivArea"] == 1800.0
    assert medians["GarageCars"] == 2.0
