import json
from pathlib import Path

from joblib import load

from codes.original.model.train import train_original_model


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "dataset" / "house_prices.arff"


def test_train_original_model_writes_artifact_metadata_and_result(tmp_path):
    model_path = tmp_path / "dnn_model.joblib"
    metadata_path = tmp_path / "dnn_metadata.json"
    result_path = tmp_path / "dnn_result.json"

    result = train_original_model(
        dataset_path=DATASET,
        model_path=model_path,
        metadata_path=metadata_path,
        result_path=result_path,
        limit_rows=240,
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert result_path.exists()
    assert result["model"] == "DNN"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["latency_ms"] >= 0.0
    assert len(result["sample_predictions"]) == 5

    artifact = load(model_path)
    mlp = artifact["pipeline"].named_steps["mlpclassifier"]
    trainable_params = sum(weight.size for weight in mlp.coefs_) + sum(
        bias.size for bias in mlp.intercepts_
    )
    assert mlp.hidden_layer_sizes == (32, 16)
    assert mlp.max_iter > 100
    assert mlp.n_iter_ > 100
    assert trainable_params > 672
    assert result["parameter_count"] == trainable_params

    metadata = json.loads(metadata_path.read_text())
    assert metadata["target"] == "SalePrice"
    assert metadata["classes"] == ["budget", "standard", "premium"]
    assert metadata["parameter_count"] == trainable_params
