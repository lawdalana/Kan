import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHOWCASE = ROOT / "codes" / "web-showcase"


def test_static_showcase_has_required_assets_and_comparison_data():
    index = (SHOWCASE / "index.html").read_text()
    app = (SHOWCASE / "app.js").read_text()
    comparison = json.loads((SHOWCASE / "comparison.json").read_text())

    assert "House Price Model Comparison" in index
    assert 'styles.css?v=params-20260526' in index
    assert 'app.js?v=params-20260526' in index
    assert "latency" in app
    assert "Params" in app
    assert "parameter-pill" in app
    assert 'fetch("comparison.json", { cache: "no-store" })' in app
    assert [model["name"] for model in comparison["models"]] == ["DNN", "KAN", "KAN-equation"]
    assert all(model["latency_ms"] >= 0 for model in comparison["models"])
    assert all("parameter_count" in model for model in comparison["models"])
