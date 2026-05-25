import json

from codes.api_common import parse_prediction_body, success_response


def test_parse_prediction_body_accepts_one_sample_or_many_samples():
    one = parse_prediction_body(json.dumps({"OverallQual": 7, "GrLivArea": 1700}).encode())
    many = parse_prediction_body(
        json.dumps({"samples": [{"OverallQual": 5}, {"OverallQual": 9}]}).encode()
    )

    assert one == [{"OverallQual": 7, "GrLivArea": 1700}]
    assert many == [{"OverallQual": 5}, {"OverallQual": 9}]


def test_success_response_uses_consistent_api_envelope():
    payload = success_response({"predictions": []})

    assert payload == {"success": True, "data": {"predictions": []}}
