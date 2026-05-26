window.COMPARISON_DATA = {
  "dataset": {
    "attributes": 81,
    "name": "OpenML 42165 house_prices",
    "rows": 1460,
    "target": "SalePrice",
    "task": "SalePrice classified into budget, standard, and premium bands"
  },
  "generated_at": "2026-05-26T01:24:35.586966+00:00",
  "models": [
    {
      "accuracy": 0.7705,
      "backend": "sklearn",
      "description": "MLPClassifier baseline over selected numeric house features.",
      "grid": null,
      "k": null,
      "latency_ms": 0.5216,
      "name": "DNN",
      "parameter_count": 899,
      "source_parameter_count": null,
      "test_rows": 292,
      "train_rows": 1168
    },
    {
      "accuracy": 0.8219,
      "backend": "pykan",
      "description": "pykan direct KAN classifier with B-spline edge activations.",
      "grid": 2,
      "k": 2,
      "latency_ms": 0.6009,
      "name": "KAN",
      "parameter_count": 270,
      "source_parameter_count": null,
      "test_rows": 292,
      "train_rows": 1168
    },
    {
      "accuracy": 0.8288,
      "backend": "pykan-symbolic-formula",
      "description": "pykan symbolic_formula exported with ex_round(..., 4).",
      "grid": 2,
      "k": 2,
      "latency_ms": 0.1503,
      "name": "KAN-equation",
      "parameter_count": 0,
      "source_parameter_count": 270,
      "test_rows": 292,
      "train_rows": 1168
    }
  ],
  "samples": [
    {
      "model": "DNN",
      "predictions": [
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 1470,
            "2ndFlrSF": 1160,
            "FullBath": 2,
            "GarageCars": 3,
            "GrLivArea": 2630,
            "LotArea": 9200,
            "OverallQual": 8,
            "TotalBsmtSF": 1470,
            "YearBuilt": 1998
          },
          "predicted": "premium",
          "probabilities": {
            "budget": 0.0,
            "premium": 0.999995,
            "standard": 5e-06
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 984,
            "2ndFlrSF": 620,
            "FullBath": 2,
            "GarageCars": 2,
            "GrLivArea": 1604,
            "LotArea": 10998,
            "OverallQual": 5,
            "TotalBsmtSF": 984,
            "YearBuilt": 1941
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.113064,
            "premium": 0.141605,
            "standard": 0.74533
          }
        },
        {
          "actual": "standard",
          "input": {
            "1stFlrSF": 1429,
            "2ndFlrSF": 0,
            "FullBath": 1,
            "GarageCars": 2,
            "GrLivArea": 1429,
            "LotArea": 14585,
            "OverallQual": 6,
            "TotalBsmtSF": 1144,
            "YearBuilt": 1960
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.015013,
            "premium": 0.00027,
            "standard": 0.984717
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 626,
            "2ndFlrSF": 591,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1217,
            "LotArea": 10762,
            "OverallQual": 6,
            "TotalBsmtSF": 626,
            "YearBuilt": 1980
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.072829,
            "premium": 0.0,
            "standard": 0.927171
          }
        },
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 901,
            "2ndFlrSF": 901,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1802,
            "LotArea": 7588,
            "OverallQual": 7,
            "TotalBsmtSF": 793,
            "YearBuilt": 1920
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.052003,
            "premium": 0.0,
            "standard": 0.947997
          }
        }
      ]
    },
    {
      "model": "KAN",
      "predictions": [
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 1470,
            "2ndFlrSF": 1160,
            "FullBath": 2,
            "GarageCars": 3,
            "GrLivArea": 2630,
            "LotArea": 9200,
            "OverallQual": 8,
            "TotalBsmtSF": 1470,
            "YearBuilt": 1998
          },
          "predicted": "premium",
          "probabilities": {
            "budget": 0.0,
            "premium": 0.99973,
            "standard": 0.00027
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 984,
            "2ndFlrSF": 620,
            "FullBath": 2,
            "GarageCars": 2,
            "GrLivArea": 1604,
            "LotArea": 10998,
            "OverallQual": 5,
            "TotalBsmtSF": 984,
            "YearBuilt": 1941
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.428535,
            "premium": 0.010754,
            "standard": 0.560711
          }
        },
        {
          "actual": "standard",
          "input": {
            "1stFlrSF": 1429,
            "2ndFlrSF": 0,
            "FullBath": 1,
            "GarageCars": 2,
            "GrLivArea": 1429,
            "LotArea": 14585,
            "OverallQual": 6,
            "TotalBsmtSF": 1144,
            "YearBuilt": 1960
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.105141,
            "premium": 0.056389,
            "standard": 0.83847
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 626,
            "2ndFlrSF": 591,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1217,
            "LotArea": 10762,
            "OverallQual": 6,
            "TotalBsmtSF": 626,
            "YearBuilt": 1980
          },
          "predicted": "budget",
          "probabilities": {
            "budget": 0.561687,
            "premium": 0.003187,
            "standard": 0.435126
          }
        },
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 901,
            "2ndFlrSF": 901,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1802,
            "LotArea": 7588,
            "OverallQual": 7,
            "TotalBsmtSF": 793,
            "YearBuilt": 1920
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.431252,
            "premium": 0.065736,
            "standard": 0.503012
          }
        }
      ]
    },
    {
      "model": "KAN-equation",
      "predictions": [
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 1470,
            "2ndFlrSF": 1160,
            "FullBath": 2,
            "GarageCars": 3,
            "GrLivArea": 2630,
            "LotArea": 9200,
            "OverallQual": 8,
            "TotalBsmtSF": 1470,
            "YearBuilt": 1998
          },
          "predicted": "premium",
          "probabilities": {
            "budget": 0.0,
            "premium": 0.999702,
            "standard": 0.000298
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 984,
            "2ndFlrSF": 620,
            "FullBath": 2,
            "GarageCars": 2,
            "GrLivArea": 1604,
            "LotArea": 10998,
            "OverallQual": 5,
            "TotalBsmtSF": 984,
            "YearBuilt": 1941
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.418558,
            "premium": 0.01013,
            "standard": 0.571312
          }
        },
        {
          "actual": "standard",
          "input": {
            "1stFlrSF": 1429,
            "2ndFlrSF": 0,
            "FullBath": 1,
            "GarageCars": 2,
            "GrLivArea": 1429,
            "LotArea": 14585,
            "OverallQual": 6,
            "TotalBsmtSF": 1144,
            "YearBuilt": 1960
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.103212,
            "premium": 0.057537,
            "standard": 0.839251
          }
        },
        {
          "actual": "budget",
          "input": {
            "1stFlrSF": 626,
            "2ndFlrSF": 591,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1217,
            "LotArea": 10762,
            "OverallQual": 6,
            "TotalBsmtSF": 626,
            "YearBuilt": 1980
          },
          "predicted": "budget",
          "probabilities": {
            "budget": 0.565146,
            "premium": 0.002836,
            "standard": 0.432018
          }
        },
        {
          "actual": "premium",
          "input": {
            "1stFlrSF": 901,
            "2ndFlrSF": 901,
            "FullBath": 1,
            "GarageCars": 1,
            "GrLivArea": 1802,
            "LotArea": 7588,
            "OverallQual": 7,
            "TotalBsmtSF": 793,
            "YearBuilt": 1920
          },
          "predicted": "standard",
          "probabilities": {
            "budget": 0.442521,
            "premium": 0.062555,
            "standard": 0.494924
          }
        }
      ]
    }
  ]
};
