# Kan

Example code for comparing an MLP/DNN baseline, a lightweight KAN-style
piecewise additive classifier, and a decoded KAN-equation classifier on the
OpenML `house_prices` dataset.

The source dataset is regression-oriented (`SalePrice`). This project turns it
into a classification task by deriving three price bands from the training split:
`budget`, `standard`, and `premium`.

## Run Locally

```bash
pip install -r requirements.txt
python codes/run_all.py
pytest
```

Generated comparison results are written to `docs/result/` and mirrored to
`codes/web-showcase/comparison.json`.

## API Services

```bash
python codes/original/classify/app.py      # DNN on :8001
python codes/kan/classify/app.py           # KAN on :8002
python codes/kan-decode/classify/app.py    # KAN-equation on :8003
```

Each service exposes:

- `GET /health`
- `POST /predict` with either one JSON sample or `{ "samples": [...] }`

## Docker

```bash
docker compose up --build
```

Services:

- DNN API: `http://localhost:8001`
- KAN API: `http://localhost:8002`
- KAN-equation API: `http://localhost:8003`
- Static showcase: `http://localhost:8080`
