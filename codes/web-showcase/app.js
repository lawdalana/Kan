const formatPercent = (value) => `${Math.round(value * 1000) / 10}%`;
const formatLatency = (value) => `${Number(value).toFixed(4)} ms`;

const tag = (label) => `<span class="tag ${label}">${label}</span>`;

async function loadComparison() {
  if (window.COMPARISON_DATA) {
    return window.COMPARISON_DATA;
  }
  const response = await fetch("comparison.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load comparison data: ${response.status}`);
  }
  return response.json();
}

function renderDataset(dataset) {
  document.querySelector("#dataset-rows").textContent = dataset.rows.toLocaleString();
  document.querySelector("#dataset-attributes").textContent = dataset.attributes.toString();
  document.querySelector("#dataset-target").textContent = dataset.target;
}

function renderMetrics(models) {
  const container = document.querySelector("#model-metrics");
  container.innerHTML = models
    .map(
      (model) => `
        <article class="metric-card">
          <h3>${model.name}</h3>
          <p>${model.description}</p>
          <div class="metric-row">
            <div>
              <span>Accuracy</span>
              <strong>${formatPercent(model.accuracy)}</strong>
            </div>
            <div>
              <span>Latency</span>
              <strong>${formatLatency(model.latency_ms)}</strong>
            </div>
          </div>
        </article>
      `,
    )
    .join("");
}

function renderChart(models) {
  const container = document.querySelector("#comparison-chart");
  const maxLatency = Math.max(...models.map((model) => model.latency_ms), 0.0001);
  container.innerHTML = models
    .map((model) => {
      const accuracyWidth = Math.max(4, model.accuracy * 100);
      const latencyWidth = Math.max(4, (model.latency_ms / maxLatency) * 100);
      return `
        <div class="chart-row">
          <div class="chart-label">${model.name}</div>
          <div class="bar-stack">
            <div class="bar-track" aria-label="${model.name} accuracy ${formatPercent(model.accuracy)}">
              <div class="bar" style="width: ${accuracyWidth}%"></div>
            </div>
            <div class="bar-track" aria-label="${model.name} latency ${formatLatency(model.latency_ms)}">
              <div class="bar latency" style="width: ${latencyWidth}%"></div>
            </div>
          </div>
          <div class="chart-value">${formatPercent(model.accuracy)} / ${formatLatency(model.latency_ms)}</div>
        </div>
      `;
    })
    .join("");
}

function renderSamples(samples) {
  const body = document.querySelector("#sample-body");
  body.innerHTML = samples
    .flatMap((group) =>
      group.predictions.map(
        (prediction) => `
          <tr>
            <td>${group.model}</td>
            <td>${prediction.input.OverallQual ?? "-"}</td>
            <td>${prediction.input.GrLivArea ?? "-"}</td>
            <td>${tag(prediction.actual)}</td>
            <td>${tag(prediction.predicted)}</td>
          </tr>
        `,
      ),
    )
    .join("");
}

loadComparison()
  .then((comparison) => {
    renderDataset(comparison.dataset);
    renderMetrics(comparison.models);
    renderChart(comparison.models);
    renderSamples(comparison.samples);
  })
  .catch((error) => {
    document.querySelector("#model-metrics").innerHTML = `<p>${error.message}</p>`;
  });
