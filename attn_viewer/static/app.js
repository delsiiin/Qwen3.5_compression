const state = {
  runs: [],
  run: null,
  sample: null,
  matrix: null,
  queryIndex: 0,
  valueMin: 0,
  valueMax: 1,
};

const elements = {
  runSelect: document.getElementById("runSelect"),
  sampleSelect: document.getElementById("sampleSelect"),
  prefillSelect: document.getElementById("prefillSelect"),
  layerSelect: document.getElementById("layerSelect"),
  headSelect: document.getElementById("headSelect"),
  minInput: document.getElementById("minInput"),
  maxInput: document.getElementById("maxInput"),
  queryTokenDisplay: document.getElementById("queryTokenDisplay"),
  fitRangeButton: document.getElementById("fitRangeButton"),
  fitP99RangeButton: document.getElementById("fitP99RangeButton"),
  metaPanel: document.getElementById("metaPanel"),
  statusPanel: document.getElementById("statusPanel"),
  shapeLabel: document.getElementById("shapeLabel"),
  queryLabel: document.getElementById("queryLabel"),
  promptText: document.getElementById("promptText"),
  tokenList: document.getElementById("tokenList"),
  heatmapCanvas: document.getElementById("heatmapCanvas"),
};

const ctx = elements.heatmapCanvas.getContext("2d");

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function setStatus(message) {
  elements.statusPanel.textContent = message;
}

function fillSelect(select, options, valueKey, labelBuilder) {
  select.innerHTML = "";
  options.forEach((option) => {
    const element = document.createElement("option");
    element.value = String(option[valueKey]);
    element.textContent = labelBuilder(option);
    select.appendChild(element);
  });
}

function selectedPrefill() {
  if (!state.sample) {
    return null;
  }
  return state.sample.prefills[Number(elements.prefillSelect.value)] || null;
}

function updateMeta() {
  if (!state.sample) {
    elements.metaPanel.innerHTML = "";
    return;
  }
  const item = state.sample.item || {};
  const result = state.sample.result || {};
  const prefill = selectedPrefill() || {};
  const rows = [
    ["Sample", item._id || "-"],
    ["Domain", item.domain || "-"],
    ["Question", item.question || "-"],
    ["Prediction", result.pred || "-"],
    ["Judge", String(result.judge)],
    ["Prefill", `${prefill.label || "-"} / ${prefill.status || "-"}`],
  ];
  elements.metaPanel.innerHTML = rows
    .map(([label, value]) => `<div class="meta-row"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</div>`)
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderPrompt() {
  const prefill = selectedPrefill();
  elements.promptText.innerHTML = "";
  if (!prefill) {
    elements.queryTokenDisplay.textContent = "Click a token in the prompt.";
    return;
  }
  if (!prefill.tokens || !prefill.tokens.length) {
    elements.promptText.textContent = prefill.prompt_text || "";
    elements.queryTokenDisplay.textContent = "No token positions available.";
    return;
  }
  prefill.tokens.forEach((token, idx) => {
    const tokenEl = createTokenElement(token, idx, "prompt-token");
    elements.promptText.appendChild(tokenEl);
  });
  updateQueryDisplay();
}

function colorForValue(value, minValue, maxValue) {
  const span = Math.max(maxValue - minValue, 1e-6);
  const clamped = Math.min(1, Math.max(0, (value - minValue) / span));
  const r = Math.round(245 - clamped * 191);
  const g = Math.round(247 - clamped * 95);
  const b = Math.round(250 - clamped * 173);
  return `rgb(${r}, ${g}, ${b})`;
}

function renderHeatmap() {
  const matrix = state.matrix;
  if (!matrix || !matrix.length) {
    ctx.clearRect(0, 0, elements.heatmapCanvas.width, elements.heatmapCanvas.height);
    return;
  }
  const rows = matrix.length;
  const cols = matrix[0].length;
  const size = Math.max(rows, cols);
  const canvasSize = Math.min(1200, Math.max(360, size));
  elements.heatmapCanvas.width = canvasSize;
  elements.heatmapCanvas.height = canvasSize;
  const cellWidth = canvasSize / cols;
  const cellHeight = canvasSize / rows;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      ctx.fillStyle = colorForValue(matrix[row][col], state.valueMin, state.valueMax);
      ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
    }
  }
  if (state.queryIndex >= 0 && state.queryIndex < rows) {
    ctx.strokeStyle = "#b45309";
    ctx.lineWidth = Math.max(1, 2 * (canvasSize / 640));
    ctx.strokeRect(0, state.queryIndex * cellHeight, canvasSize, cellHeight);
  }
}

function renderTokens() {
  const prefill = selectedPrefill();
  if (!prefill) {
    elements.tokenList.innerHTML = "";
    updateQueryDisplay();
    return;
  }
  const values = state.matrix && state.matrix[state.queryIndex] ? state.matrix[state.queryIndex] : [];
  elements.tokenList.innerHTML = "";
  prefill.tokens.forEach((token, idx) => {
    const tokenEl = createTokenElement(token, idx, "token");
    const value = values[idx] == null ? 0 : values[idx];
    tokenEl.style.background = colorForValue(value, state.valueMin, state.valueMax);
    elements.tokenList.appendChild(tokenEl);
  });
  updateQueryDisplay();
}

function createTokenElement(token, idx, baseClass) {
  const tokenEl = document.createElement("span");
  tokenEl.className = `${baseClass}${idx === state.queryIndex ? " active" : ""}`;
  tokenEl.textContent = token.text || token.piece || String(token.id);
  tokenEl.title = `#${idx} id=${token.id} piece=${token.piece}`;
  tokenEl.setAttribute("role", "button");
  tokenEl.setAttribute("tabindex", "0");
  tokenEl.addEventListener("click", () => setQueryIndex(idx));
  tokenEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setQueryIndex(idx);
    }
  });
  return tokenEl;
}

function updateQueryDisplay() {
  const prefill = selectedPrefill();
  if (!prefill || !prefill.tokens || !prefill.tokens.length) {
    elements.queryLabel.textContent = "";
    elements.queryTokenDisplay.textContent = "Click a token in the prompt.";
    return;
  }
  const active = prefill.tokens[state.queryIndex];
  if (!active) {
    elements.queryLabel.textContent = "";
    elements.queryTokenDisplay.textContent = "Click a token in the prompt.";
    return;
  }
  const label = `#${state.queryIndex}: ${active.text || active.piece || active.id}`;
  elements.queryLabel.textContent = `Query token ${label}`;
  elements.queryTokenDisplay.textContent = label;
}

function setQueryIndex(index) {
  state.queryIndex = Number(index) || 0;
  renderPrompt();
  renderHeatmap();
  renderTokens();
}

function collectMatrixValues() {
  const matrix = state.matrix || [];
  const values = [];
  matrix.forEach((row) => {
    row.forEach((value) => {
      if (Number.isFinite(value)) {
        values.push(value);
      }
    });
  });
  return values;
}

function percentile(values, quantile) {
  if (!values.length) {
    return NaN;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * quantile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sorted[lower];
  }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function autoFitRange(maxQuantile = 1) {
  const values = collectMatrixValues();
  let minValue = Infinity;
  let fullMaxValue = -Infinity;
  values.forEach((value) => {
    if (value < minValue) {
      minValue = value;
    }
    if (value > fullMaxValue) {
      fullMaxValue = value;
    }
  });
  let maxValue = maxQuantile >= 1 ? fullMaxValue : percentile(values, maxQuantile);
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    minValue = 0;
    maxValue = 1;
  }
  if (maxValue <= minValue && fullMaxValue > minValue) {
    maxValue = fullMaxValue;
  }
  if (maxValue <= minValue) {
    maxValue = minValue + 1e-6;
  }
  state.valueMin = minValue;
  state.valueMax = maxValue;
  elements.minInput.value = String(minValue);
  elements.maxInput.value = String(maxValue);
}

async function loadMatrix() {
  if (!state.run || !state.sample) {
    return;
  }
  const prefillIndex = elements.prefillSelect.value;
  const layer = elements.layerSelect.value;
  const head = elements.headSelect.value;
  const prefill = selectedPrefill();
  if (!prefill || prefill.status !== "saved") {
    state.matrix = null;
    elements.shapeLabel.textContent = prefill ? `Status: ${prefill.status}` : "";
    renderHeatmap();
    renderTokens();
    return;
  }
  setStatus("Loading matrix...");
  const payload = await fetchJson(
    `/api/runs/${encodeURIComponent(state.run.run_id)}/samples/${encodeURIComponent(state.sample.sample_id)}/prefills/${encodeURIComponent(prefillIndex)}/matrix?layer=${encodeURIComponent(layer)}&head=${encodeURIComponent(head)}`
  );
  state.matrix = payload.matrix;
  elements.shapeLabel.textContent = `Layer ${payload.layer_index}, head ${payload.head_index}, shape ${payload.shape.join(" x ")}`;
  const rowCount = payload.shape[0];
  if (state.queryIndex >= rowCount) {
    state.queryIndex = 0;
  }
  renderPrompt();
  renderHeatmap();
  renderTokens();
  setStatus("Ready.");
}

function syncLayerOptions() {
  const prefill = selectedPrefill();
  const layerIndices = prefill && prefill.layer_indices ? prefill.layer_indices : [];
  fillSelect(
    elements.layerSelect,
    layerIndices.map((layer) => ({ layer })),
    "layer",
    (option) => `Layer ${option.layer}`
  );
}

function syncHeadOptions() {
  const prefill = selectedPrefill();
  const headCount = prefill && prefill.attn_shape ? prefill.attn_shape[1] : 0;
  fillSelect(
    elements.headSelect,
    Array.from({ length: headCount }, (_, idx) => ({ head: idx })),
    "head",
    (option) => `Head ${option.head}`
  );
}

async function loadSample() {
  if (!state.run) {
    return;
  }
  setStatus("Loading sample...");
  const sampleId = elements.sampleSelect.value;
  state.sample = await fetchJson(`/api/runs/${encodeURIComponent(state.run.run_id)}/samples/${encodeURIComponent(sampleId)}`);
  fillSelect(
    elements.prefillSelect,
    state.sample.prefills.map((prefill) => ({ index: prefill.prefill_index, label: prefill.label, status: prefill.status })),
    "index",
    (option) => `#${option.index} ${option.label} (${option.status})`
  );
  syncLayerOptions();
  syncHeadOptions();
  renderPrompt();
  updateMeta();
  await loadMatrix();
}

async function loadRun() {
  const runId = elements.runSelect.value;
  if (!runId) {
    return;
  }
  setStatus("Loading run...");
  state.run = await fetchJson(`/api/runs/${encodeURIComponent(runId)}`);
  fillSelect(
    elements.sampleSelect,
    state.run.samples || [],
    "sample_id",
    (sample) => `${sample.sample_index}: ${sample._id} [${sample.prefill_statuses.join(", ")}]`
  );
  await loadSample();
}

async function loadRuns() {
  setStatus("Loading runs...");
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs || [];
  fillSelect(elements.runSelect, state.runs, "run_id", (run) => `${run.run_id} (${run.sample_count})`);
  if (!state.runs.length) {
    setStatus("No runs found under the configured heatmap root.");
    return;
  }
  await loadRun();
}

elements.runSelect.addEventListener("change", () => {
  loadRun().catch((error) => setStatus(error.message));
});

elements.sampleSelect.addEventListener("change", () => {
  loadSample().catch((error) => setStatus(error.message));
});

elements.prefillSelect.addEventListener("change", () => {
  syncLayerOptions();
  syncHeadOptions();
  renderPrompt();
  updateMeta();
  loadMatrix().catch((error) => setStatus(error.message));
});

elements.layerSelect.addEventListener("change", () => {
  loadMatrix().catch((error) => setStatus(error.message));
});

elements.headSelect.addEventListener("change", () => {
  loadMatrix().catch((error) => setStatus(error.message));
});

elements.minInput.addEventListener("change", () => {
  state.valueMin = Number(elements.minInput.value);
  renderHeatmap();
  renderTokens();
});

elements.maxInput.addEventListener("change", () => {
  state.valueMax = Number(elements.maxInput.value);
  renderHeatmap();
  renderTokens();
});

elements.fitRangeButton.addEventListener("click", () => {
  autoFitRange();
  renderHeatmap();
  renderTokens();
});

elements.fitP99RangeButton.addEventListener("click", () => {
  autoFitRange(0.99);
  renderHeatmap();
  renderTokens();
});

loadRuns().catch((error) => setStatus(error.message));
