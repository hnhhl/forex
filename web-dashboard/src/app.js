const API_BASE = localStorage.getItem("API_BASE") || "http://localhost:8000";

document.getElementById("api-base").textContent = API_BASE;

async function fetchJSON(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

function pretty(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

// Buttons
const btnRefreshStatus = document.getElementById("btn-refresh-status");
const btnGenerateSignal = document.getElementById("btn-generate-signal");
const btnEnsembleStatus = document.getElementById("btn-ensemble-status");
const btnListModels = document.getElementById("btn-list-models");
const btnTrainingStatus = document.getElementById("btn-training-status");
const btnTrainingStart = document.getElementById("btn-training-start");

// Outputs
const outStatus = document.getElementById("status-output");
const outSignal = document.getElementById("signal-output");
const outEnsemble = document.getElementById("ensemble-output");
const outTraining = document.getElementById("training-output");

btnRefreshStatus.addEventListener("click", async () => {
  outStatus.textContent = "Loading...";
  try {
    const data = await fetchJSON("/system/status");
    pretty(outStatus, data);
  } catch (e) {
    outStatus.textContent = e.message;
  }
});

btnGenerateSignal.addEventListener("click", async () => {
  outSignal.textContent = "Generating... (có thể mất vài giây khi model load)";
  try {
    const data = await fetchJSON("/signal", { method: "POST", body: JSON.stringify({}) });
    pretty(outSignal, data);
  } catch (e) {
    outSignal.textContent = e.message;
  }
});

btnEnsembleStatus.addEventListener("click", async () => {
  outEnsemble.textContent = "Loading ensemble status...";
  try {
    const data = await fetchJSON("/ensemble/status");
    pretty(outEnsemble, data);
  } catch (e) {
    outEnsemble.textContent = e.message;
  }
});

btnListModels.addEventListener("click", async () => {
  outEnsemble.textContent = "Listing models...";
  try {
    const data = await fetchJSON("/models/list");
    pretty(outEnsemble, data);
  } catch (e) {
    outEnsemble.textContent = e.message;
  }
});

btnTrainingStatus.addEventListener("click", async () => {
  outTraining.textContent = "Loading training status...";
  try {
    const data = await fetchJSON("/training/status");
    pretty(outTraining, data);
  } catch (e) {
    outTraining.textContent = e.message;
  }
});

btnTrainingStart.addEventListener("click", async () => {
  outTraining.textContent = "Starting training...";
  try {
    const data = await fetchJSON("/training/start", { method: "POST" });
    pretty(outTraining, data);
  } catch (e) {
    outTraining.textContent = e.message;
  }
});

// Auto-load some data on first open
btnRefreshStatus.click();
btnEnsembleStatus.click(); 