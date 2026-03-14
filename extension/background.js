/**
 * OptiPrompt — Background Service Worker (Manifest V3)
 *
 * Acts as the bridge between popup / content scripts and the local
 * FastAPI backend.  Centralising API calls here avoids CSP issues
 * that content scripts face on third-party pages.
 */

const API_BASE = "https://optiprompt-gqd9hqf6dffvaacb.eastasia-01.azurewebsites.net";
const PREDICTION_CACHE = new Map();
const PREDICTION_CACHE_MAX = 120;

/* ------------------------------------------------------------------ */
/* Message router                                                      */
/* ------------------------------------------------------------------ */

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "OPTIMIZE") {
    handleOptimize(message.payload)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep the channel open for async response
  }

  if (message.type === "HEALTH_CHECK") {
    handleHealthCheck()
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  if (message.type === "PREDICT") {
    handlePredict(message.payload)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }

  // Forward stats update from content script
  if (message.type === "UPDATE_STATS") {
    updateStatsFromContent(message.payload);
    sendResponse({ ok: true });
    return false;
  }

  // Forward history entry from content script
  if (message.type === "ADD_HISTORY") {
    addHistoryFromContent(message.payload);
    sendResponse({ ok: true });
    return false;
  }

  if (message.type === "ANALYZE") {
    handleAnalyze(message.payload)
      .then(sendResponse)
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true;
  }
});

/* ------------------------------------------------------------------ */
/* API helpers                                                         */
/* ------------------------------------------------------------------ */

async function handleOptimize({ prompt, mode, aggressiveness, auto_aggressiveness, intent_override }) {
  const body = { prompt, auto_aggressiveness: auto_aggressiveness ?? true };
  if (aggressiveness != null) body.aggressiveness = aggressiveness;
  if (mode) body.mode = mode;
  if (intent_override) body.intent_override = intent_override;

  const res = await fetch(`${API_BASE}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }

  return { ok: true, data: await res.json() };
}

async function handleHealthCheck() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return { ok: true, data: await res.json() };
}

async function handleAnalyze({ prompt, mode, aggressiveness, auto_aggressiveness, intent_override }) {
  const body = { prompt, auto_aggressiveness: auto_aggressiveness ?? true };
  if (aggressiveness != null) body.aggressiveness = aggressiveness;
  if (mode) body.mode = mode;
  if (intent_override) body.intent_override = intent_override;

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }

  return { ok: true, data: await res.json() };
}

async function handlePredict({ text }) {
  const normalized = String(text || "").trim();
  if (!normalized) return { ok: true, data: { prediction: "" } };

  if (PREDICTION_CACHE.has(normalized)) {
    return { ok: true, data: { prediction: PREDICTION_CACHE.get(normalized) } };
  }

  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: normalized }),
  });

  if (!res.ok) {
    const textBody = await res.text();
    throw new Error(`API ${res.status}: ${textBody}`);
  }

  const data = await res.json();
  const prediction = data?.prediction || "";

  if (prediction) {
    if (PREDICTION_CACHE.has(normalized)) PREDICTION_CACHE.delete(normalized);
    PREDICTION_CACHE.set(normalized, prediction);
    if (PREDICTION_CACHE.size > PREDICTION_CACHE_MAX) {
      const oldest = PREDICTION_CACHE.keys().next().value;
      PREDICTION_CACHE.delete(oldest);
    }
  }

  return { ok: true, data: { prediction } };
}

/* ------------------------------------------------------------------ */
/* Storage helpers (called from content script messages)               */
/* ------------------------------------------------------------------ */

function updateStatsFromContent(data) {
  chrome.storage.local.get(["opti_stats"], (items) => {
    const stats = items.opti_stats || {
      total_prompts_processed: 0,
      tokens_saved: 0,
      total_compression_sum: 0,
      total_templates_used: 0,
      lifetime_prompt_usage: 0,
      last_used_timestamp: null,
    };

    stats.total_prompts_processed += 1;
    stats.lifetime_prompt_usage += 1;
    const tokensSaved = (data.original_token_count ?? 0) - (data.compressed_token_count ?? 0);
    stats.tokens_saved += Math.max(0, tokensSaved);
    stats.total_compression_sum += data.token_reduction_percent || 0;
    stats.last_used_timestamp = Date.now();

    chrome.storage.local.set({ opti_stats: stats });
  });
}

function addHistoryFromContent(entry) {
  chrome.storage.local.get(["opti_history"], (items) => {
    const history = items.opti_history || [];
    history.push(entry);
    while (history.length > 20) history.shift();
    chrome.storage.local.set({ opti_history: history });
  });
}
