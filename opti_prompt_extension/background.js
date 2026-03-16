const DEFAULT_SETTINGS = {
  backendUrl: "http://127.0.0.1:8000",
  aggressiveness: 0.3,
};

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(["optiPromptSettings"], (items) => {
    if (!items.optiPromptSettings) {
      chrome.storage.sync.set({ optiPromptSettings: DEFAULT_SETTINGS });
    }
  });
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "OPTIPROMPT_GET_SETTINGS") {
    chrome.storage.sync.get(["optiPromptSettings"], (items) => {
      sendResponse({ ok: true, settings: { ...DEFAULT_SETTINGS, ...(items.optiPromptSettings || {}) } });
    });
    return true;
  }

  if (message?.type === "OPTIPROMPT_SAVE_SETTINGS") {
    const payload = message.payload || {};
    const settings = {
      backendUrl: normalizeBaseUrl(payload.backendUrl || DEFAULT_SETTINGS.backendUrl),
      aggressiveness: clampNumber(payload.aggressiveness, 0, 1, DEFAULT_SETTINGS.aggressiveness),
    };
    chrome.storage.sync.set({ optiPromptSettings: settings }, () => {
      sendResponse({ ok: true, settings });
    });
    return true;
  }

  if (message?.type === "OPTIPROMPT_PING") {
    pingBackend(message.payload?.backendUrl)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }

  if (message?.type === "OPTIPROMPT_OPTIMIZE") {
    optimizePrompt(message.payload)
      .then((data) => sendResponse({ ok: true, data }))
      .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));
    return true;
  }

  return false;
});

async function optimizePrompt(payload) {
  const settings = await getSettings();
  const backendUrl = normalizeBaseUrl(payload?.backendUrl || settings.backendUrl);
  const prompt = String(payload?.prompt || "").trim();
  if (!prompt) {
    throw new Error("Prompt is empty.");
  }

  const aggressiveness = clampNumber(
    payload?.aggressiveness,
    0,
    1,
    settings.aggressiveness,
  );

  const body = {
    prompt,
    aggressiveness,
    auto_aggressiveness: false,
  };

  const response = await fetchWithTimeout(`${backendUrl}/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, 7000);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Optimization failed (${response.status}): ${text}`);
  }

  return response.json();
}

async function pingBackend(url) {
  const backendUrl = normalizeBaseUrl(url || (await getSettings()).backendUrl);
  const response = await fetchWithTimeout(`${backendUrl}/health`, { method: "GET" }, 3500);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status}).`);
  }
  return response.json();
}

function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["optiPromptSettings"], (items) => {
      resolve({ ...DEFAULT_SETTINGS, ...(items.optiPromptSettings || {}) });
    });
  });
}

function normalizeBaseUrl(url) {
  return String(url || "").trim().replace(/\/$/, "") || DEFAULT_SETTINGS.backendUrl;
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  if (Number.isNaN(number)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, number));
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}
