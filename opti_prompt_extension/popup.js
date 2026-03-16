const backendInput = document.getElementById("backendUrl");
const aggressivenessInput = document.getElementById("aggressiveness");
const aggressivenessValue = document.getElementById("aggressivenessValue");
const saveButton = document.getElementById("saveSettings");
const testButton = document.getElementById("testBackend");
const statusElement = document.getElementById("status");

init();

function init() {
  chrome.runtime.sendMessage({ type: "OPTIPROMPT_GET_SETTINGS" }, (response) => {
    if (!response?.ok) {
      showStatus("Failed to load settings.", true);
      return;
    }

    backendInput.value = response.settings.backendUrl;
    aggressivenessInput.value = String(response.settings.aggressiveness);
    renderAggressiveness();
  });

  aggressivenessInput.addEventListener("input", renderAggressiveness);

  saveButton.addEventListener("click", () => {
    const payload = collectSettings();
    chrome.runtime.sendMessage({ type: "OPTIPROMPT_SAVE_SETTINGS", payload }, (response) => {
      if (!response?.ok) {
        showStatus("Could not save settings.", true);
        return;
      }
      showStatus("Settings saved.", false);
    });
  });

  testButton.addEventListener("click", () => {
    const payload = collectSettings();
    showStatus("Checking backend...", false);

    chrome.runtime.sendMessage(
      { type: "OPTIPROMPT_PING", payload: { backendUrl: payload.backendUrl } },
      (response) => {
        if (!response?.ok) {
          showStatus(response?.error || "Backend not reachable.", true);
          return;
        }
        showStatus("Backend reachable.", false);
      },
    );
  });
}

function collectSettings() {
  return {
    backendUrl: backendInput.value.trim(),
    aggressiveness: Number(aggressivenessInput.value),
  };
}

function renderAggressiveness() {
  aggressivenessValue.textContent = Number(aggressivenessInput.value).toFixed(2);
}

function showStatus(message, isError) {
  statusElement.textContent = message;
  statusElement.classList.toggle("error", !!isError);
}
