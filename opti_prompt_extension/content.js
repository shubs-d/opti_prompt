(() => {
  const BUTTON_ID = "opti-prompt-optimize-btn";
  const STATUS_ID = "opti-prompt-status";

  const SITE_SELECTORS = {
    "chatgpt.com": ["#prompt-textarea", "textarea"],
    "claude.ai": ["div[contenteditable='true']", "textarea"],
    "gemini.google.com": ["div[contenteditable='true']", "textarea"],
    "www.perplexity.ai": ["textarea", "div[contenteditable='true']"],
  };

  init();

  function init() {
    injectButton();
    window.addEventListener("resize", placeControls);
    window.addEventListener("scroll", placeControls, true);

    const observer = new MutationObserver(() => {
      injectButton();
      placeControls();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: false,
    });

    setInterval(placeControls, 1500);
  }

  function currentHost() {
    const host = window.location.hostname;
    if (SITE_SELECTORS[host]) {
      return host;
    }
    return null;
  }

  function findInput() {
    const host = currentHost();
    const selectors = host ? SITE_SELECTORS[host] : ["textarea", "div[contenteditable='true']"];

    for (const selector of selectors) {
      const nodes = Array.from(document.querySelectorAll(selector));
      const node = nodes.find((item) => isUsableInput(item));
      if (node) {
        return node;
      }
    }

    return null;
  }

  function isUsableInput(element) {
    if (!(element instanceof HTMLElement)) {
      return false;
    }
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden") {
      return false;
    }
    const rect = element.getBoundingClientRect();
    return rect.width > 180 && rect.height > 28;
  }

  function injectButton() {
    if (document.getElementById(BUTTON_ID)) {
      placeControls();
      return;
    }

    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.textContent = "Optimize Prompt";
    button.className = "opti-prompt-btn";
    button.addEventListener("click", handleOptimizeClick);

    const status = document.createElement("div");
    status.id = STATUS_ID;
    status.className = "opti-prompt-status";

    document.body.appendChild(button);
    document.body.appendChild(status);
    placeControls();
  }

  function placeControls() {
    const input = findInput();
    const button = document.getElementById(BUTTON_ID);
    const status = document.getElementById(STATUS_ID);
    if (!input || !button || !status) {
      return;
    }

    const rect = input.getBoundingClientRect();
    const top = window.scrollY + rect.top - 40;
    const left = window.scrollX + rect.right - 150;

    button.style.top = `${Math.max(8, top)}px`;
    button.style.left = `${Math.max(8, left)}px`;

    status.style.top = `${Math.max(8, top - 30)}px`;
    status.style.left = `${Math.max(8, left)}px`;
  }

  function getInputText(element) {
    if (!element) {
      return "";
    }

    if (element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) {
      return element.value || "";
    }

    return element.innerText || element.textContent || "";
  }

  function setInputText(element, text) {
    if (element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement) {
      element.focus();
      const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      if (nativeSetter && element instanceof HTMLTextAreaElement) {
        nativeSetter.call(element, text);
      } else {
        element.value = text;
      }
      element.dispatchEvent(new Event("input", { bubbles: true }));
      return;
    }

    if (element instanceof HTMLElement && element.isContentEditable) {
      element.focus();
      element.textContent = text;
      element.dispatchEvent(new InputEvent("input", { bubbles: true, data: text }));
    }
  }

  async function handleOptimizeClick() {
    const button = document.getElementById(BUTTON_ID);
    const status = document.getElementById(STATUS_ID);
    const input = findInput();
    if (!button || !status || !input) {
      showStatus("No prompt editor detected", true);
      return;
    }

    const prompt = getInputText(input).trim();
    if (!prompt) {
      showStatus("Prompt is empty", true);
      return;
    }

    button.disabled = true;
    button.textContent = "Optimizing...";
    showStatus("Sending prompt to OptiPrompt backend...", false);

    try {
      const settings = await sendMessage({ type: "OPTIPROMPT_GET_SETTINGS" });
      if (!settings.ok) {
        throw new Error(settings.error || "Could not load extension settings.");
      }

      const response = await sendMessage({
        type: "OPTIPROMPT_OPTIMIZE",
        payload: {
          prompt,
          aggressiveness: settings.settings.aggressiveness,
          backendUrl: settings.settings.backendUrl,
        },
      });

      if (!response.ok) {
        throw new Error(response.error || "Optimization failed.");
      }

      const optimizedPrompt = String(response.data?.compressed_prompt || "");
      if (!optimizedPrompt) {
        throw new Error("Backend did not return a compressed prompt.");
      }

      setInputText(input, optimizedPrompt);
      showStatus("Prompt optimized and replaced", false);
    } catch (error) {
      showStatus(String(error.message || error), true);
    } finally {
      button.disabled = false;
      button.textContent = "Optimize Prompt";
      placeControls();
    }
  }

  function showStatus(message, isError) {
    const status = document.getElementById(STATUS_ID);
    if (!status) {
      return;
    }

    status.textContent = message;
    status.classList.toggle("error", !!isError);
    status.classList.add("visible");

    window.clearTimeout(status._hideTimeout);
    status._hideTimeout = window.setTimeout(() => {
      status.classList.remove("visible");
    }, 2600);
  }

  function sendMessage(message) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        resolve(response || { ok: false, error: "No response from extension service worker." });
      });
    });
  }
})();
