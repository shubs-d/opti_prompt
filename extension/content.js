/**
 * OptiPrompt — Content Script (v2)
 *
 * Injects a dropdown button group into AI chat interfaces.
 * Shows a result preview panel instead of directly replacing text.
 * Supports: ChatGPT, Claude
 */

(() => {
  "use strict";

  const BUTTON_ID    = "optiprompt-inject-btn";
  const DROPDOWN_ID  = "optiprompt-dropdown";
  const PANEL_ID     = "optiprompt-result-panel";
  const TOAST_ID     = "optiprompt-toast";
  const AUTO_DEBOUNCE_MS = 1200;

  let autoOptimizeEnabled = false;
  let preferredMode = "optimize";
  let autoOptimizeTimer = null;
  let lastAutoOptimizedText = "";
  let isProgrammaticEdit = false;

  /* ---------------------------------------------------------------- */
  /* Textarea selectors per site                                       */
  /* ---------------------------------------------------------------- */
  const SELECTORS = {
    "chatgpt.com": [
      "#prompt-textarea",
      'div[contenteditable="true"][id="prompt-textarea"]',
    ],
    "claude.ai": [
      'div[contenteditable="true"].ProseMirror',
      'fieldset div[contenteditable="true"]',
    ],
  };

  /* ---------------------------------------------------------------- */
  /* Utility: detect host                                              */
  /* ---------------------------------------------------------------- */
  function getHost() {
    const h = location.hostname;
    if (h.includes("chatgpt.com")) return "chatgpt.com";
    if (h.includes("claude.ai"))   return "claude.ai";
    return null;
  }

  /* ---------------------------------------------------------------- */
  /* Utility: find the active textarea / contenteditable               */
  /* ---------------------------------------------------------------- */
  function findTextbox() {
    const host = getHost();
    if (!host) return null;
    for (const sel of SELECTORS[host] || []) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  }

  /* ---------------------------------------------------------------- */
  /* Read / write text from element                                    */
  /* ---------------------------------------------------------------- */
  function getText(el) {
    if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") return el.value;
    return el.innerText || el.textContent || "";
  }

  function setText(el, text) {
    if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
      const setter =
        Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set ||
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      if (setter) setter.call(el, text);
      else el.value = text;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      // contenteditable (ChatGPT's editor, Claude's ProseMirror)
      el.focus();
      el.innerHTML = "";
      document.execCommand("insertText", false, text);
      el.dispatchEvent(new InputEvent("input", { bubbles: true, data: text }));
    }
  }

  /* ---------------------------------------------------------------- */
  /* Toast notification                                                */
  /* ---------------------------------------------------------------- */
  function showToast(msg, isError = false) {
    let toast = document.getElementById(TOAST_ID);
    if (!toast) {
      toast = document.createElement("div");
      toast.id = TOAST_ID;
      toast.className = "optiprompt-toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.toggle("error", isError);
    toast.classList.remove("show");
    requestAnimationFrame(() => {
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 2400);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Create the inject button group                                    */
  /* ---------------------------------------------------------------- */
  function createButton() {
    if (document.getElementById(BUTTON_ID)) return;

    // Main button container
    const container = document.createElement("div");
    container.id = BUTTON_ID;
    container.className = "optiprompt-btn-group";

    // Main button
    const mainBtn = document.createElement("button");
    mainBtn.className = "optiprompt-main-btn";
    mainBtn.textContent = "⚡ Optimize";
    mainBtn.addEventListener("click", () => handleOptimize("optimize"));

    // Dropdown toggle
    const dropdownToggle = document.createElement("button");
    dropdownToggle.className = "optiprompt-dropdown-toggle";
    dropdownToggle.innerHTML = "▾";
    dropdownToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleDropdown();
    });

    // Dropdown menu
    const dropdown = document.createElement("div");
    dropdown.id = DROPDOWN_ID;
    dropdown.className = "optiprompt-dropdown";

    const modes = [
      { mode: "optimize", icon: "⚡", label: "Optimize" },
      { mode: "enhance",  icon: "✨", label: "Enhance" },
      { mode: "both",     icon: "⚡✨", label: "Optimize + Enhance" },
    ];

    modes.forEach(({ mode, icon, label }) => {
      const item = document.createElement("button");
      item.className = "optiprompt-dropdown-item";
      item.textContent = `${icon} ${label}`;
      item.addEventListener("click", (e) => {
        e.stopPropagation();
        hideDropdown();
        handleOptimize(mode);
      });
      dropdown.appendChild(item);
    });

    container.appendChild(mainBtn);
    container.appendChild(dropdownToggle);
    container.appendChild(dropdown);
    document.body.appendChild(container);

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
      if (!container.contains(e.target)) hideDropdown();
    });
  }

  function toggleDropdown() {
    const dd = document.getElementById(DROPDOWN_ID);
    if (dd) dd.classList.toggle("show");
  }

  function hideDropdown() {
    const dd = document.getElementById(DROPDOWN_ID);
    if (dd) dd.classList.remove("show");
  }

  /* ---------------------------------------------------------------- */
  /* Optimize handler                                                  */
  /* ---------------------------------------------------------------- */
  async function handleOptimize(mode = "optimize") {
    return handleOptimizeWithOptions(mode, { autoTriggered: false });
  }

  async function handleOptimizeWithOptions(mode = "optimize", options = {}) {
    const textbox = findTextbox();
    if (!textbox) {
      showToast("Cannot find the chat textbox.", true);
      return;
    }

    const prompt = getText(textbox).trim();
    if (!prompt) {
      showToast("Textbox is empty.", true);
      return;
    }

    const mainBtn = document.querySelector(".optiprompt-main-btn");
    if (mainBtn && !options.autoTriggered) {
      mainBtn.textContent = "⏳ Working…";
      mainBtn.disabled = true;
    }

    try {
      const res = await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(
          { type: "OPTIMIZE", payload: { prompt, mode, auto_aggressiveness: true } },
          (response) => {
            if (chrome.runtime.lastError) {
              reject(new Error(chrome.runtime.lastError.message));
            } else {
              resolve(response);
            }
          }
        );
      });

      if (!res?.ok) throw new Error(res?.error || "Backend error");

      if (options.autoTriggered) {
        lastAutoOptimizedText = prompt;
        showToast("✨ Auto optimization ready");
      }

      showResultPanel(prompt, res.data, mode);

      // Update stats via background
      chrome.runtime.sendMessage({
        type: "UPDATE_STATS",
        payload: res.data,
      });

      // Add to history via background
      chrome.runtime.sendMessage({
        type: "ADD_HISTORY",
        payload: {
          original_prompt: prompt.substring(0, 200),
          optimized_prompt: (res.data.compressed_prompt || "").substring(0, 200),
          timestamp: Date.now(),
          mode_used: mode,
          tokens_before: res.data.original_token_count ?? 0,
          tokens_after: res.data.compressed_token_count ?? 0,
          tokens_saved: Math.max(0, (res.data.original_token_count ?? 0) - (res.data.compressed_token_count ?? 0)),
          compression: res.data.token_reduction_percent || 0,
        },
      });

    } catch (err) {
      showToast(`Error: ${err.message}`, true);
    } finally {
      if (mainBtn && !options.autoTriggered) {
        mainBtn.textContent = "⚡ Optimize";
        mainBtn.disabled = false;
      }
    }
  }

  /* ---------------------------------------------------------------- */
  /* Result Preview Panel                                              */
  /* ---------------------------------------------------------------- */
  function showResultPanel(original, data, mode) {
    removeResultPanel();

    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.className = "optiprompt-panel";

    const tokensBefore = data.original_token_count ?? "—";
    const tokensAfter  = data.compressed_token_count ?? "—";
    const reduction    = data.token_reduction_percent?.toFixed(1) ?? "—";
    const similarity   = data.evaluation?.semantic_similarity?.toFixed(4) ?? "—";
    const density      = data.density?.density_score?.toFixed(4) ?? "—";
    const modeLabel    = mode === "both"
      ? "Optimize + Enhance"
      : mode.charAt(0).toUpperCase() + mode.slice(1);
    const diffHtml = buildDiffHtml(data.diff || {});

    panel.innerHTML = `
      <div class="optiprompt-panel-header">
        <span class="optiprompt-panel-title">⚡ OptiPrompt Result</span>
        <span class="optiprompt-panel-mode">${modeLabel}</span>
        <button class="optiprompt-panel-close" id="optiprompt-panel-close">✕</button>
      </div>

      <div class="optiprompt-panel-stats">
        <div class="optiprompt-stat">
          <div class="optiprompt-stat-value">${tokensBefore}</div>
          <div class="optiprompt-stat-label">Before</div>
        </div>
        <div class="optiprompt-stat">
          <div class="optiprompt-stat-value">${tokensAfter}</div>
          <div class="optiprompt-stat-label">After</div>
        </div>
        <div class="optiprompt-stat">
          <div class="optiprompt-stat-value">${reduction}%</div>
          <div class="optiprompt-stat-label">Saved</div>
        </div>
        <div class="optiprompt-stat">
          <div class="optiprompt-stat-value">${similarity}</div>
          <div class="optiprompt-stat-label">Similarity</div>
        </div>
        <div class="optiprompt-stat">
          <div class="optiprompt-stat-value">${density}</div>
          <div class="optiprompt-stat-label">Density</div>
        </div>
      </div>

      <div class="optiprompt-panel-section">
        <div class="optiprompt-panel-label">Original</div>
        <div class="optiprompt-panel-text optiprompt-panel-original">${escapeHtml(original.substring(0, 300))}${original.length > 300 ? "…" : ""}</div>
      </div>

      <div class="optiprompt-panel-section">
        <div class="optiprompt-panel-label">Optimized</div>
        <div class="optiprompt-panel-text optiprompt-panel-optimized">${escapeHtml(data.compressed_prompt)}</div>
      </div>

      <div class="optiprompt-panel-section">
        <div class="optiprompt-panel-label">Diff</div>
        <div class="optiprompt-panel-text optiprompt-panel-diff">${diffHtml}</div>
      </div>

      <div class="optiprompt-panel-actions">
        <button class="optiprompt-btn-action optiprompt-btn-replace" id="optiprompt-btn-replace">✅ Replace Prompt</button>
        <button class="optiprompt-btn-action optiprompt-btn-copy" id="optiprompt-btn-copy">📋 Copy Prompt</button>
      </div>
    `;

    document.body.appendChild(panel);

    // Animate in
    requestAnimationFrame(() => panel.classList.add("show"));

    // Close button
    document.getElementById("optiprompt-panel-close").addEventListener("click", removeResultPanel);

    // Replace button
    document.getElementById("optiprompt-btn-replace").addEventListener("click", () => {
      const textbox = findTextbox();
      if (textbox) {
        isProgrammaticEdit = true;
        setText(textbox, data.compressed_prompt);
        setTimeout(() => { isProgrammaticEdit = false; }, 100);
        showToast("✅ Prompt replaced!");
        removeResultPanel();
      }
    });

    // Copy button
    document.getElementById("optiprompt-btn-copy").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(data.compressed_prompt);
        const btn = document.getElementById("optiprompt-btn-copy");
        if (btn) {
          btn.textContent = "✅ Copied!";
          setTimeout(() => { if (btn) btn.textContent = "📋 Copy Prompt"; }, 1500);
        }
      } catch {
        showToast("Clipboard access denied.", true);
      }
    });
  }

  function removeResultPanel() {
    const panel = document.getElementById(PANEL_ID);
    if (panel) {
      panel.classList.remove("show");
      setTimeout(() => panel.remove(), 200);
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function buildDiffHtml(diff) {
    const removed = Array.isArray(diff.removed) ? diff.removed : [];
    const rewritten = Array.isArray(diff.rewritten) ? diff.rewritten : [];
    if (!removed.length && !rewritten.length) {
      return '<span class="optiprompt-muted">No major changes detected.</span>';
    }

    const chunks = [];
    if (removed.length) {
      chunks.push(`<div class="optiprompt-diff-group"><strong>Removed:</strong> ${removed.slice(0, 6).map((item) => `<span class="optiprompt-diff-chip">${escapeHtml(item)}</span>`).join(" ")}</div>`);
    }
    if (rewritten.length) {
      chunks.push(`<div class="optiprompt-diff-group">${rewritten.slice(0, 4).map((item) => `<div class="optiprompt-diff-row"><span>${escapeHtml(item.original || "")}</span><span>→</span><span>${escapeHtml(item.compressed || "")}</span></div>`).join("")}</div>`);
    }
    return chunks.join("");
  }

  function loadSettings() {
    chrome.storage.local.get(["opti_settings"], (items) => {
      const settings = items.opti_settings || {};
      autoOptimizeEnabled = !!settings.autoOptimize;
      preferredMode = settings.mode || "optimize";
    });
  }

  function wireAutoOptimize(textbox) {
    if (textbox.dataset.optipromptBound === "true") return;
    textbox.dataset.optipromptBound = "true";
    textbox.addEventListener("input", () => {
      if (!autoOptimizeEnabled || isProgrammaticEdit) return;
      const text = getText(textbox).trim();
      if (!text || text.length < 24 || text === lastAutoOptimizedText) return;
      clearTimeout(autoOptimizeTimer);
      autoOptimizeTimer = setTimeout(() => {
        handleOptimizeWithOptions(preferredMode, { autoTriggered: true });
      }, AUTO_DEBOUNCE_MS);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Listen for REPLACE_PROMPT from popup                              */
  /* ---------------------------------------------------------------- */
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "REPLACE_PROMPT") {
      const textbox = findTextbox();
      if (textbox) {
        isProgrammaticEdit = true;
        setText(textbox, message.text);
        setTimeout(() => { isProgrammaticEdit = false; }, 100);
        showToast("✅ Prompt replaced from popup!");
      }
      sendResponse({ ok: true });
    }
  });

  /* ---------------------------------------------------------------- */
  /* Position the button group relative to textbox                     */
  /* ---------------------------------------------------------------- */
  function positionButton() {
    const container = document.getElementById(BUTTON_ID);
    const textbox = findTextbox();
    if (!container || !textbox) return;

    const rect = textbox.getBoundingClientRect();
    container.style.top  = `${window.scrollY + rect.top - 40}px`;
    container.style.left = `${window.scrollX + rect.right - container.offsetWidth}px`;
  }

  /* ---------------------------------------------------------------- */
  /* Observer — wait for textbox to appear in SPA                      */
  /* ---------------------------------------------------------------- */
  function init() {
    const host = getHost();
    if (!host) return;
    loadSettings();

    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === "local" && changes.opti_settings) {
        const next = changes.opti_settings.newValue || {};
        autoOptimizeEnabled = !!next.autoOptimize;
        preferredMode = next.mode || "optimize";
      }
    });

    const observer = new MutationObserver(() => {
      const textbox = findTextbox();
      if (textbox && !document.getElementById(BUTTON_ID)) {
        createButton();
        positionButton();
      }
      if (textbox) wireAutoOptimize(textbox);
    });

    observer.observe(document.body, { childList: true, subtree: true });

    // Also try immediately
    if (findTextbox()) {
      createButton();
      wireAutoOptimize(findTextbox());
    }

    // Reposition on scroll/resize
    window.addEventListener("scroll", positionButton, { passive: true });
    window.addEventListener("resize", positionButton, { passive: true });

    // Re-position periodically (SPA layout shifts)
    setInterval(positionButton, 2000);
  }

  /* ---------------------------------------------------------------- */
  /* Boot                                                              */
  /* ---------------------------------------------------------------- */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
