/**
 * OptiPrompt — Popup Script
 * Multi-tab prompt productivity toolkit.
 */

(() => {
  "use strict";

  const STORAGE_KEYS = {
    settings: "opti_settings",
    templates: "opti_templates",
    history: "opti_history",
    stats: "opti_stats",
  };

  const $ = (id) => document.getElementById(id);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const refs = {
    tabs: $$(".tab"),
    tabContents: $$(".tab-content"),
    prompt: $("prompt"),
    promptGhostWrap: $("promptGhostWrap"),
    promptGhostText: $("promptGhostText"),
    autoMode: $("autoMode"),
    autoOptimizeToggle: $("autoOptimizeToggle"),
    intentOverride: $("intentOverride"),
    aggr: $("aggr"),
    aggrVal: $("aggrVal"),
    btnOptimize: $("btnOptimize"),
    btnText: $("btnText"),
    btnSpinner: $("btnSpinner"),
    errorBanner: $("errorBanner"),
    statusDot: $("statusDot"),
    results: $("results"),
    intentBadge: $("intentBadge"),
    intentMeta: $("intentMeta"),
    mTokensBefore: $("mTokensBefore"),
    mTokensAfter: $("mTokensAfter"),
    mReduction: $("mReduction"),
    mSimilarity: $("mSimilarity"),
    mDensity: $("mDensity"),
    mDecision: $("mDecision"),
    originalOutput: $("originalOutput"),
    compressedOutput: $("compressedOutput"),
    diffOutput: $("diffOutput"),
    btnCopy: $("btnCopy"),
    btnReplace: $("btnReplace"),
    successOverlay: $("successOverlay"),
    modeBtns: $$(".mode-btn"),
    qualityScore: $("qualityScore"),
    qualityBadge: $("qualityBadge"),
    qClarity: $("qClarity"),
    qSpecificity: $("qSpecificity"),
    qStructure: $("qStructure"),
    qEfficiency: $("qEfficiency"),
    templateSuggestionCard: $("templateSuggestionCard"),
    templateSuggestionText: $("templateSuggestionText"),
    btnCreateSuggestedTemplate: $("btnCreateSuggestedTemplate"),
    templateList: $("templateList"),
    templateEditor: $("templateEditor"),
    tplName: $("tplName"),
    tplPrompt: $("tplPrompt"),
    btnNewTemplate: $("btnNewTemplate"),
    btnSaveTemplate: $("btnSaveTemplate"),
    btnCancelTemplate: $("btnCancelTemplate"),
    variableFillDialog: $("variableFillDialog"),
    variableInputs: $("variableInputs"),
    btnApplyTemplate: $("btnApplyTemplate"),
    btnCancelFill: $("btnCancelFill"),
    btnImportTemplates: $("btnImportTemplates"),
    btnExportTemplates: $("btnExportTemplates"),
    templateImportInput: $("templateImportInput"),
    statPrompts: $("statPrompts"),
    statTokens: $("statTokens"),
    statCompression: $("statCompression"),
    statTemplates: $("statTemplates"),
    statLifetime: $("statLifetime"),
    chartContainer: $("chartContainer"),
    historyList: $("historyList"),
    btnClearHistory: $("btnClearHistory"),
    // Analyze tab
    analyzePrompt: $("analyzePrompt"),
    analyzeModeBtns: $$(".analyze-mode-btn"),
    analyzeIntentOverride: $("analyzeIntentOverride"),
    btnAnalyze: $("btnAnalyze"),
    btnAnalyzeText: $("btnAnalyzeText"),
    btnAnalyzeSpinner: $("btnAnalyzeSpinner"),
    analyzeError: $("analyzeError"),
    analyzeResults: $("analyzeResults"),
    analyzeIntent: $("analyzeIntent"),
    analyzeOptimized: $("analyzeOptimized"),
    btnCopyAnalyzed: $("btnCopyAnalyzed"),
    btnUseAnalyzed: $("btnUseAnalyzed"),
    comparisonTable: $("comparisonTable"),
    templateDetection: $("templateDetection"),
    analyzeSummary: $("analyzeSummary"),
    computeCompare: $("computeCompare"),
    analyzeMetaStats: $("analyzeMetaStats"),
  };

  const state = {
    mode: "optimize",
    analyzeMode: "both",
    intentOverride: "",
    analyzeIntentOverride: "",
    lastOptimizedPrompt: "",
    lastCompressed: "",
    lastAnalyzedOptimized: "",
    editingTemplateId: null,
    pendingTemplate: null,
    autoOptimizeTimer: null,
    ghostPrediction: "",
    predictRequestId: 0,
    debouncedPredict: null,
  };

  function init() {
    state.debouncedPredict = debounce(requestGhostPrediction, 400);
    bindTabs();
    bindOptimizeControls();
    bindAnalyzeControls();
    bindTemplateControls();
    bindHistoryControls();
    restoreSettings();
    checkHealth();
    loadTemplates();
    loadStats();
    updateQualityScore(refs.prompt.value || "");
  }

  function bindTabs() {
    refs.tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        switchTab(target);
        if (target === "templates") loadTemplates();
        if (target === "stats") loadStats();
      });
    });
  }

  function bindOptimizeControls() {
    refs.modeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        refs.modeBtns.forEach((item) => item.classList.toggle("active", item === btn));
        state.mode = btn.dataset.mode;
        saveSettings();
      });
    });

    refs.autoMode.addEventListener("change", () => {
      refs.aggr.disabled = refs.autoMode.checked;
      refs.aggrVal.textContent = refs.autoMode.checked ? "auto" : Number(refs.aggr.value).toFixed(2);
      saveSettings();
    });

    refs.aggr.addEventListener("input", () => {
      refs.aggrVal.textContent = Number(refs.aggr.value).toFixed(2);
      saveSettings();
    });

    refs.autoOptimizeToggle.addEventListener("change", saveSettings);
    refs.intentOverride.addEventListener("change", () => {
      const previousIntent = state.intentOverride;
      state.intentOverride = refs.intentOverride.value;
      // Keep analyze tab default aligned unless user explicitly changed it.
      if (!state.analyzeIntentOverride || state.analyzeIntentOverride === previousIntent) {
        state.analyzeIntentOverride = refs.intentOverride.value;
        refs.analyzeIntentOverride.value = state.analyzeIntentOverride;
      }
      saveSettings();
    });

    refs.prompt.addEventListener("input", () => {
      const prompt = refs.prompt.value;
      updateQualityScore(prompt);
      suggestTemplate(prompt);
      clearGhostPrediction();
      state.debouncedPredict(prompt);
      if (refs.autoOptimizeToggle.checked) {
        clearTimeout(state.autoOptimizeTimer);
        state.autoOptimizeTimer = setTimeout(() => {
          const trimmed = refs.prompt.value.trim();
          if (trimmed && trimmed !== state.lastOptimizedPrompt) {
            runOptimization();
          }
        }, 900);
      }
    });

    refs.prompt.addEventListener("keydown", (event) => {
      if (event.key === "Tab" && state.ghostPrediction) {
        event.preventDefault();
        acceptGhostPrediction();
      }
    });

    refs.prompt.addEventListener("scroll", renderGhostPrediction);
    refs.prompt.addEventListener("click", renderGhostPrediction);
    refs.prompt.addEventListener("keyup", (event) => {
      if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) {
        renderGhostPrediction();
      }
    });

    refs.btnOptimize.addEventListener("click", runOptimization);

    refs.btnCopy.addEventListener("click", async () => {
      if (!state.lastCompressed) return;
      await navigator.clipboard.writeText(state.lastCompressed);
      pulseButton(refs.btnCopy, "✅ Copied!", "📋 Copy");
    });

    refs.btnReplace.addEventListener("click", () => {
      if (!state.lastCompressed) return;
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs && tabs[0];
        if (!tab?.id) return;
        chrome.tabs.sendMessage(tab.id, { type: "REPLACE_PROMPT", text: state.lastCompressed });
        pulseButton(refs.btnReplace, "✅ Replaced!", "✅ Replace in Chat");
      });
    });

    refs.btnCreateSuggestedTemplate.addEventListener("click", () => {
      refs.tplName.value = createSuggestionName(refs.prompt.value.trim());
      refs.tplPrompt.value = refs.prompt.value.trim();
      refs.templateEditor.classList.remove("hidden");
      switchTab("templates");
    });
  }

  /* --------------------------------------------------------------- */
  /* Analyze tab                                                      */
  /* --------------------------------------------------------------- */

  function bindAnalyzeControls() {
    refs.analyzeModeBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        refs.analyzeModeBtns.forEach((b) => b.classList.toggle("active", b === btn));
        state.analyzeMode = btn.dataset.mode;
      });
    });

    refs.analyzeIntentOverride.addEventListener("change", () => {
      state.analyzeIntentOverride = refs.analyzeIntentOverride.value;
      saveSettings();
    });

    refs.btnAnalyze.addEventListener("click", runAnalysis);

    refs.btnCopyAnalyzed.addEventListener("click", async () => {
      if (!state.lastAnalyzedOptimized) return;
      await navigator.clipboard.writeText(state.lastAnalyzedOptimized);
      pulseButton(refs.btnCopyAnalyzed, "✅ Copied!", "📋 Copy");
    });

    refs.btnUseAnalyzed.addEventListener("click", () => {
      if (!state.lastAnalyzedOptimized) return;
      refs.prompt.value = state.lastAnalyzedOptimized;
      updateQualityScore(state.lastAnalyzedOptimized);
      switchTab("optimize");
    });
  }

  function runAnalysis() {
    const prompt = refs.analyzePrompt.value.trim();
    if (!prompt) {
      refs.analyzeError.textContent = "Please enter a prompt to analyze.";
      refs.analyzeError.classList.remove("hidden");
      return;
    }

    refs.analyzeError.classList.add("hidden");
    refs.btnAnalyze.disabled = true;
    refs.btnAnalyzeSpinner.classList.remove("hidden");
    refs.btnAnalyzeText.textContent = "Analyzing…";

    const payload = { prompt, mode: state.analyzeMode };
    if (state.analyzeIntentOverride) payload.intent_override = state.analyzeIntentOverride;

    chrome.runtime.sendMessage({ type: "ANALYZE", payload }, (res) => {
      refs.btnAnalyze.disabled = false;
      refs.btnAnalyzeSpinner.classList.add("hidden");
      refs.btnAnalyzeText.textContent = "Analyze Prompt";

      if (chrome.runtime.lastError) {
        refs.analyzeError.textContent = chrome.runtime.lastError.message;
        refs.analyzeError.classList.remove("hidden");
        return;
      }
      if (!res?.ok) {
        refs.analyzeError.textContent = res?.error || "Analysis failed.";
        refs.analyzeError.classList.remove("hidden");
        return;
      }

      renderAnalysis(res.data);
    });
  }

  function renderAnalysis(data) {
    refs.analyzeResults.classList.remove("hidden");
    state.lastAnalyzedOptimized = data.optimized_prompt || "";

    const intent = data.intent || data.intent_detail?.intent || "—";
    const confidence = Number(data.intent_detail?.intent_confidence || 0);
    refs.analyzeIntent.innerHTML = `
      <strong>${escapeHtml(intent)}</strong>
      <span class="meta">${confidence ? `${Math.round(confidence * 100)}% confidence` : "manual/auto-selected"}</span>
    `;

    // Optimized prompt
    refs.analyzeOptimized.textContent = data.optimized_prompt || "";

    // Top improvements only (reduced visual load)
    const comparison = (data.comparison || [])
      .map((d) => ({ ...d, delta: (d.optimized_score || 0) - (d.original_score || 0) }))
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 3);

    refs.comparisonTable.innerHTML = `
      <div class="cmp-row cmp-header">
        <span class="cmp-dim">What Improved</span>
        <span class="cmp-score">Gain</span>
      </div>
      ${comparison.map((d) => `
        <div class="cmp-row">
          <span class="cmp-dim">${escapeHtml(d.dimension)}</span>
          <span class="cmp-score"><span class="improvement-pill">+${d.delta}</span></span>
        </div>
      `).join("")}
    `;

    // Template detection (compact)
    const tpl = data.template || {};
    if (tpl.is_templatizable) {
      refs.templateDetection.innerHTML = `
        <div class="tpl-detected">
          <div class="tpl-name"><strong>Reusable:</strong> Yes (${escapeHtml(tpl.template_name)})</div>
          <div class="tpl-vars"><strong>Variables:</strong> ${tpl.variables.length || 0}</div>
          <button class="btn-small btn-primary tpl-save-btn" id="btnSaveDetectedTemplate">Save as Template</button>
        </div>
      `;

      const saveBtn = document.getElementById("btnSaveDetectedTemplate");
      if (saveBtn) {
        saveBtn.addEventListener("click", () => {
          refs.tplName.value = tpl.template_name || "Detected Template";
          refs.tplPrompt.value = tpl.template_structure.replace(/\{(\w+)\}/g, "{{$1}}");
          refs.templateEditor.classList.remove("hidden");
          switchTab("templates");
        });
      }
    } else {
      refs.templateDetection.innerHTML = '<div class="tpl-none">Not a reusable template pattern yet.</div>';
    }

    // Summary (shortened)
    refs.analyzeSummary.textContent = conciseText(data.summary || "No summary available.");

    // Compute score
    const co = data.compute_original || {};
    const cc = data.compute_optimized || {};
    const reduction = data.compute_reduction_percent ?? 0;
    refs.computeCompare.innerHTML = `
      <div class="compute-overall">
        <div class="compute-pill">
          <span class="compute-label">Original</span>
          <span class="compute-value compute-${computeTier(co.overall || 0)}">${co.overall || 0}/10</span>
        </div>
        <span class="compute-arrow">→</span>
        <div class="compute-pill">
          <span class="compute-label">Optimized</span>
          <span class="compute-value compute-${computeTier(cc.overall || 0)}">${cc.overall || 0}/10</span>
        </div>
        <div class="compute-reduction ${reduction > 0 ? 'positive' : ''}">${reduction > 0 ? '↓' : ''}${Math.abs(reduction).toFixed(1)}% compute</div>
      </div>
    `;

    // Meta stats
    refs.analyzeMetaStats.innerHTML = `
      <span>${data.original_token_count || 0} → ${data.compressed_token_count || 0} tokens</span>
      <span>Compression: ${((1 - (data.compression_ratio || 1)) * 100).toFixed(1)}%</span>
      <span>Mode: ${escapeHtml(data.mode || "both")}</span>
    `;
  }

  function scoreTier(score) {
    if (score >= 8) return "high";
    if (score >= 5) return "mid";
    return "low";
  }

  function computeTier(score) {
    if (score <= 3) return "low";
    if (score <= 6) return "mid";
    return "high";
  }

  function runOptimization() {
    const prompt = refs.prompt.value.trim();
    if (!prompt) {
      showError("Please enter a prompt first.");
      return;
    }

    hideError();
    setLoading(true);

    const payload = {
      prompt,
      mode: state.mode,
      auto_aggressiveness: refs.autoMode.checked,
    };
    if (state.intentOverride) payload.intent_override = state.intentOverride;
    if (!refs.autoMode.checked) payload.aggressiveness = Number(refs.aggr.value);

    chrome.runtime.sendMessage({ type: "OPTIMIZE", payload }, (res) => {
      setLoading(false);
      if (chrome.runtime.lastError) {
        showError(chrome.runtime.lastError.message);
        return;
      }
      if (!res?.ok) {
        showError(res?.error || "Unknown backend error.");
        return;
      }

      state.lastOptimizedPrompt = prompt;
      renderResults(prompt, res.data);
      updateStats(res.data);
      addToHistory(prompt, res.data);
      showSuccessAnimation();
    });
  }

  function requestGhostPrediction(text) {
    const currentText = String(text || "");
    if (!currentText.trim()) {
      clearGhostPrediction();
      return;
    }

    const requestId = ++state.predictRequestId;
    chrome.runtime.sendMessage({ type: "PREDICT", payload: { text: currentText } }, (res) => {
      if (requestId !== state.predictRequestId) return;

      if (chrome.runtime.lastError || !res?.ok) {
        clearGhostPrediction();
        return;
      }

      const prediction = String(res?.data?.prediction || "");
      if (!prediction.trim()) {
        clearGhostPrediction();
        return;
      }

      state.ghostPrediction = prediction;
      renderGhostPrediction();
    });
  }

  function acceptGhostPrediction() {
    const prediction = state.ghostPrediction;
    if (!prediction) return;

    const start = refs.prompt.selectionStart ?? refs.prompt.value.length;
    const end = refs.prompt.selectionEnd ?? start;
    const current = refs.prompt.value;
    refs.prompt.value = `${current.slice(0, start)}${prediction}${current.slice(end)}`;

    const nextCaret = start + prediction.length;
    refs.prompt.selectionStart = nextCaret;
    refs.prompt.selectionEnd = nextCaret;

    updateQualityScore(refs.prompt.value);
    suggestTemplate(refs.prompt.value);
    clearGhostPrediction();
    state.debouncedPredict(refs.prompt.value);
  }

  function renderGhostPrediction() {
    if (!state.ghostPrediction) {
      refs.promptGhostText.classList.add("hidden");
      return;
    }

    if (document.activeElement !== refs.prompt) {
      refs.promptGhostText.classList.add("hidden");
      return;
    }

    const caretPosition = refs.prompt.selectionStart ?? refs.prompt.value.length;
    const caret = getCaretCoordinates(refs.prompt, caretPosition);

    refs.promptGhostText.textContent = state.ghostPrediction;
    refs.promptGhostText.style.left = `${caret.left}px`;
    refs.promptGhostText.style.top = `${caret.top}px`;
    refs.promptGhostText.classList.remove("hidden");
  }

  function clearGhostPrediction() {
    state.ghostPrediction = "";
    refs.promptGhostText.textContent = "";
    refs.promptGhostText.classList.add("hidden");
  }

  function getCaretCoordinates(textarea, position) {
    const div = document.createElement("div");
    const style = window.getComputedStyle(textarea);
    const properties = [
      "boxSizing",
      "width",
      "height",
      "overflowX",
      "overflowY",
      "borderTopWidth",
      "borderRightWidth",
      "borderBottomWidth",
      "borderLeftWidth",
      "paddingTop",
      "paddingRight",
      "paddingBottom",
      "paddingLeft",
      "fontStyle",
      "fontVariant",
      "fontWeight",
      "fontStretch",
      "fontSize",
      "lineHeight",
      "fontFamily",
      "textAlign",
      "textTransform",
      "textIndent",
      "textDecoration",
      "letterSpacing",
      "wordSpacing",
      "tabSize",
      "MozTabSize",
      "whiteSpace",
      "wordBreak",
    ];

    div.style.position = "absolute";
    div.style.visibility = "hidden";
    div.style.whiteSpace = "pre-wrap";
    div.style.wordWrap = "break-word";

    properties.forEach((prop) => {
      div.style[prop] = style[prop];
    });

    div.textContent = textarea.value.substring(0, position);
    const span = document.createElement("span");
    span.textContent = textarea.value.substring(position) || ".";
    div.appendChild(span);
    document.body.appendChild(div);

    const left = span.offsetLeft - textarea.scrollLeft + parseFloat(style.borderLeftWidth || "0");
    const top = span.offsetTop - textarea.scrollTop + parseFloat(style.borderTopWidth || "0");

    document.body.removeChild(div);
    return { left, top };
  }

  function renderResults(originalPrompt, data) {
    refs.results.classList.remove("hidden");
    state.lastCompressed = data.compressed_prompt || "";

    refs.intentBadge.textContent = data.intent?.intent || "—";
    refs.intentMeta.textContent = `${((data.intent?.intent_confidence || 0) * 100).toFixed(0)}% confidence · ${data.mode || state.mode}`;
    refs.mTokensBefore.textContent = String(data.original_token_count ?? 0);
    refs.mTokensAfter.textContent = String(data.compressed_token_count ?? 0);
    refs.mReduction.textContent = `${Number(data.token_reduction_percent || 0).toFixed(1)}%`;
    refs.mSimilarity.textContent = Number(data.evaluation?.semantic_similarity || 0).toFixed(4);
    refs.mDensity.textContent = Number(data.density?.density_score || 0).toFixed(4);

    const decision = data.decision?.decision || "—";
    refs.mDecision.innerHTML = `<span class="badge ${badgeClass(decision)}">${escapeHtml(decision)}</span>`;
    refs.originalOutput.textContent = originalPrompt;
    refs.compressedOutput.textContent = data.compressed_prompt || "";
    refs.diffOutput.innerHTML = renderDiff(data.diff || {});
  }

  function renderDiff(diff) {
    const removed = Array.isArray(diff.removed) ? diff.removed : [];
    const rewritten = Array.isArray(diff.rewritten) ? diff.rewritten : [];
    if (!removed.length && !rewritten.length) {
      return '<div class="diff-empty">No major changes detected.</div>';
    }

    const parts = [];
    if (removed.length) {
      parts.push(`<div class="diff-group"><div class="diff-title">Removed</div>${removed.slice(0, 8).map((item) => `<span class="diff-chip removed">${escapeHtml(item)}</span>`).join("")}</div>`);
    }
    if (rewritten.length) {
      parts.push(`<div class="diff-group"><div class="diff-title">Rewritten</div>${rewritten.slice(0, 6).map((item) => `<div class="diff-row"><span class="diff-old">${escapeHtml(item.original || "")}</span><span class="diff-arrow">→</span><span class="diff-new">${escapeHtml(item.compressed || "")}</span></div>`).join("")}</div>`);
    }
    return parts.join("");
  }

  function updateQualityScore(prompt) {
    const quality = evaluatePromptQuality(prompt);
    refs.qualityScore.textContent = String(quality.total);
    refs.qClarity.textContent = String(quality.clarity);
    refs.qSpecificity.textContent = String(quality.specificity);
    refs.qStructure.textContent = String(quality.structure);
    refs.qEfficiency.textContent = String(quality.efficiency);
    refs.qualityBadge.textContent = quality.label;
    refs.qualityBadge.dataset.tier = quality.tier;
  }

  function evaluatePromptQuality(prompt) {
    const text = (prompt || "").trim();
    if (!text) return { total: 0, clarity: 0, specificity: 0, structure: 0, efficiency: 0, label: "Needs work", tier: "low" };

    const words = text.split(/\s+/).filter(Boolean);
    const sentences = text.split(/[.!?]\s+/).filter(Boolean);
    const hasGoalVerb = /(explain|write|build|compare|analyze|summarize|debug|design|solve|answer|describe)/i.test(text);
    const hasConstraints = /(step-by-step|examples?|bullet|table|format|json|markdown|concise|detailed|tone|style)/i.test(text);
    const hasStructure = /\n|:|\-|\d+\./.test(text);
    const fillerHits = (text.match(/please|kindly|i would like|can you|could you|would you|just/gi) || []).length;

    const clarity = clampScore((hasGoalVerb ? 30 : 12) + Math.min(words.length, 25) + (sentences.length <= 3 ? 15 : 5) - fillerHits * 4);
    const specificity = clampScore((hasConstraints ? 35 : 10) + Math.min((text.match(/\d+/g) || []).length * 10, 20) + Math.min(words.length, 35));
    const structure = clampScore((hasStructure ? 40 : 12) + (text.includes("\n") ? 20 : 0) + (/:/.test(text) ? 15 : 0));
    const efficiency = clampScore(100 - Math.max(0, words.length - 90) - fillerHits * 8);
    const total = Math.round((clarity + specificity + structure + efficiency) / 4);

    return {
      total,
      clarity,
      specificity,
      structure,
      efficiency,
      label: total >= 85 ? "Excellent" : total >= 70 ? "Strong" : total >= 50 ? "Fair" : "Needs work",
      tier: total >= 85 ? "high" : total >= 70 ? "mid" : "low",
    };
  }

  function bindTemplateControls() {
    refs.btnNewTemplate.addEventListener("click", () => openTemplateEditor());
    refs.btnCancelTemplate.addEventListener("click", closeTemplateEditor);
    refs.btnSaveTemplate.addEventListener("click", saveTemplate);
    refs.btnCancelFill.addEventListener("click", () => refs.variableFillDialog.classList.add("hidden"));
    refs.btnApplyTemplate.addEventListener("click", applyPendingTemplate);
    refs.btnExportTemplates.addEventListener("click", exportTemplates);
    refs.btnImportTemplates.addEventListener("click", () => refs.templateImportInput.click());
    refs.templateImportInput.addEventListener("change", importTemplates);
  }

  function openTemplateEditor(template = null) {
    state.editingTemplateId = template?.id || null;
    refs.tplName.value = template?.name || "";
    refs.tplPrompt.value = template?.prompt || refs.prompt.value.trim();
    refs.templateEditor.classList.remove("hidden");
  }

  function closeTemplateEditor() {
    state.editingTemplateId = null;
    refs.templateEditor.classList.add("hidden");
  }

  function loadTemplates() {
    chrome.storage.local.get([STORAGE_KEYS.templates], (items) => {
      renderTemplateList(items[STORAGE_KEYS.templates] || []);
    });
  }

  function renderTemplateList(templates) {
    if (!templates.length) {
      refs.templateList.innerHTML = '<div class="empty-state">No templates yet. Create one to get started.</div>';
      return;
    }

    refs.templateList.innerHTML = templates.map((template) => `
      <div class="template-item">
        <div class="template-info">
          <div class="template-name">${escapeHtml(template.name)}</div>
          <div class="template-preview">${escapeHtml(template.prompt.slice(0, 100))}</div>
        </div>
        <div class="template-actions">
          <button class="btn-use" data-action="use" data-id="${template.id}">Use</button>
          <button data-action="edit" data-id="${template.id}">Edit</button>
          <button class="btn-del" data-action="delete" data-id="${template.id}">Delete</button>
        </div>
      </div>
    `).join("");

    refs.templateList.querySelectorAll("button[data-action]").forEach((button) => {
      button.addEventListener("click", () => handleTemplateAction(button.dataset.action, button.dataset.id));
    });
  }

  function handleTemplateAction(action, templateId) {
    chrome.storage.local.get([STORAGE_KEYS.templates], (items) => {
      const templates = items[STORAGE_KEYS.templates] || [];
      const template = templates.find((item) => item.id === templateId);
      if (!template && action !== "delete") return;

      if (action === "use") useTemplate(template);
      if (action === "edit") openTemplateEditor(template);
      if (action === "delete") {
        const next = templates.filter((item) => item.id !== templateId);
        chrome.storage.local.set({ [STORAGE_KEYS.templates]: next }, () => renderTemplateList(next));
      }
    });
  }

  function saveTemplate() {
    const name = refs.tplName.value.trim();
    const prompt = refs.tplPrompt.value.trim();
    if (!name || !prompt) {
      showError("Template name and prompt are required.");
      return;
    }

    chrome.storage.local.get([STORAGE_KEYS.templates], (items) => {
      const templates = items[STORAGE_KEYS.templates] || [];
      const payload = {
        id: state.editingTemplateId || `tpl_${Date.now()}`,
        name,
        prompt,
        created_at: new Date().toISOString(),
      };

      const next = state.editingTemplateId
        ? templates.map((item) => item.id === state.editingTemplateId ? { ...item, name, prompt } : item)
        : [payload, ...templates];

      chrome.storage.local.set({ [STORAGE_KEYS.templates]: next }, () => {
        closeTemplateEditor();
        renderTemplateList(next);
      });
    });
  }

  function useTemplate(template) {
    const variables = [...new Set((template.prompt.match(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g) || []).map((match) => match.replace(/[{}\s]/g, "")))];
    if (!variables.length) {
      applyTemplateText(template.prompt);
      incrementTemplateUsage();
      return;
    }

    state.pendingTemplate = template;
    refs.variableInputs.innerHTML = variables.map((variable) => `
      <div class="var-input-group">
        <label class="label">${escapeHtml(variable)}</label>
        <textarea rows="3" data-variable="${escapeHtml(variable)}" placeholder="Enter ${escapeHtml(variable)}…"></textarea>
      </div>
    `).join("");
    refs.variableFillDialog.classList.remove("hidden");
  }

  function applyPendingTemplate() {
    if (!state.pendingTemplate) return;
    let rendered = state.pendingTemplate.prompt;
    refs.variableInputs.querySelectorAll("textarea[data-variable]").forEach((field) => {
      rendered = rendered.replaceAll(`{{${field.dataset.variable}}}`, field.value);
    });
    refs.variableFillDialog.classList.add("hidden");
    applyTemplateText(rendered);
    incrementTemplateUsage();
    state.pendingTemplate = null;
  }

  function applyTemplateText(text) {
    refs.prompt.value = text;
    updateQualityScore(text);
    switchTab("optimize");
  }

  function exportTemplates() {
    chrome.storage.local.get([STORAGE_KEYS.templates], (items) => {
      const blob = new Blob([JSON.stringify(items[STORAGE_KEYS.templates] || [], null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "optiprompt-templates.json";
      link.click();
      URL.revokeObjectURL(url);
    });
  }

  function importTemplates(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    file.text().then((text) => {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) throw new Error("Imported file must contain a template array.");
      chrome.storage.local.set({ [STORAGE_KEYS.templates]: parsed }, loadTemplates);
    }).catch((err) => showError(`Import failed: ${err.message}`)).finally(() => {
      refs.templateImportInput.value = "";
    });
  }

  function bindHistoryControls() {
    refs.btnClearHistory.addEventListener("click", () => {
      chrome.storage.local.remove([STORAGE_KEYS.history, STORAGE_KEYS.stats], loadStats);
    });
  }

  function addToHistory(originalPrompt, data) {
    chrome.storage.local.get([STORAGE_KEYS.history], (items) => {
      const history = items[STORAGE_KEYS.history] || [];
      history.push({
        original_prompt: originalPrompt,
        optimized_prompt: data.compressed_prompt || "",
        timestamp: Date.now(),
        mode_used: data.mode || state.mode,
        tokens_before: data.original_token_count || 0,
        tokens_after: data.compressed_token_count || 0,
        tokens_saved: Math.max((data.original_token_count || 0) - (data.compressed_token_count || 0), 0),
        compression: data.token_reduction_percent || 0,
      });
      while (history.length > 20) history.shift();
      chrome.storage.local.set({ [STORAGE_KEYS.history]: history }, loadStats);
    });
  }

  function updateStats(data) {
    chrome.storage.local.get([STORAGE_KEYS.stats], (items) => {
      const stats = items[STORAGE_KEYS.stats] || {
        total_prompts_processed: 0,
        tokens_saved: 0,
        total_compression_sum: 0,
        total_templates_used: 0,
        lifetime_prompt_usage: 0,
      };
      stats.total_prompts_processed += 1;
      stats.tokens_saved += Math.max((data.original_token_count || 0) - (data.compressed_token_count || 0), 0);
      stats.total_compression_sum += Number(data.token_reduction_percent || 0);
      stats.lifetime_prompt_usage += 1;
      chrome.storage.local.set({ [STORAGE_KEYS.stats]: stats });
    });
  }

  function incrementTemplateUsage() {
    chrome.storage.local.get([STORAGE_KEYS.stats], (items) => {
      const stats = items[STORAGE_KEYS.stats] || {
        total_prompts_processed: 0,
        tokens_saved: 0,
        total_compression_sum: 0,
        total_templates_used: 0,
        lifetime_prompt_usage: 0,
      };
      stats.total_templates_used += 1;
      chrome.storage.local.set({ [STORAGE_KEYS.stats]: stats }, loadStats);
    });
  }

  function loadStats() {
    chrome.storage.local.get([STORAGE_KEYS.stats, STORAGE_KEYS.history], (items) => {
      const stats = items[STORAGE_KEYS.stats] || {
        total_prompts_processed: 0,
        tokens_saved: 0,
        total_compression_sum: 0,
        total_templates_used: 0,
        lifetime_prompt_usage: 0,
      };
      const history = items[STORAGE_KEYS.history] || [];

      refs.statPrompts.textContent = String(stats.total_prompts_processed || 0);
      refs.statTokens.textContent = Number(stats.tokens_saved || 0).toLocaleString();
      refs.statCompression.textContent = `${stats.total_prompts_processed ? (stats.total_compression_sum / stats.total_prompts_processed).toFixed(1) : "0.0"}%`;
      refs.statTemplates.textContent = String(stats.total_templates_used || 0);
      refs.statLifetime.textContent = String(stats.lifetime_prompt_usage || 0);

      renderChart(history);
      renderHistory(history);
    });
  }

  function renderChart(history) {
    if (!history.length) {
      refs.chartContainer.innerHTML = '<div class="empty-state">No optimization history yet.</div>';
      return;
    }
    const recent = history.slice(-12);
    const maxSaved = Math.max(...recent.map((item) => item.tokens_saved || 1), 1);
    refs.chartContainer.innerHTML = `
      <div class="chart-bars">
        ${recent.map((item, index) => `<div class="chart-bar" style="height:${Math.max(10, ((item.tokens_saved || 0) / maxSaved) * 100)}%" data-label="${index + 1}" title="${item.tokens_saved || 0} tokens saved"></div>`).join("")}
      </div>
      <div class="chart-legend"><span>Older</span><span>Saved tokens</span><span>Newer</span></div>
    `;
  }

  function renderHistory(history) {
    if (!history.length) {
      refs.historyList.innerHTML = '<div class="empty-state">No history yet.</div>';
      return;
    }
    refs.historyList.innerHTML = [...history].reverse().map((entry, index) => `
      <div class="history-item">
        <div class="history-item-header">
          <span class="history-mode">${escapeHtml(entry.mode_used || "optimize")}</span>
          <span class="history-time">${formatTime(entry.timestamp)}</span>
        </div>
        <div class="history-prompt">${escapeHtml(entry.original_prompt)}</div>
        <div class="history-stats">Saved <span>${entry.tokens_saved || 0}</span> tokens · <span>${Number(entry.compression || 0).toFixed(1)}%</span></div>
        <div class="history-actions">
          <button class="btn-small btn-secondary" data-history-action="reuse" data-index="${index}">Reuse</button>
          <button class="btn-small btn-secondary" data-history-action="copy" data-index="${index}">Copy</button>
        </div>
      </div>
    `).join("");

    refs.historyList.querySelectorAll("button[data-history-action]").forEach((button) => {
      button.addEventListener("click", async () => {
        const reversed = [...history].reverse();
        const entry = reversed[Number(button.dataset.index)];
        if (!entry) return;
        if (button.dataset.historyAction === "reuse") {
          refs.prompt.value = entry.original_prompt;
          updateQualityScore(entry.original_prompt);
          switchTab("optimize");
        } else {
          await navigator.clipboard.writeText(entry.optimized_prompt || entry.original_prompt);
          pulseButton(button, "Copied", "Copy");
        }
      });
    });
  }

  function suggestTemplate(prompt) {
    const text = (prompt || "").trim();
    if (!text || text.length < 30) {
      refs.templateSuggestionCard.classList.add("hidden");
      return;
    }

    chrome.storage.local.get([STORAGE_KEYS.history], (items) => {
      const history = items[STORAGE_KEYS.history] || [];
      const seed = normalizePromptSeed(text);
      const matches = history.filter((item) => normalizePromptSeed(item.original_prompt || "") === seed);
      if (matches.length >= 2) {
        refs.templateSuggestionText.textContent = `You have used a similar prompt ${matches.length + 1} times. Save it as a template?`;
        refs.templateSuggestionCard.classList.remove("hidden");
      } else {
        refs.templateSuggestionCard.classList.add("hidden");
      }
    });
  }

  function restoreSettings() {
    chrome.storage.local.get([STORAGE_KEYS.settings], (items) => {
      const settings = items[STORAGE_KEYS.settings] || {};
      refs.autoMode.checked = settings.autoAggressiveness ?? true;
      refs.autoOptimizeToggle.checked = settings.autoOptimize ?? false;
      refs.aggr.value = settings.aggressiveness ?? 0.3;
      refs.aggr.disabled = refs.autoMode.checked;
      refs.aggrVal.textContent = refs.autoMode.checked ? "auto" : Number(refs.aggr.value).toFixed(2);
      state.mode = settings.mode || "optimize";
      state.intentOverride = settings.intentOverride || "";
      state.analyzeIntentOverride = settings.analyzeIntentOverride || state.intentOverride;
      refs.intentOverride.value = state.intentOverride;
      refs.analyzeIntentOverride.value = state.analyzeIntentOverride;
      refs.modeBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === state.mode));
    });
  }

  function saveSettings() {
    chrome.storage.local.set({
      [STORAGE_KEYS.settings]: {
        mode: state.mode,
        aggressiveness: Number(refs.aggr.value),
        autoAggressiveness: refs.autoMode.checked,
        autoOptimize: refs.autoOptimizeToggle.checked,
        intentOverride: state.intentOverride,
        analyzeIntentOverride: state.analyzeIntentOverride,
      },
    });
  }

  function checkHealth() {
    chrome.runtime.sendMessage({ type: "HEALTH_CHECK" }, (res) => {
      const online = !chrome.runtime.lastError && !!res?.ok;
      refs.statusDot.className = `status-dot ${online ? "online" : "offline"}`;
      refs.statusDot.title = online ? "Backend online" : "Backend offline";
    });
  }

  function switchTab(target) {
    refs.tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === target));
    refs.tabContents.forEach((content) => content.classList.toggle("active", content.id === `tab-${target}`));
  }

  function setLoading(on) {
    refs.btnOptimize.disabled = on;
    refs.btnSpinner.classList.toggle("hidden", !on);
    refs.btnText.textContent = on ? "Optimizing…" : "Optimize Prompt";
  }

  function showError(message) {
    refs.errorBanner.textContent = message;
    refs.errorBanner.classList.remove("hidden");
  }

  function hideError() {
    refs.errorBanner.classList.add("hidden");
  }

  function showSuccessAnimation() {
    refs.successOverlay.classList.remove("hidden");
    setTimeout(() => refs.successOverlay.classList.add("hidden"), 700);
  }

  function pulseButton(button, temporaryText, defaultText) {
    const original = defaultText || button.textContent;
    button.textContent = temporaryText;
    setTimeout(() => { button.textContent = original; }, 1300);
  }

  function badgeClass(decision) {
    if (decision === "APPROVE") return "badge-approve";
    if (decision === "REJECT") return "badge-reject";
    return "badge-conservative";
  }

  function normalizePromptSeed(text) {
    return (text || "")
      .toLowerCase()
      .replace(/\d+/g, "#")
      .replace(/".*?"|'.*?'/g, "{{value}}")
      .split(/\s+/)
      .slice(0, 6)
      .join(" ");
  }

  function createSuggestionName(text) {
    return text.split(/\s+/).slice(0, 3).map((part) => part[0]?.toUpperCase() + part.slice(1)).join(" ") || "Reusable Template";
  }

  function clampScore(value) {
    return Math.max(0, Math.min(100, Math.round(value)));
  }

  function conciseText(text) {
    const normalized = String(text || "").trim().replace(/\s+/g, " ");
    if (!normalized) return "";
    const firstSentence = normalized.split(/(?<=[.!?])\s+/)[0] || normalized;
    return firstSentence.length <= 180 ? firstSentence : `${firstSentence.slice(0, 177)}...`;
  }

  function debounce(fn, delay) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), delay);
    };
  }

  function formatTime(timestamp) {
    const diff = Date.now() - timestamp;
    if (diff < 60000) return "Just now";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return new Date(timestamp).toLocaleDateString();
  }

  function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }

  init();
})();
