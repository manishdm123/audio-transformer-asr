function parseInterval(value) {
  const match = /^every\s+(\d+(?:\.\d+)?)s$/.exec(value || "");
  if (!match) return null;
  return Number(match[1]) * 1000;
}

function elementsMatching(root, selector) {
  const matches = [];
  if (root.matches?.(selector)) matches.push(root);
  matches.push(...(root.querySelectorAll?.(selector) || []));
  return matches;
}

function wirePolling(root = document) {
  elementsMatching(root, "[hx-get][hx-trigger][hx-swap='outerHTML']").forEach((element) => {
    if (element.dataset.polling === "true") return;
    const interval = parseInterval(element.getAttribute("hx-trigger"));
    const url = element.getAttribute("hx-get");
    if (!interval || !url) return;

    element.dataset.polling = "true";
    const timer = window.setInterval(async () => {
      if (!document.body.contains(element)) {
        window.clearInterval(timer);
        return;
      }
      const response = await fetch(url, {
        cache: "no-store",
        headers: { "X-Requested-With": "fetch" },
      });
      if (!response.ok) {
        console.warn(`Transcript polling failed with HTTP ${response.status}.`);
        return;
      }
      const html = await response.text();
      const replacement = elementFromHtml(html);
      if (!replacement) return;
      element.replaceWith(replacement);
      window.clearInterval(timer);
      wireApp(replacement);
    }, interval);
  });
}

function wireStageElapsed(root = document) {
  elementsMatching(root, "[data-stage-elapsed]").forEach((element) => {
    if (element.dataset.elapsedWired === "true") return;
    element.dataset.elapsedWired = "true";
    const startedAt = new Date(element.dataset.startedAt);
    if (Number.isNaN(startedAt.getTime())) return;
    if (element.dataset.running !== "true") {
      element.textContent = "complete";
      return;
    }

    const update = () => {
      if (!document.body.contains(element)) {
        window.clearInterval(timer);
        return;
      }
      const elapsedSeconds = Math.max(0, Math.floor((Date.now() - startedAt.getTime()) / 1000));
      element.textContent = `${formatClock(elapsedSeconds)} elapsed`;
    };
    const timer = window.setInterval(update, 1000);
    update();
  });
}

function elementFromHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

function formatClock(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const base = `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  return hours ? `${String(hours).padStart(2, "0")}:${base}` : base;
}

async function apiRequest(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "The edit could not be saved.");
  }
  return payload;
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.max(textarea.scrollHeight, 56)}px`;
}

function setFeedback(panel, message, isError = false) {
  const feedback = panel.querySelector("[data-editor-feedback]");
  if (!feedback) return;
  feedback.textContent = message;
  feedback.classList.toggle("is-error", isError);
}

function dirtyTextareas(panel) {
  return [...panel.querySelectorAll("[data-segment-text]")].filter(
    (textarea) => textarea.value.trim() !== textarea.dataset.original,
  );
}

async function saveTextareas(panel, textareas, quiet = false) {
  if (!textareas.length) {
    if (!quiet) setFeedback(panel, "Everything is already saved.");
    return true;
  }

  const edits = textareas.map((textarea) => ({
    index: Number(textarea.closest("[data-segment-index]").dataset.segmentIndex),
    text: textarea.value,
  }));

  try {
    const payload = await apiRequest(`/jobs/${panel.dataset.jobId}/segments`, {
      method: "PUT",
      body: JSON.stringify({ edits }),
    });
    textareas.forEach((textarea) => {
      textarea.value = textarea.value.trim();
      textarea.dataset.original = textarea.value;
      textarea.closest(".segment").classList.remove("is-dirty");
      autoResize(textarea);
    });
    if (!quiet) setFeedback(panel, payload.message);
    return true;
  } catch (error) {
    setFeedback(panel, error.message, true);
    return false;
  }
}

async function refreshPanel(panel, { message = "", focusIndex = null } = {}) {
  const player = panel.querySelector("[data-audio-player]");
  const playbackState = player
    ? { currentTime: player.currentTime, shouldPlay: !player.paused }
    : { currentTime: 0, shouldPlay: false };
  const follow = panel.querySelector("[data-follow-playback]")?.checked ?? true;

  const response = await fetch(`/jobs/${panel.dataset.jobId}/panel`, {
    cache: "no-store",
    headers: { "X-Requested-With": "fetch" },
  });
  if (!response.ok) throw new Error("Could not refresh the transcript editor.");
  const replacement = elementFromHtml(await response.text());
  if (!replacement) throw new Error("The refreshed transcript was empty.");
  panel.replaceWith(replacement);
  wireApp(replacement);

  const newFollow = replacement.querySelector("[data-follow-playback]");
  if (newFollow) newFollow.checked = follow;
  const newPlayer = replacement.querySelector("[data-audio-player]");
  if (newPlayer) {
    const restorePlayback = () => {
      newPlayer.currentTime = Math.min(playbackState.currentTime, newPlayer.duration || playbackState.currentTime);
      if (playbackState.shouldPlay) newPlayer.play().catch(() => {});
    };
    if (newPlayer.readyState >= 1) restorePlayback();
    else newPlayer.addEventListener("loadedmetadata", restorePlayback, { once: true });
  }

  if (message) setFeedback(replacement, message);
  if (focusIndex !== null && focusIndex !== undefined) {
    const target = replacement.querySelector(`[data-segment-index="${focusIndex}"]`);
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
    target?.querySelector("[data-segment-text]")?.focus({ preventScroll: true });
  }
  return replacement;
}

function wireAudioSync(panel) {
  const player = panel.querySelector("[data-audio-player]");
  if (!player) return;
  const segments = [...panel.querySelectorAll("[data-segment-index]")];
  const timeOutput = panel.querySelector("[data-playback-time]");
  let activeSegment = null;

  const sync = () => {
    if (timeOutput) timeOutput.textContent = formatClock(player.currentTime);
    const active = segments.find((segment) => {
      const start = Number(segment.dataset.start);
      const end = Number(segment.dataset.end);
      return player.currentTime >= start && player.currentTime < end;
    });
    if (active === activeSegment) return;
    activeSegment?.classList.remove("is-active");
    activeSegment = active || null;
    activeSegment?.classList.add("is-active");
    if (activeSegment && panel.querySelector("[data-follow-playback]")?.checked) {
      const focused = document.activeElement?.matches?.("[data-segment-text]");
      if (!focused) activeSegment.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  player.addEventListener("timeupdate", sync);
  player.addEventListener("seeked", sync);
  player.addEventListener("ended", () => {
    activeSegment?.classList.remove("is-active");
    activeSegment = null;
  });
  panel.querySelectorAll("[data-seek]").forEach((button) => {
    button.addEventListener("click", () => {
      player.currentTime = Number(button.dataset.seek);
      player.play().catch(() => {});
      sync();
    });
  });
  panel.querySelectorAll("[data-skip]").forEach((button) => {
    button.addEventListener("click", () => {
      player.currentTime = Math.max(0, Math.min(player.duration || Infinity, player.currentTime + Number(button.dataset.skip)));
      sync();
    });
  });
}

function wireTextEditing(panel) {
  const textareas = [...panel.querySelectorAll("[data-segment-text]")];
  textareas.forEach((textarea) => {
    textarea.dataset.original = textarea.value.trim();
    autoResize(textarea);
    textarea.addEventListener("input", () => {
      autoResize(textarea);
      textarea.closest(".segment").classList.toggle(
        "is-dirty",
        textarea.value.trim() !== textarea.dataset.original,
      );
    });
    textarea.addEventListener("keydown", async (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        await saveTextareas(panel, [textarea]);
      }
    });
  });

  panel.querySelector("[data-save-all]")?.addEventListener("click", async () => {
    await saveTextareas(panel, dirtyTextareas(panel));
  });
  panel.querySelectorAll("[data-save-segment]").forEach((button) => {
    button.addEventListener("click", async () => {
      const textarea = button.closest(".segment").querySelector("[data-segment-text]");
      await saveTextareas(panel, [textarea]);
    });
  });

  panel.querySelectorAll("[data-split-segment]").forEach((button) => {
    button.addEventListener("click", async () => {
      const segment = button.closest("[data-segment-index]");
      const textarea = segment.querySelector("[data-segment-text]");
      const otherDirty = dirtyTextareas(panel).filter((candidate) => candidate !== textarea);
      if (!(await saveTextareas(panel, otherDirty, true))) return;
      try {
        const payload = await apiRequest(
          `/jobs/${panel.dataset.jobId}/segments/${segment.dataset.segmentIndex}/split`,
          {
            method: "POST",
            body: JSON.stringify({ character_offset: textarea.selectionStart, text: textarea.value }),
          },
        );
        await refreshPanel(panel, { message: payload.message, focusIndex: payload.focus_index });
      } catch (error) {
        setFeedback(panel, error.message, true);
      }
    });
  });

  panel.querySelectorAll("[data-merge-segment]").forEach((button) => {
    button.addEventListener("click", async () => {
      if (!(await saveTextareas(panel, dirtyTextareas(panel), true))) return;
      const segment = button.closest("[data-segment-index]");
      try {
        const payload = await apiRequest(
          `/jobs/${panel.dataset.jobId}/segments/${segment.dataset.segmentIndex}/merge`,
          { method: "POST", body: JSON.stringify({ direction: button.dataset.mergeSegment }) },
        );
        await refreshPanel(panel, { message: payload.message, focusIndex: payload.focus_index });
      } catch (error) {
        setFeedback(panel, error.message, true);
      }
    });
  });
}

function wireFindReplace(panel) {
  const search = panel.querySelector("[data-search-text]");
  const replacement = panel.querySelector("[data-replacement-text]");
  const matchCase = panel.querySelector("[data-match-case]");
  if (!search || !replacement) return;
  let cursor = -1;

  const clearMatches = () => {
    panel.querySelectorAll(".segment.is-match").forEach((segment) => segment.classList.remove("is-match"));
  };
  search.addEventListener("input", () => {
    cursor = -1;
    clearMatches();
  });

  panel.querySelector("[data-find-next]")?.addEventListener("click", () => {
    const needle = matchCase.checked ? search.value : search.value.toLocaleLowerCase();
    if (!needle) {
      setFeedback(panel, "Enter text to find.", true);
      return;
    }
    const textareas = [...panel.querySelectorAll("[data-segment-text]")];
    for (let step = 1; step <= textareas.length; step += 1) {
      const index = (cursor + step) % textareas.length;
      const haystack = matchCase.checked ? textareas[index].value : textareas[index].value.toLocaleLowerCase();
      const matchIndex = haystack.indexOf(needle);
      if (matchIndex >= 0) {
        clearMatches();
        cursor = index;
        const segment = textareas[index].closest(".segment");
        segment.classList.add("is-match");
        segment.scrollIntoView({ behavior: "smooth", block: "center" });
        textareas[index].focus({ preventScroll: true });
        textareas[index].setSelectionRange(matchIndex, matchIndex + search.value.length);
        setFeedback(panel, `Match found in segment ${index + 1}.`);
        return;
      }
    }
    setFeedback(panel, "No matches found.", true);
  });

  panel.querySelector("[data-replace-all]")?.addEventListener("click", async () => {
    if (!(await saveTextareas(panel, dirtyTextareas(panel), true))) return;
    try {
      const payload = await apiRequest(`/jobs/${panel.dataset.jobId}/replace`, {
        method: "POST",
        body: JSON.stringify({
          search: search.value,
          replacement: replacement.value,
          match_case: matchCase.checked,
        }),
      });
      await refreshPanel(panel, { message: payload.message });
    } catch (error) {
      setFeedback(panel, error.message, true);
    }
  });
}

function wireSpeakerEditing(panel) {
  panel.querySelectorAll("[data-speaker-rename]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!(await saveTextareas(panel, dirtyTextareas(panel), true))) return;
      const input = form.querySelector("input[type='text']");
      try {
        const payload = await apiRequest(`/jobs/${panel.dataset.jobId}/speakers/rename`, {
          method: "POST",
          body: JSON.stringify({ current: form.dataset.currentSpeaker, replacement: input.value }),
        });
        await refreshPanel(panel, { message: payload.message });
      } catch (error) {
        setFeedback(panel, error.message, true);
      }
    });
  });
}

function wireEditor(root = document) {
  const panel = root.matches?.("#job-panel") ? root : root.querySelector?.("#job-panel");
  if (!panel || panel.dataset.editorWired === "true") return;
  panel.dataset.editorWired = "true";
  wireAudioSync(panel);
  wireTextEditing(panel);
  wireFindReplace(panel);
  wireSpeakerEditing(panel);
}

function wireApp(root = document) {
  wirePolling(root);
  wireStageElapsed(root);
  wireEditor(root);
}

window.addEventListener("DOMContentLoaded", () => wireApp(document));
