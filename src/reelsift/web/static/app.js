"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { trip: null, length: null, edit: null, poolOffset: 0, poolRecording: "", queue: [], playing: -1, playGen: 0, view: 0 };
// Each kind of request counts its loads; a response is used only if no newer load of the same
// kind started after it, so a slow old response can't overwrite a newer one.
const gen = { edit: 0, revisions: 0, pool: 0, suggest: 0, prefs: 0, ride: 0 };

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }
  return res.json();
}

function toast(text) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = text;
  document.body.append(el);
  setTimeout(() => el.remove(), 3500);
}

async function guarded(fn) {
  try {
    await fn();
  } catch (e) {
    toast(e.message);
    // Someone (another tab, an earlier click) changed the edit: show its current state.
    if (e.status === 409 && state.length === RIDE) {
      await loadRide().catch(() => {});
    } else if (e.status === 409) {
      await loadEdit().catch(() => {});
      await loadPreferences(state.length).catch(() => {});
    }
  }
}

// Responses for a tab that is no longer shown are dropped: every load remembers the view it
// started in, and selecting a length starts a new view.
function currentView() { return state.view; }
function stillCurrent(view, length) { return view === state.view && length === state.length; }
// The board an action was made on: the server refuses it (409) if that edit has changed since.
function board() { return { board: state.length, expected_edit_id: state.edit ? state.edit.edit_id : null }; }
function expectedQuery() { return state.edit ? `?expected_edit_id=${state.edit.edit_id}` : ""; }

function fmt(seconds) {
  const s = Math.round(seconds);
  return s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s} s`;
}

function lengthLabel(n) {
  return n >= 60 && n % 60 === 0 ? `${n / 60} min` : `${n} s`;
}

// --- loading -----------------------------------------------------------------------------

async function loadTrip() {
  state.trip = await api("GET", "/api/trip");
  $("#trip-name").textContent = state.trip.name || "Trip";
  document.title = `${state.trip.name} · Reelsift`;
  $("#trip-sub").textContent =
    `${state.trip.recordings} recordings, ${state.trip.cameras.length} camera(s), ${state.trip.candidates} candidates`;
  api("GET", "/api/music").then((info) => {
    state.musicInfo = info;
    $("#music-genres").innerHTML = info.genres.map((g) => `<option value="${escapeHtml(g)}">`).join("");
    renderMusic();
  }).catch(() => {});
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (const n of state.trip.lengths) {
    const b = document.createElement("button");
    b.textContent = lengthLabel(n);
    b.setAttribute("role", "tab");
    b.onclick = () => guarded(() => selectLength(n));
    b.dataset.length = n;
    tabs.append(b);
  }
  const filter = $("#filter-recording");
  filter.innerHTML = `<option value="">All recordings</option>` +
    state.trip.recording_list.map((r) => `<option value="${r.id}">${escapeHtml(r.time)} · ${escapeHtml(r.camera)}</option>`).join("");
  filter.onchange = () => { state.poolRecording = filter.value; state.poolOffset = 0; guarded(loadPool); };
  const rideTab = document.createElement("button");
  rideTab.textContent = "Ride";
  rideTab.setAttribute("role", "tab");
  rideTab.dataset.length = RIDE;
  rideTab.title = "The whole trip, stops cut out and sped up";
  rideTab.onclick = () => guarded(() => selectLength(RIDE));
  tabs.append(rideTab);
  // Without analysed candidates there are no highlight edits to show, but there may be a ride.
  if (!state.trip.candidates) return selectLength(RIDE);
  await selectLength(state.trip.lengths[Math.min(1, state.trip.lengths.length - 1)]);
}

const RIDE = -1;
const HIGHLIGHT_ONLY = ["#board", ".prefs", ".browser", "#undo", "#reselect", "#refill", "#approve", "#music-fit"];

async function selectLength(n) {
  state.length = n;
  state.view += 1;
  for (const b of $("#tabs").children) b.setAttribute("aria-selected", String(Number(b.dataset.length) === n));
  hideSuggestions();
  for (const sel of HIGHLIGHT_ONLY) $(sel).hidden = n === RIDE;
  $("#ride").hidden = n !== RIDE;
  if (n === RIDE) {
    state.edit = null;
    await loadRide();
    return;
  }
  showPrefsLoading();
  await loadEdit();
  state.poolOffset = 0;
  await loadPool();
  await loadPreferences(n);
}

async function loadEdit(edit) {
  const view = currentView();
  const length = state.length;
  const mine = ++gen.edit;
  const fresh = edit || await api("GET", `/api/edits/${length}`);
  if (mine !== gen.edit || !stillCurrent(view, length) || fresh.length !== length) return;
  // Every change saves a new edit with a higher id: an older one is never shown over a newer.
  if (state.edit && state.edit.length === length && fresh.edit_id < state.edit.edit_id) return;
  state.edit = fresh;
  if (length !== RIDE) renderBoard();
  renderStatus();
  renderMusic();
  await loadRevisions();
}

function renderStatus() {
  const e = state.edit;
  if (e.length === RIDE) {
    const target = state.ride && state.ride.settings.speed === "fit" ? state.ride.settings.target : null;
    $("#meter-fill").style.width = target ? `${Math.min(100, (e.duration / target) * 100)}%` : "100%";
    $("#meter-fill").classList.remove("over");
    $("#meter-text").textContent = `${fmt(e.duration)}${target ? ` of ${fmt(target)}` : ""} · ${e.items.length} stretches · revision ${e.revision}`;
    $("#approved").hidden = true;
    return;
  }
  const ratio = e.duration / e.length;
  const fill = $("#meter-fill");
  fill.style.width = `${Math.min(100, ratio * 100)}%`;
  fill.classList.toggle("over", ratio > 1.05);
  $("#meter-text").textContent = `${fmt(e.duration)} of ${fmt(e.length)} · ${e.items.length} clips · revision ${e.revision}`;
  $("#approved").hidden = !e.approved;
}

async function loadRevisions() {
  const view = currentView();
  const length = state.length;
  const mine = ++gen.revisions;
  const list = await api("GET", `/api/edits/${length}/revisions`);
  if (mine !== gen.revisions || !stillCurrent(view, length)) return;
  const sel = $("#revisions");
  sel.innerHTML = list.map((r) =>
    `<option value="${r.id}" ${r.revision === state.edit.revision ? "selected" : ""}>` +
    `Revision ${r.revision} · ${escapeHtml(r.note || "edit")} · ${r.clips} clips${r.approved ? " · approved" : ""}</option>`
  ).join("");
}

// --- storyboard ----------------------------------------------------------------------------

function reasonChips(reasons) {
  return reasons.map((r) =>
    `<span class="chip ${r.value >= 0 ? "pos" : "neg"}" title="${r.value}">${r.value >= 0 ? "+" : "−"} ${escapeHtml(r.name)}</span>`
  ).join("");
}

function renderBoard() {
  const board = $("#board");
  board.innerHTML = "";
  if (!state.edit.items.length) {
    board.innerHTML = `<div class="empty">This edit is empty. Use “Select again”.</div>`;
    return;
  }
  const tpl = $("#card");
  state.edit.items.forEach((item, index) => {
    const card = tpl.content.firstElementChild.cloneNode(true);
    card.dataset.index = index;
    card.classList.toggle("locked", item.locked);
    card.querySelector("img").src = item.thumb;
    card.querySelector(".dur").textContent = `${item.duration.toFixed(1)} s`;
    card.querySelector(".when").textContent = item.time;
    card.querySelector(".cam").textContent = `${item.camera} · ${item.photo ? "photo" : item.source}`;
    for (const b of card.querySelectorAll("[data-trim^='start']")) b.disabled = item.photo;
    card.querySelector(".why").innerHTML = reasonChips(item.reasons);
    card.querySelector(".thumb").onclick = () => playItems([item], [index]);
    const keep = card.querySelector(".keep");
    keep.classList.toggle("on", item.locked);
    keep.textContent = item.locked ? "Kept" : "Keep";
    keep.disabled = item.locked;
    keep.onclick = () => guarded(() => decide("keep", item));
    card.querySelector(".drop").onclick = () => guarded(() => drop(item));
    card.querySelector(".drop-all").onclick = () => guarded(() => drop(item, true));
    setupCrop(card, item);
    for (const b of card.querySelectorAll("[data-trim]")) {
      b.onclick = () => guarded(() => trim(item, ...b.dataset.trim.split(",")));
    }
    board.append(card);
  });
}

function setupCrop(card, item) {
  const show = $("#show-crop").checked;
  const frame = card.querySelector(".crop-window");
  const slider = card.querySelector("input.crop");
  // Share of the 16:9 thumbnail box covered by the picture, and by a 9:16 window over it.
  const box = 16 / 9;
  const pictureWidth = Math.min(1, item.aspect / box);
  const windowWidth = Math.min(pictureWidth, (9 / 16) / box);
  const landscape = item.aspect > 9 / 16 + 0.01;
  frame.hidden = slider.hidden = !show || !landscape;
  if (!show || !landscape) return;
  const place = (x) => {
    const left = (1 - pictureWidth) / 2 + (pictureWidth - windowWidth) * x;
    frame.style.left = `${left * 100}%`;
    frame.style.width = `${windowWidth * 100}%`;
  };
  slider.value = item.crop_x;
  place(item.crop_x);
  slider.oninput = () => place(Number(slider.value));
  slider.onchange = () => guarded(async () => {
    await loadEdit(await api("POST", `/api/edits/${state.length}/crop`, {
      recording_id: item.recording_id, start: item.start, end: item.end, crop_x: Number(slider.value),
      expected_edit_id: state.edit.edit_id,
    }));
  });
}

$("#show-crop").onchange = () => renderBoard();

// on: the board the action was offered on (see board()); defaults to the one shown now.
// Every decision applies to that edit only, except a drop with extra {length: null}.
async function decide(kind, item, extra = {}, on = board()) {
  await api("POST", "/api/decisions", {
    kind, recording_id: item.recording_id, start: item.start, end: item.end,
    length: on.board, ...on, ...extra,
  });
  await loadEdit();
  await loadPool();
}

async function drop(item, everywhere = false) {
  await decide("drop", item, everywhere ? { length: null } : {});
  await showSuggestions(item);
}

async function trim(item, side, delta) {
  const d = Number(delta);
  const start = side === "start" ? item.start + d : item.start;
  const end = side === "end" ? item.end + d : item.end;
  if (end - start < 0.5) return toast("The clip can't get shorter than half a second.");
  await api("POST", "/api/decisions", {
    kind: "keep", recording_id: item.recording_id, start, end, length: state.length,
    replaces: [item.start, item.end], ...board(),
  });
  await loadEdit();
}

// --- suggestions and the pool ------------------------------------------------------------------

function candidateRow(c, actionLabel, onAction) {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `
    <button class="thumb" title="Preview"><img alt="" loading="lazy" src="${c.thumb}"><span class="dur">${c.duration.toFixed(1)} s</span></button>
    <div>
      <div class="when">${escapeHtml(c.time)}</div>
      <div class="muted cam">${escapeHtml(c.camera)} · ${escapeHtml(c.source)} · score ${c.score.toFixed(2)}</div>
      <div class="why">${reasonChips(c.reasons)}</div>
      <div class="row-tools"><button class="add">${actionLabel}</button></div>
    </div>`;
  row.querySelector(".thumb").onclick = () => playItems([c], []);
  row.querySelector(".add").onclick = () => guarded(() => onAction(c));
  return row;
}

async function showSuggestions(item) {
  const view = currentView();
  const length = state.length;
  const mine = ++gen.suggest;
  const on = board(); // suggestions fit this edit; if another is shown by then, they are dropped
  const list = await api("GET", `/api/suggestions/${length}?recording_id=${item.recording_id}&t=${item.start}`);
  if (mine !== gen.suggest || !stillCurrent(view, length)) return;
  if (!state.edit || state.edit.edit_id !== on.expected_edit_id) return;
  const box = $("#suggest-list");
  box.innerHTML = "";
  for (const c of list) {
    box.append(candidateRow(c, "Use", async (cand) => {
      await decide("force", cand, {}, on);
      hideSuggestions();
    }));
  }
  $("#suggest").hidden = list.length === 0;
}

function hideSuggestions() {
  gen.suggest += 1; // suggestions still loading belong to what was hidden
  $("#suggest").hidden = true;
}

async function loadPool(append = false) {
  const view = currentView();
  const length = state.length;
  const recording = state.poolRecording;
  // Loading more continues the current list; any other load starts a new one.
  const mine = append ? gen.pool : ++gen.pool;
  const more = $("#pool-more");
  more.disabled = true;
  const q = recording ? `&recording_id=${recording}` : "";
  let list;
  try {
    list = await api("GET", `/api/candidates/${length}?offset=${state.poolOffset}&limit=20${q}`);
  } finally {
    if (mine === gen.pool) more.disabled = false;
  }
  if (mine !== gen.pool || !stillCurrent(view, length) || recording !== state.poolRecording) return;
  const on = board();
  const pool = $("#pool");
  if (!append) pool.innerHTML = "";
  const inEdit = new Set(state.edit.items.map((i) => i.candidate_id));
  for (const c of list) {
    if (inEdit.has(c.candidate_id)) continue;
    pool.append(candidateRow(c, "Add to edit", (cand) => decide("force", cand, {}, on)));
  }
  more.hidden = list.length < 20;
}

// --- playback ------------------------------------------------------------------------------

const video = $("#video");

function playItems(items, boardIndexes) {
  // One queue entry per chapter span: a clip crossing a chapter boundary plays both parts.
  state.queue = items.flatMap((it, i) => {
    const spans = it.spans && it.spans.length ? it.spans : (it.preview ? [it.preview] : []);
    return spans.map((span) => ({ item: it, span, board: boardIndexes[i] }));
  });
  state.playing = -1;
  playNext();
}

function markPlaying(boardIndex) {
  for (const card of document.querySelectorAll(".card.playing")) card.classList.remove("playing");
  if (boardIndex !== undefined) {
    const card = document.querySelector(`.card[data-index="${boardIndex}"]`);
    if (card) { card.classList.add("playing"); card.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  }
}

function playNext() {
  state.playing += 1;
  if (state.playing >= state.queue.length) {
    video.pause();
    markPlaying(undefined);
    $("#now-playing").textContent = "";
    return;
  }
  const { item, span, board } = state.queue[state.playing];
  const src = `/media/proxy/${item.recording_id}/${span.chapter}`;
  clearTimeout(state.stillTimer);
  const still = $("#still");
  if (item.photo) {
    video.pause();
    still.src = src;
    still.hidden = false;
    markPlaying(board);
    $("#now-playing").textContent = `${state.queue.length > 1 ? `Clip ${state.playing + 1} of ${state.queue.length} · ` : ""}${item.time} · photo`;
    state.stillTimer = setTimeout(playNext, item.duration * 1000);
    return;
  }
  still.hidden = true;
  const mine = ++state.playGen;
  const start = () => {
    if (mine !== state.playGen) return; // something else was played meanwhile
    video.currentTime = span.start;
    // Sped-up stretches play fast; browsers go up to about 16x (a render shows the real thing).
    try { video.playbackRate = Math.min(16, item.speed || 1); } catch (_) { video.playbackRate = 1; }
    video.play().catch(() => {});
  };
  if (!video.src.endsWith(src)) {
    video.src = src;
    video.addEventListener("loadedmetadata", start, { once: true });
  } else if (video.readyState < 1) {
    video.addEventListener("loadedmetadata", start, { once: true }); // still loading
  } else {
    start();
  }
  markPlaying(board);
  const total = state.queue.length;
  $("#now-playing").textContent = total > 1 ? `Clip ${state.playing + 1} of ${total} · ${item.time}` : item.time;
}

function stopQueue() {
  state.queue = [];
  state.playing = -1;
  state.playGen += 1;
  clearTimeout(state.stillTimer);
  markPlaying(undefined);
}

video.addEventListener("timeupdate", () => {
  const current = state.queue[state.playing];
  if (current && !current.item.photo && video.currentTime >= current.span.end) playNext();
});
// A proxy that ends early or fails to load must not stall the queue.
video.addEventListener("ended", () => { if (state.queue[state.playing]) playNext(); });
video.addEventListener("error", () => { if (state.queue[state.playing]) playNext(); });

// --- buttons ---------------------------------------------------------------------------------

$("#play-all").onclick = () => playItems(state.edit.items, state.edit.items.map((_, i) => i));
$("#undo").onclick = () => guarded(async () => {
  const r = await api("POST", "/api/undo");
  if (!r.changed.length) toast("Nothing to undo.");
  hideSuggestions();
  await loadEdit();
  await loadPool();
});
$("#reselect").onclick = () => guarded(async () => { await loadEdit(await api("POST", `/api/edits/${state.length}/select${expectedQuery()}`)); await loadPool(); });
$("#refill").onclick = () => guarded(async () => { await loadEdit(await api("POST", `/api/edits/${state.length}/refill${expectedQuery()}`)); await loadPool(); });
$("#approve").onclick = () => guarded(async () => { await loadEdit(await api("POST", `/api/edits/${state.length}/approve${expectedQuery()}`)); toast("Approved."); });
$("#revisions").onchange = (e) => guarded(async () => {
  const restored = await api("POST", `/api/edits/${state.length}/restore/${e.target.value}`);
  // A ride's revision carries its settings and stops as well: show the whole ride again.
  if (restored.length === RIDE) await loadRide();
  else await loadEdit(restored);
});
$("#suggest-close").onclick = hideSuggestions;
$("#pool-more").onclick = () => guarded(async () => { state.poolOffset += 20; await loadPool(true); });

async function render(vertical) {
  await api("POST", "/api/render", { length: state.length, vertical, ...outputOf() });
  pollRender();
}

// Render and export use the edit shown; if it changed on the server, they are refused (409).
function outputOf() { return { expected_edit_id: state.edit ? state.edit.edit_id : null }; }

async function pollRender() {
  const s = await api("GET", "/api/render");
  const el = $("#render-state");
  for (const b of [$("#render-h"), $("#render-v")]) b.disabled = s.running;
  if (s.running) {
    el.textContent = s.total ? `Rendering ${s.done} of ${s.total} clips…` : "Rendering…";
    setTimeout(pollRender, 1000);
  } else if (s.error) {
    el.textContent = `Render failed: ${s.error}`;
  } else if (s.output) {
    const url = `/media/output/${encodeURIComponent(s.output)}?preview=${Boolean(s.preview)}`;
    el.innerHTML = `Done: <a href="${url}" target="_blank" rel="noopener">${escapeHtml(s.output)}</a>
      <a class="button primary" id="render-download" href="${url}&download=true" download="${escapeHtml(s.output)}">Download</a>`;
  }
}
$("#export").onclick = () => guarded(async () => {
  const r = await api("POST", "/api/export", { length: state.length, vertical: false, ...outputOf() });
  $("#render-state").innerHTML = "Timeline: " + r.files.map((f) =>
    `<a href="/media/output/${encodeURIComponent(f)}">${escapeHtml(f)}</a>`).join(" · ");
});
$("#render-h").onclick = () => guarded(() => render(false));
$("#render-v").onclick = () => guarded(() => render(true));

// --- preferences ---------------------------------------------------------------------------

const prefs = { data: null, rows: [] };

// Each edit length has its own description, sliders, style and clip lengths. Until a tab's
// own settings have arrived its controls are off, so nothing shown for another edit can be
// saved to this one.
const PREF_CONTROLS = ["#pref-text", "#pref-apply", "#pref-draft", "#pref-style", "#clip-min",
  "#clip-max", "#clip-apply", "#pref-sliders-apply", "#pref-add-feature", "#pref-add-search",
  "#pref-add-search-btn"];

function showPrefsLoading() {
  prefs.data = null;
  for (const sel of PREF_CONTROLS) $(sel).disabled = true;
  $("#pref-text").value = "";
  $("#pref-engine").textContent = "";
}

// The edit the settings on screen belong to, or null while they are loading.
function prefsLength() {
  return prefs.data && prefs.data.length === state.length ? prefs.data.length : null;
}

async function loadPreferences(length, data) {
  const mine = ++gen.prefs;
  const fresh = data || await api("GET", `/api/preferences/${length}`);
  if (mine !== gen.prefs || length !== state.length || fresh.length !== length) return;
  prefs.data = fresh;
  for (const sel of PREF_CONTROLS) $(sel).disabled = false;
  const d = prefs.data;
  $("#pref-title").textContent = `What you want in the ${lengthLabel(length)} edit`;
  $("#pref-style").innerHTML = d.styles.map((s) =>
    `<option value="${s.name}" ${s.name === d.style ? "selected" : ""}>${escapeHtml(s.label)}</option>`).join("");
  $("#pref-text").value = d.text || "";
  $("#pref-engine").textContent = d.drafted ? "suggested from your footage"
    : d.engine && d.engine !== "none" ? `read by: ${d.engine}` : "";
  $("#clip-min").placeholder = d.auto_clip[0];
  $("#clip-max").placeholder = d.auto_clip[1];
  $("#clip-min").value = d.clip ? d.clip[0] : "";
  $("#clip-max").value = d.clip ? d.clip[1] : "";
  $("#pref-lines").innerHTML = (d.interpretation || []).map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  prefs.rows = [
    ...Object.entries(d.adjustments).map(([name, value]) => ({ kind: "feature", name, value })),
    ...d.searches.map((s) => ({ kind: "search", name: s.text, value: s.value })),
  ];
  // Something left out always has a row, so it can be let back in.
  for (const key of d.exclusions) {
    if (prefs.rows.some((r) => exclusionKey(r) === key)) continue;
    prefs.rows.push(key.startsWith("search:")
      ? { kind: "search", name: key.slice("search:".length), value: 0 }
      : { kind: "feature", name: key, value: 0 });
  }
  const labels = Object.fromEntries(d.vocabulary.map((v) => [v.name, v.label]));
  prefs.labels = labels;
  $("#pref-add-feature").innerHTML = `<option value="">Add a setting…</option>` +
    d.vocabulary.map((v) => `<option value="${v.name}">${escapeHtml(v.label)}</option>`).join("");
  $("#pref-add-search").disabled = $("#pref-add-search-btn").disabled = !d.searches_available;
  $("#pref-draft").title = d.drafting_available
    ? "Write this edit's description again from your footage"
    : "Without the vision model this uses the style's standard description";
  if (!d.searches_available) $("#pref-add-search").placeholder = "Searching needs the analysis models";
  renderSliders();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function exclusionKey(row) {
  return row.kind === "search" ? `search:${row.name.toLowerCase()}` : row.name;
}

function renderSliders() {
  const box = $("#pref-sliders");
  box.innerHTML = "";
  const excluded = new Set(prefs.data.exclusions);
  prefs.rows.forEach((row, i) => {
    const el = document.createElement("div");
    el.className = "slider";
    const label = row.kind === "search" ? `looks like “${row.name}”` : (prefs.labels[row.name] || row.name);
    el.innerHTML = `
      <span>${escapeHtml(label)}</span>
      <input type="range" min="-3" max="3" step="1" value="${row.value}" aria-label="${escapeHtml(label)}">
      <output>${row.value > 0 ? "+" : ""}${row.value}</output>
      <label><input type="checkbox" ${excluded.has(exclusionKey(row)) ? "checked" : ""}> leave out</label>
      <button class="link" title="Remove">✕</button>`;
    const range = el.querySelector("input[type=range]");
    range.oninput = () => { row.value = Number(range.value); el.querySelector("output").textContent = `${row.value > 0 ? "+" : ""}${row.value}`; };
    el.querySelector("input[type=checkbox]").onchange = (e) => {
      const key = exclusionKey(row);
      prefs.data.exclusions = prefs.data.exclusions.filter((x) => x !== key);
      if (e.target.checked) prefs.data.exclusions.push(key);
    };
    el.querySelector("button").onclick = () => {
      // Removing a setting removes its "leave out" too.
      prefs.data.exclusions = prefs.data.exclusions.filter((x) => x !== exclusionKey(row));
      prefs.rows.splice(i, 1);
      renderSliders();
    };
    box.append(el);
  });
}

async function afterPreferences(length, data) {
  await loadPreferences(length, data);
  if (length !== state.length) return;
  await loadEdit();
  await loadPool();
  toast(`The ${lengthLabel(length)} edit was re-selected.`);
}

// The current settings of the edit shown, as the server takes them, with some changed.
function prefsBody(changes = {}) {
  return {
    text: $("#pref-text").value,
    adjustments: Object.fromEntries(prefs.rows.filter((r) => r.kind === "feature").map((r) => [r.name, r.value])),
    searches: prefs.rows.filter((r) => r.kind === "search").map((r) => ({ text: r.name, value: r.value })),
    exclusions: prefs.data.exclusions,
    style: prefs.data.style,
    clip: prefs.data.clip,
    expected_version: prefs.data.version,
    ...changes,
  };
}

async function busy(button, label, fn) {
  const text = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try { await fn(); } finally { button.disabled = false; button.textContent = text; }
}

$("#pref-apply").onclick = () => guarded(() => busy($("#pref-apply"), "Reading…", async () => {
  const length = prefsLength();
  if (length === null) return;
  await afterPreferences(length, await api("POST", `/api/preferences/${length}/text`, { text: $("#pref-text").value }));
}));

$("#pref-draft").onclick = () => guarded(() => busy($("#pref-draft"), "Writing…", async () => {
  const length = prefsLength();
  if (length === null) return;
  await afterPreferences(length, await api("POST", `/api/preferences/${length}/draft`));
}));

$("#pref-sliders-apply").onclick = () => guarded(async () => {
  const length = prefsLength();
  if (length === null) return;
  await afterPreferences(length, await api("PUT", `/api/preferences/${length}`, prefsBody()));
});

$("#pref-style").onchange = (e) => guarded(async () => {
  const length = prefsLength();
  if (length === null) return;
  await afterPreferences(length, await api("PUT", `/api/preferences/${length}`, prefsBody({ style: e.target.value })));
});

$("#clip-apply").onclick = () => guarded(async () => {
  const length = prefsLength();
  if (length === null) return;
  const lo = $("#clip-min").value, hi = $("#clip-max").value;
  if ((lo === "") !== (hi === "")) return toast("Give both clip lengths, or leave both empty for automatic.");
  const clip = lo === "" ? null : [Number(lo), Number(hi)];
  await afterPreferences(length, await api("PUT", `/api/preferences/${length}`, prefsBody({ clip })));
});

$("#pref-add-feature").onchange = (e) => {
  const name = e.target.value;
  if (name && !prefs.rows.some((r) => r.kind === "feature" && r.name === name)) {
    prefs.rows.push({ kind: "feature", name, value: 1 });
    renderSliders();
  }
  e.target.value = "";
};

$("#pref-add-search-btn").onclick = () => {
  const text = $("#pref-add-search").value.trim();
  if (text) {
    prefs.rows.push({ kind: "search", name: text, value: 2 });
    $("#pref-add-search").value = "";
    $("#pref-details").open = true;
    renderSliders();
  }
};

// --- music ---------------------------------------------------------------------------------

let audioElement = null;
function player() { return audioElement || (audioElement = document.createElement("audio")); }
const MUSIC_CONTROLS = ["#music-source", "#music-generate", "#music-more", "#music-fit", "#music-preview",
  "#mix-music", "#mix-background", "#mix-duck", "#mix-sound", "#mix-reset"];

function dB(v) { return Number(v) <= -60 ? "off" : `${v > 0 ? "+" : ""}${Number(v).toFixed(1)} dB`; }

function renderMusic() {
  const m = state.edit && state.edit.music;
  if (!m) return;
  const info = state.musicInfo || { installed: false };
  $("#music-source").value = m.source;
  $("#music-files").textContent = m.source === "files"
    ? (m.files.length ? `Playing: ${m.files.join(", ")}` : "No music files set (reelsift music set).")
    : m.source === "generated" && m.ends_early ? "This track ends a little before the edit does." : "";
  $("#music-gen").hidden = m.source === "files";
  if (document.activeElement !== $("#music-genre")) $("#music-genre").value = m.genre || "";
  if (document.activeElement !== $("#music-mood")) $("#music-mood").value = m.mood || "";
  $("#music-generate").disabled = $("#music-more").disabled = !info.installed;
  if (!info.installed) $("#music-job").textContent = "Install the music model first: reelsift music install";
  const list = $("#music-tracks");
  list.innerHTML = "";
  for (const t of m.tracks.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "track" + (m.source === "generated" && m.asset_id === t.id ? " on" : "");
    const fit = t.hint === null || t.hint === undefined ? "" : ` · fit ${t.hint >= 0 ? "+" : ""}${t.hint.toFixed(2)}`;
    row.innerHTML = `<button class="play" title="Listen">▶</button>
      <span class="what" title="${escapeHtml(t.caption)}">${t.bpm} BPM · ${fmt(t.duration)}${fit}</span>
      <button class="use">${m.source === "generated" && m.asset_id === t.id ? "Playing" : "Use"}</button>`;
    row.querySelector(".play").onclick = () => {
      const audio = player();
      if (audio.dataset.id === String(t.id) && !audio.paused) { audio.pause(); return; }
      $("#video").pause();
      audio.src = t.url; audio.dataset.id = String(t.id); audio.play().catch((e) => toast(e.message));
    };
    row.querySelector(".use").disabled = t.missing || (m.source === "generated" && m.asset_id === t.id);
    row.querySelector(".use").onclick = () => guarded(async () => {
      await loadEdit(await api("POST", `/api/edits/${state.length}/music/use`, { asset_id: t.id, ...outputOf() }));
    });
    list.append(row);
  }
  for (const k of ["music", "background", "duck", "sound"]) {
    const input = $(`#mix-${k}`);
    if (document.activeElement !== input) input.value = m.mix[k];
    $(`#mix-${k}-v`).textContent = dB(input.value);
  }
  const silent = m.source === "none";
  for (const sel of ["#mix-music", "#mix-background", "#mix-duck", "#music-fit"]) $(sel).disabled = silent;
}

$("#music-source").onchange = (e) => guarded(async () => {
  await loadEdit(await api("POST", `/api/edits/${state.length}/music/source`, { source: e.target.value, ...outputOf() }));
});

for (const k of ["music", "background", "duck", "sound"]) {
  const input = $(`#mix-${k}`);
  input.oninput = () => { $(`#mix-${k}-v`).textContent = dB(input.value); };
  input.onchange = () => guarded(async () => {
    await loadEdit(await api("POST", `/api/edits/${state.length}/music/mix`, { [k]: Number(input.value), ...outputOf() }));
  });
}
$("#mix-reset").onclick = () => guarded(async () => {
  await loadEdit(await api("POST", `/api/edits/${state.length}/music/mix`, { reset: true, ...outputOf() }));
});

async function generateMusic(candidates) {
  await api("POST", `/api/edits/${state.length}/music/generate`, {
    genre: $("#music-genre").value, mood: $("#music-mood").value, candidates, ...outputOf(),
  });
  pollMusic();
}
$("#music-generate").onclick = () => guarded(() => generateMusic(2));
$("#music-more").onclick = () => guarded(() => generateMusic(1));
$("#music-cancel").onclick = () => guarded(async () => { await api("POST", "/api/music/job/cancel"); });

async function pollMusic() {
  const s = await api("GET", "/api/music/job");
  const el = $("#music-job");
  $("#music-cancel").hidden = !s.running;
  for (const b of [$("#music-generate"), $("#music-more")]) b.disabled = s.running || !(state.musicInfo || {}).installed;
  if (s.running) {
    el.textContent = (s.messages && s.messages.length ? s.messages[s.messages.length - 1] : "Starting…");
    setTimeout(pollMusic, 1000);
    return;
  }
  if (s.error) el.textContent = s.error === "cancelled" ? "Cancelled." : `Failed: ${s.error}`;
  else if (s.note) el.textContent = s.note;
  else if (s.made && s.made.length) el.textContent = `Made ${s.made.length} track(s).`;
  if (s.length === state.length) await loadEdit();
}

$("#music-fit").onclick = () => guarded(() => busy($("#music-fit"), "Fitting…", async () => {
  await loadEdit(await api("POST", `/api/edits/${state.length}/music/fit${expectedQuery()}`));
}));

$("#music-preview").onclick = () => guarded(async () => {
  await api("POST", "/api/render", { length: state.length, vertical: false, preview: true, ...outputOf() });
  pollPreview();
});

async function pollPreview() {
  const s = await api("GET", "/api/render");
  const el = $("#music-job");
  $("#music-preview").disabled = s.running;
  if (s.running) {
    el.textContent = s.total ? `Preview: ${s.done} of ${s.total} clips…` : "Preview…";
    setTimeout(pollPreview, 1000);
  } else if (s.error) {
    el.textContent = `Preview failed: ${s.error}`;
  } else if (s.output && s.preview) {
    el.textContent = "";
    player().pause();
    // The rendered preview plays on its own: not as part of clips queued earlier, and at its
    // own speed (a sped-up stretch played before is already sped up in it).
    stopQueue();
    $("#still").hidden = true;
    video.hidden = false;
    video.src = `/media/output/${encodeURIComponent(s.output)}?preview=true`;
    video.playbackRate = 1;
    $("#now-playing").textContent = "Preview with music";
    video.play().catch(() => {});
  }
}

// --- the ride ---------------------------------------------------------------------------------

const SPEEDS = [1, 2, 4, 8, 16, 32, 64];

function parseLength(text) {
  const t = text.trim();
  if (!t) return null;
  const parts = t.split(":").map(Number);
  if (parts.some((p) => !Number.isFinite(p) || p < 0)) return NaN;
  return parts.reduce((total, p) => total * 60 + p, 0);
}

// A ride response is shown only in the view it was asked from, and never over a newer revision
// of the ride (every change saves one with a higher id). Returns whether it was shown.
function showRide(data, view) {
  if (!stillCurrent(view, RIDE)) return false;
  const shown = state.ride && state.ride.exists ? state.ride.edit_id : null;
  if (shown !== null && (!data.exists || data.edit_id < shown)) return false;
  state.ride = data;
  if (data.exists) state.edit = data;
  return true;
}

async function loadRide() {
  const view = currentView();
  const mine = ++gen.ride;
  const data = await api("GET", "/api/ride");
  if (mine !== gen.ride || !showRide(data, view)) return;
  if (data.exists) {
    renderStatus();
    renderMusic();
    await loadRevisions();
  } else {
    $("#meter-text").textContent = "No ride yet: choose a speed or a length and build it.";
    $("#meter-fill").style.width = "0";
  }
  renderRide();
  if (data.job && data.job.running) watchRide(view);
}

// One status check at a time while a ride is being built.
function watchRide(view) {
  if (state.rideWatch === view) return;
  state.rideWatch = view;
  setTimeout(() => guarded(() => pollRide(view)), 1000);
}

function renderRide() {
  const r = state.ride;
  const s = r.settings;
  const report = r.report;
  const speed = $("#ride-speed");
  const moving = report ? report.moving : null;
  const kept = report ? report.kept_stops : 0;
  speed.innerHTML = `<option value="fit">Fit a length</option>` + SPEEDS.map((x) =>
    `<option value="${x}">${x}x${moving !== null ? ` · ${fmt(moving / x + kept)}` : ""}</option>`).join("");
  const saved = s.speed === "fit" ? "fit" : String(Number(s.speed));
  if (saved !== "fit" && Number.isFinite(Number(saved)) && !SPEEDS.map(String).includes(saved)) {
    // set elsewhere (reelsift ride --speed 3.1): offered as it is, never swapped for a preset
    speed.innerHTML += `<option value="${saved}">${saved}x${moving !== null ? ` · ${fmt(moving / Number(saved) + kept)}` : ""}</option>`;
  }
  speed.value = saved;
  if (speed.value === "") speed.value = "8";
  $("#ride-target-box").hidden = speed.value !== "fit";
  if (document.activeElement !== $("#ride-target")) $("#ride-target").value = s.target ? fmt(s.target).replace(" s", "") : "";
  $("#ride-camera").innerHTML = `<option value="">Most footage</option>` + r.cameras.map((c) =>
    `<option value="${escapeHtml(c.label)}">${escapeHtml(c.label)} · ${fmt(c.seconds)}</option>`).join("");
  $("#ride-camera").value = s.camera || "";
  $("#ride-keep-unknown").checked = s.keep_unknown !== false;
  $("#ride-build").disabled = Boolean(r.job && r.job.running);
  $("#ride-job").textContent = r.job && r.job.running ? (r.job.message || "Building…") : (r.job && r.job.error ? r.job.error : "");
  $("#ride-summary").textContent = report
    ? `${report.camera}: ${fmt(report.footage)} of footage, ${fmt(report.moving)} travelling, stops cut ${fmt(report.stopped)}` +
      `${report.unknown ? `, ${fmt(report.unknown)} unclear (kept)` : ""} · ${report.speed.toFixed(1)}x → ${fmt(report.duration)}` +
      (report.notes && report.notes.length ? ` · ${report.notes.join(" ")}` : "")
    : "";
  const list = $("#ride-stretches");
  list.innerHTML = "";
  for (const [i, it] of (r.items || []).entries()) {
    const li = document.createElement("li");
    li.className = "ride-row";
    li.dataset.index = i;
    li.innerHTML = `<img alt="" loading="lazy" src="${it.thumb}">
      <span><strong>${escapeHtml(it.time)}</strong><br><span class="speed muted">${fmt(it.source_duration)} of footage → ${it.duration < 10 ? `${it.duration.toFixed(1)} s` : fmt(it.duration)}${it.speed !== 1 ? ` (${Number(it.speed).toFixed(1)}x)` : " (normal speed)"}</span></span>
      <span><button class="play" title="Play">▶</button> <button class="drop" title="Leave this stretch out of the ride">Drop</button></span>`;
    li.querySelector(".play").onclick = () => playItems([it], [i]);
    li.querySelector(".drop").onclick = () => guarded(() => fixRide("drop", it.recording_id, it.start, it.end));
    list.append(li);
  }
  const stops = $("#ride-stops");
  stops.innerHTML = "";
  $("#ride-stop-count").textContent = (r.stops || []).length;
  for (const st of r.stops || []) {
    const li = document.createElement("li");
    li.className = "ride-row";
    li.innerHTML = `<img alt="" loading="lazy" src="${st.thumb}">
      <span><strong>${escapeHtml(st.time)}</strong><br><span class="muted">stopped ${fmt(st.end - st.start)}</span></span>
      <span><button class="play" title="Play">▶</button> <button class="keep" title="Put this stop back in the ride, at normal speed">Keep at normal speed</button></span>`;
    li.querySelector(".play").onclick = () => playItems([{ ...st, speed: 1, duration: st.end - st.start, photo: false }], []);
    li.querySelector(".keep").onclick = () => guarded(() => fixRide("keep", st.recording_id, st.start, st.end));
    stops.append(li);
  }
}

async function fixRide(kind, recording_id, start, end) {
  const view = currentView();
  const data = await api("POST", "/api/ride/fix", { kind, recording_id, start, end, ...outputOf() });
  ++gen.ride; // a load asked for before this answer is older than it
  if (!showRide(data, view)) return;
  renderStatus();
  renderMusic();
  renderRide();
  await loadRevisions();
}

async function pollRide(view) {
  if (state.rideWatch === view) state.rideWatch = null;
  if (!stillCurrent(view, RIDE)) return;
  const mine = ++gen.ride;
  const data = await api("GET", "/api/ride");
  if (mine !== gen.ride || !stillCurrent(view, RIDE)) return;
  if (data.job && data.job.running) {
    if (showRide(data, view)) renderRide();
    watchRide(view);
    return;
  }
  await loadRide();
}

$("#ride-speed").onchange = () => { $("#ride-target-box").hidden = $("#ride-speed").value !== "fit"; };
$("#ride-build").onclick = () => guarded(async () => {
  const speed = $("#ride-speed").value;
  const target = parseLength($("#ride-target").value);
  if (speed === "fit" && !(target > 0)) return toast("Give the length, such as 4:00 or 240.");
  const view = currentView();
  const data = await api("POST", "/api/ride/build", {
    speed, target: speed === "fit" ? target : (target > 0 ? target : null),
    camera: $("#ride-camera").value || null, keep_unknown: $("#ride-keep-unknown").checked,
    expected_edit_id: state.ride && state.ride.exists ? state.ride.edit_id : null,
  });
  ++gen.ride;
  if (!showRide(data, view)) return;
  renderRide();
  watchRide(view);
});

guarded(async () => { await loadTrip(); await pollRender(); await pollMusic(); });
