// Response-ordering tests for the review app's frontend (static/app.js), run with `node --test`.
// app.js runs in a sandbox with a minimal stand-in for the page and a fetch whose responses the
// test releases in any order, so slow and fast requests can be reordered on purpose.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "../../src/reelsift/web/static/app.js"), "utf8");

function element() {
  const found = new Map();
  const el = {
    children: [], dataset: {}, style: {}, hidden: false, disabled: false, value: "",
    textContent: "", onclick: null, onchange: null, oninput: null,
    classList: { toggle() {}, add() {}, remove() {} },
    content: { firstElementChild: { cloneNode: () => element() } },
    get innerHTML() { return ""; },
    set innerHTML(_v) { el.children = []; },
    append(...kids) { el.children.push(...kids); },
    setAttribute() {}, addEventListener() {}, scrollIntoView() {}, remove() {}, pause() {},
    play() { return Promise.resolve(); },
    querySelector(sel) {
      if (!found.has(sel)) found.set(sel, element());
      return found.get(sel);
    },
    querySelectorAll() { return []; },
  };
  return el;
}

// Loads app.js afresh. Every fetch waits until the test answers it with reply(n, body).
function load() {
  const page = new Map();
  const requests = [];
  const document = {
    title: "",
    body: element(),
    querySelector(sel) {
      if (!page.has(sel)) page.set(sel, element());
      return page.get(sel);
    },
    querySelectorAll() { return []; },
    createElement: () => element(),
  };
  function fetch(url, options = {}) {
    return new Promise((resolve) => {
      requests.push({
        url, method: options.method, body: options.body ? JSON.parse(options.body) : undefined,
        answer: (body, status = 200) => resolve({
          ok: status < 400, status, statusText: "", json: async () => body,
        }),
      });
    });
  }
  const sandbox = { document, fetch, console, setTimeout: () => 0, clearTimeout() {} };
  vm.createContext(sandbox);
  vm.runInContext(
    source + "\n;globalThis.app = { state, loadEdit, loadPool, showSuggestions, selectLength, render, decide, drop, loadPreferences, prefs, loadRide, fixRide, prefsBody, pollPreview, playItems };",
    sandbox,
  );
  const settle = () => new Promise((r) => setImmediate(r));
  const pending = (prefix) => requests.filter((r) => r.url.startsWith(prefix) && !r.done);
  const reply = async (request, body, status) => {
    request.done = true;
    request.answer(body, status);
    for (let i = 0; i < 5; i++) await settle();
  };
  return { app: sandbox.app, page: document, requests, pending, reply, settle };
}

const edit = (length, id) => ({
  length, edit_id: id, revision: id, items: [], duration: 0, approved: false,
});
const preferences = (length, extra = {}) => ({
  length, text: "", adjustments: {}, searches: [], exclusions: [], interpretation: [],
  engine: "none", style: "story", clip: null, drafted: false, own: false,
  styles: [{ name: "story", label: "Story" }], auto_clip: [2, 4], vocabulary: [],
  searches_available: false, drafting_available: false, ...extra,
});
const candidate = (id) => ({
  candidate_id: id, recording_id: 1, start: id, end: id + 2, thumb: "", duration: 2, time: "",
  camera: "", source: "window", score: 1, reasons: [],
});

test("an older edit response arriving last doesn't replace the newer one", async () => {
  const { app, pending, reply } = load();
  app.state.length = 8;
  const first = app.loadEdit();
  const second = app.loadEdit();
  const [slow, fast] = pending("/api/edits/8");
  await reply(fast, edit(8, 2));
  await reply(slow, edit(8, 1));
  for (const r of pending("/api/edits/8/revisions")) await reply(r, []);
  await Promise.all([first, second]);
  assert.equal(app.state.edit.edit_id, 2);
});

test("a response to an old load doesn't replace an edit a change just returned", async () => {
  const { app, pending, reply } = load();
  app.state.length = 8;
  const loading = app.loadEdit();
  const [get] = pending("/api/edits/8");
  const saved = app.loadEdit(edit(8, 3)); // the edit a keep returned
  await reply(get, edit(8, 2));
  for (const r of pending("/api/edits/8/revisions")) await reply(r, []);
  await Promise.all([loading, saved]);
  assert.equal(app.state.edit.edit_id, 3);
});

test("suggestions that arrive after switching length are not shown", async () => {
  const { app, page, pending, reply } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 1);
  page.querySelector("#suggest").hidden = true;
  const shown = app.showSuggestions({ recording_id: 1, start: 4 });
  const [request] = pending("/api/suggestions/8");
  const switching = app.selectLength(16);
  await reply(request, [candidate(7)]);
  assert.equal(page.querySelector("#suggest").hidden, true);
  assert.equal(page.querySelector("#suggest-list").children.length, 0);
  for (const r of pending("/api/edits/16")) await reply(r, r.url.endsWith("revisions") ? [] : edit(16, 5));
  for (const r of pending("/api/edits/16/revisions")) await reply(r, []);
  for (const r of pending("/api/candidates/16")) await reply(r, []);
  for (const r of pending("/api/preferences/16")) await reply(r, preferences(16));
  await Promise.all([shown, switching]);
});

test("suggestions asked for one edit are not shown over a newer edit of the same length", async () => {
  const { app, page, pending, reply, requests } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 1);
  page.querySelector("#suggest").hidden = true;
  const shown = app.showSuggestions({ recording_id: 1, start: 4 });
  const [request] = pending("/api/suggestions/8");
  const reloaded = app.loadEdit(edit(8, 2)); // "Select again" showed a new revision meanwhile
  for (const r of pending("/api/edits/8/revisions")) await reply(r, []);
  await reply(request, [candidate(7)]);
  await Promise.all([shown, reloaded]);
  assert.equal(page.querySelector("#suggest").hidden, true);
  assert.equal(page.querySelector("#suggest-list").children.length, 0);
  assert.equal(requests.filter((r) => r.url === "/api/decisions").length, 0);
});

test("the pool shows the list for the recording filter chosen last", async () => {
  const { app, page, pending, reply } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 1);
  const all = app.loadPool();
  app.state.poolRecording = "5";
  const filtered = app.loadPool();
  const [allRequest, filteredRequest] = pending("/api/candidates/8");
  assert.match(filteredRequest.url, /recording_id=5/);
  await reply(filteredRequest, [candidate(50)]);
  await reply(allRequest, [candidate(1), candidate(2)]);
  await Promise.all([all, filtered]);
  assert.equal(page.querySelector("#pool").children.length, 1);
});

test("adding from the pool acts on the edit the pool was shown for", async () => {
  const { app, page, pending, reply, requests } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 4);
  const loading = app.loadPool();
  await reply(pending("/api/candidates/8")[0], [candidate(1)]);
  await loading;
  app.state.length = 16; // switched, but the old rows are still on screen
  app.state.edit = edit(16, 9);
  page.querySelector("#pool").children[0].querySelector(".add").onclick();
  await new Promise((r) => setImmediate(r));
  const post = requests.find((r) => r.url === "/api/decisions");
  assert.equal(post.body.length, 8);
  assert.equal(post.body.board, 8);
  assert.equal(post.body.expected_edit_id, 4);
});

test("preferences that arrive after switching length are not shown", async () => {
  const { app, page, pending, reply } = load();
  app.state.length = 8;
  const loading = app.loadPreferences(8);
  app.state.length = 16;
  await reply(pending("/api/preferences/8")[0], preferences(8, { text: "the 8 s edit" }));
  await loading;
  assert.notEqual(page.querySelector("#pref-text").value, "the 8 s edit");
});

test("settings still shown from another tab can't be saved to the new one", async () => {
  const { app, page, pending, reply, requests } = load();
  app.state.length = 4;
  const shown = app.loadPreferences(4);
  await reply(pending("/api/preferences/4")[0], preferences(4, { text: "the 4 s edit", style: "preview" }));
  await shown;
  const switching = app.selectLength(8); // the 8 s edit's loads are still on their way
  assert.equal(page.querySelector("#pref-apply").disabled, true);
  page.querySelector("#pref-sliders-apply").onclick();
  page.querySelector("#pref-apply").onclick();
  await new Promise((r) => setImmediate(r));
  assert.equal(requests.filter((r) => r.url.startsWith("/api/preferences/8") && r.method !== "GET").length, 0);
  for (const r of pending("/api/edits/8")) await reply(r, r.url.endsWith("revisions") ? [] : edit(8, 2));
  for (const r of pending("/api/edits/8/revisions")) await reply(r, []);
  for (const r of pending("/api/candidates/8")) await reply(r, []);
  for (const r of pending("/api/preferences/8")) await reply(r, preferences(8, { text: "the 8 s edit" }));
  await switching;
  assert.equal(page.querySelector("#pref-apply").disabled, false);
  assert.equal(page.querySelector("#pref-text").value, "the 8 s edit");
});

test("a drop acts on the edit shown unless it is for every edit", async () => {
  const { app, requests } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 3);
  app.drop({ recording_id: 1, start: 0, end: 2 });
  app.drop({ recording_id: 1, start: 4, end: 6 }, true);
  await new Promise((r) => setImmediate(r));
  const [one, all] = requests.filter((r) => r.url === "/api/decisions").map((r) => r.body);
  assert.equal(one.length, 8);
  assert.equal(all.length, null);
  assert.equal(all.expected_edit_id, 3);
});

test("render and export say which edit they were asked for", async () => {
  const { app, requests } = load();
  app.state.length = 8;
  app.state.edit = edit(8, 7);
  app.render(false);
  await new Promise((r) => setImmediate(r));
  const post = requests.find((r) => r.url === "/api/render");
  assert.equal(post.body.expected_edit_id, 7);
});

const RIDE = -1;
const ride = (id, speed = "8") => ({
  ...edit(RIDE, id), exists: true, settings: { speed, target: null, camera: null, keep_unknown: true, overrides: [] },
  cameras: [], job: { running: false }, report: null, stops: [],
});

test("a ride load answered after a correction doesn't bring back the older ride", async () => {
  const { app, pending, reply } = load();
  app.state.length = RIDE;
  const loading = app.loadRide();
  const [get] = pending("/api/ride");
  const fixing = app.fixRide("drop", 1, 2, 3);
  const [fix] = pending("/api/ride/fix");
  await reply(fix, ride(5));
  await reply(get, ride(4));
  for (const r of pending("/api/edits/-1/revisions")) await reply(r, []);
  await Promise.all([loading, fixing]);
  assert.equal(app.state.ride.edit_id, 5);
  assert.equal(app.state.edit.edit_id, 5);
});

test("a correction answered after switching to a highlight tab leaves that tab alone", async () => {
  const { app, pending, reply } = load();
  app.state.length = RIDE;
  app.state.ride = ride(4);
  app.state.edit = app.state.ride;
  const fixing = app.fixRide("drop", 1, 2, 3);
  const switching = app.selectLength(30);
  for (const r of pending("/api/edits/30")) await reply(r, r.url.endsWith("revisions") ? [] : edit(30, 7));
  await reply(pending("/api/ride/fix")[0], ride(5));
  for (const r of pending("/api/edits/30/revisions")) await reply(r, []);
  for (const r of pending("/api/candidates/30")) await reply(r, []);
  for (const r of pending("/api/preferences/30")) await reply(r, preferences(30));
  await Promise.all([fixing, switching]);
  assert.equal(app.state.edit.length, 30);
  assert.equal(app.state.edit.edit_id, 7);
});

test("restoring a ride revision shows that revision's ride, settings and all", async () => {
  const { app, page, pending, reply } = load();
  app.state.length = RIDE;
  app.state.ride = ride(5, "8");
  app.state.edit = app.state.ride;
  const restoring = page.querySelector("#revisions").onchange({ target: { value: 3 } });
  await reply(pending("/api/edits/-1/restore/3")[0], edit(RIDE, 6));
  await reply(pending("/api/ride")[0], ride(6, "4"));
  for (const r of pending("/api/edits/-1/revisions")) await reply(r, []);
  await restoring;
  assert.equal(app.state.ride.settings.speed, "4");
  assert.equal(app.state.edit.edit_id, 6);
});

test("a ride speed set on the command line is shown and built as it is", async () => {
  const { app, page, pending, reply } = load();
  const select = page.querySelector("#ride-speed");
  let html = "";
  let value = "";
  Object.defineProperty(select, "innerHTML", { get: () => html, set: (v) => { html = v; } });
  // like a real select: only a value one of its options has
  Object.defineProperty(select, "value", { get: () => value, set: (v) => { value = html.includes(`value="${v}"`) ? v : ""; } });
  app.state.length = RIDE;
  const loading = app.loadRide();
  await reply(pending("/api/ride")[0], ride(5, "3.1"));
  for (const r of pending("/api/edits/-1/revisions")) await reply(r, []);
  await loading;
  assert.equal(select.value, "3.1");
});

test("removing a setting lets back in what it left out", async () => {
  const { app, page } = load();
  app.state.length = 8;
  const vocabulary = [{ name: "people", label: "People" }];
  await app.loadPreferences(8, preferences(8, { adjustments: { people: 1 }, exclusions: ["people"], vocabulary }));
  const [row] = page.querySelector("#pref-sliders").children;
  row.querySelector("button").onclick();
  assert.equal(JSON.stringify(app.prefsBody().exclusions), "[]");
  assert.equal(JSON.stringify(app.prefsBody().adjustments), "{}");
  // something left out with no slider still gets a row, so it can be let back in
  await app.loadPreferences(8, preferences(8, { adjustments: {}, exclusions: ["people"], vocabulary }));
  assert.equal(app.prefs.rows.length, 1);
  assert.equal(app.prefs.rows[0].name, "people");
});

test("the music preview plays on its own, at its own speed", async () => {
  const { app, page, pending, reply } = load();
  const video = page.querySelector("#video");
  app.state.queue = [{ item: { recording_id: 1, photo: false, speed: 8 }, span: { chapter: 0, start: 0, end: 4 } }];
  app.state.playing = 0;
  video.playbackRate = 8;
  const polling = app.pollPreview();
  await reply(pending("/api/render")[0], { running: false, output: "trip-8s-r1.mp4", preview: true });
  await polling;
  assert.equal(app.state.queue.length, 0);
  assert.equal(video.playbackRate, 1);
  assert.match(video.src, /preview=true/);
});

test("a clip clicked while the video still loads starts where that clip starts", async () => {
  const { app, page } = load();
  const video = page.querySelector("#video");
  const waiting = [];
  video.addEventListener = (name, fn) => { if (name === "loadedmetadata") waiting.push(fn); };
  video.readyState = 0;
  video.src = "";
  const clip = (start, end) => ({ recording_id: 1, photo: false, speed: 1, time: "", spans: [{ chapter: 0, start, end }] });
  app.playItems([clip(1, 3)], [0]);
  video.src = "http://127.0.0.1:8765" + video.src; // as the browser reports it
  app.playItems([clip(8, 10)], [1]);
  for (const fn of waiting) fn(); // the metadata arrives
  assert.equal(video.currentTime, 8);
});
