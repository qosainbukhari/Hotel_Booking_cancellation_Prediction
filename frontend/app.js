"use strict";

/* ---------- Static descriptions of the booking form ---------- */

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
  "September", "October", "November", "December"];
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

const FORM_GROUPS = [
  ["Stay", [
    { name: "hotel", label: "Hotel", type: "select" },
    { name: "arrival_date", label: "Arrival date", type: "date" },
    { name: "lead_time", label: "Lead time (days)", type: "number", help: "Days between booking and arrival" },
    { name: "stays_in_weekend_nights", label: "Weekend nights", type: "number" },
    { name: "stays_in_week_nights", label: "Weeknights", type: "number" },
  ]],
  ["Guests", [
    { name: "adults", label: "Adults", type: "number" },
    { name: "children", label: "Children", type: "number" },
    { name: "babies", label: "Babies", type: "number" },
    { name: "customer_type", label: "Customer type", type: "select" },
  ]],
  ["Booking channel", [
    { name: "market_segment", label: "Market segment", type: "select" },
    { name: "distribution_channel", label: "Distribution channel", type: "select" },
    { name: "deposit_type", label: "Deposit", type: "select" },
    { name: "agent", label: "Travel agent ID", type: "number", optional: true, help: "Empty if booked without an agent" },
    { name: "company", label: "Company ID", type: "number", optional: true, help: "Empty for private bookings" },
    { name: "days_in_waiting_list", label: "Days on waiting list", type: "number" },
  ]],
  ["Room & requests", [
    { name: "reserved_room_type", label: "Room type", type: "select" },
    { name: "meal", label: "Meal plan", type: "select" },
    { name: "adr", label: "Average daily rate (€)", type: "number", step: "0.01" },
    { name: "total_of_special_requests", label: "Special requests", type: "number" },
  ]],
  ["Guest history", [
    { name: "is_repeated_guest", label: "Returning guest", type: "select" },
    { name: "previous_cancellations", label: "Past cancellations", type: "number" },
    { name: "previous_bookings_not_canceled", label: "Past completed stays", type: "number" },
  ]],
];

const OPTION_LABELS = {
  meal: { BB: "Bed & breakfast", HB: "Half board", FB: "Full board", SC: "Self catering", Undefined: "Not specified" },
  is_repeated_guest: { 0: "No", 1: "Yes" },
};
// Room types have no fixed list in the API (the model groups rare ones), so list the known ones.
const ROOM_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"];

const BASE_BOOKING = {
  hotel: "City Hotel", arrival_date: "2017-08-15", lead_time: 120,
  stays_in_weekend_nights: 1, stays_in_week_nights: 3,
  adults: 2, children: 0, babies: 0, customer_type: "Transient",
  market_segment: "Online TA", distribution_channel: "TA/TO", deposit_type: "No Deposit",
  agent: 9, company: "", days_in_waiting_list: 0,
  reserved_room_type: "A", meal: "BB", adr: 110.5,
  total_of_special_requests: 1,
  is_repeated_guest: 0, previous_cancellations: 0, previous_bookings_not_canceled: 0,
};

const PRESETS = {
  safe: {
    ...BASE_BOOKING, hotel: "Resort Hotel", arrival_date: "2017-07-08", lead_time: 12,
    stays_in_weekend_nights: 2, stays_in_week_nights: 5, children: 1,
    market_segment: "Direct", distribution_channel: "Direct", agent: "", adr: 145,
    total_of_special_requests: 2, is_repeated_guest: 1,
    previous_bookings_not_canceled: 2, meal: "HB", reserved_room_type: "E",
  },
  risky: {
    ...BASE_BOOKING, arrival_date: "2017-05-12", lead_time: 290, stays_in_weekend_nights: 0,
    stays_in_week_nights: 2, market_segment: "Groups", distribution_channel: "TA/TO",
    deposit_type: "Non Refund", agent: 1, adr: 75, total_of_special_requests: 0,
    previous_cancellations: 1,
  },
};

const FEATURE_LABELS = {
  lead_time: "Lead time", arrival_month: "Arrival month", arrival_date_week_number: "Arrival week",
  arrival_date_day_of_month: "Day of month", arrival_weekday: "Arrival weekday",
  stays_in_weekend_nights: "Weekend nights", stays_in_week_nights: "Weeknights",
  total_nights: "Length of stay", adults: "Adults", children: "Children", babies: "Babies",
  total_guests: "Party size", is_repeated_guest: "Returning guest",
  previous_cancellations: "Past cancellations", previous_bookings_not_canceled: "Past completed stays",
  days_in_waiting_list: "Waiting list", adr: "Daily rate", adr_per_guest: "Rate per guest",
  total_of_special_requests: "Special requests",
  has_company: "Company booking", hotel: "Hotel", meal: "Meal plan",
  market_segment: "Market segment", distribution_channel: "Channel",
  reserved_room_type: "Room type", deposit_type: "Deposit", customer_type: "Customer type",
  agent: "Travel agent",
};

/* ---------- Helpers ---------- */

const $ = (selector) => document.querySelector(selector);
const pct = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
// Probabilities shown as whole percentages; avoid claiming certainty at the extremes.
const riskPct = (p) => (p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : pct(p, 0));
const fmt = (value, digits = 3) => Number(value).toFixed(digits);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

class ApiError extends Error {
  constructor(message, fields = []) {
    super(message);
    this.fields = fields;
  }
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("Cannot reach the API. Is the server running?");
  }
  if (response.ok) return response;

  let detail = response.statusText;
  const fields = [];
  try {
    const body = await response.json();
    detail = body.detail ?? detail;
  } catch { /* not JSON */ }

  if (Array.isArray(detail)) {
    // Pydantic validation errors.
    const messages = detail.map((err) => {
      const field = err.loc?.[err.loc.length - 1];
      if (field) fields.push(field);
      return `${FEATURE_LABELS[field] ?? field}: ${err.msg}`;
    });
    throw new ApiError(messages.join(" · "), fields);
  }
  if (detail && Array.isArray(detail.errors)) throw new ApiError(detail.errors.join(" · "));
  throw new ApiError(String(detail));
}

function showError(target, error) {
  const node = $(target);
  node.textContent = error ? error.message ?? String(error) : "";
  node.hidden = !error;
}

function formatValue(feature, value) {
  if (value === null || value === undefined) return "—";
  if (feature === "arrival_month") return MONTHS[value - 1] ?? value;
  if (feature === "arrival_weekday") return WEEKDAYS[value] ?? value;
  if (feature === "has_company" || feature === "is_repeated_guest") return value ? "Yes" : "No";
  if (feature === "agent" && value === "none") return "None";
  if (feature === "lead_time") return `${value} days`;
  if (feature === "adr" || feature === "adr_per_guest") return `€${Number(value).toFixed(2)}`;
  if (typeof value === "number" && !Number.isInteger(value)) return value.toFixed(2);
  return String(value);
}

function isoWeek(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  return Math.ceil(((d - yearStart) / 86400000 + 1) / 7);
}

/* ---------- Tabs ---------- */

function selectTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === name));
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.hidden = panel.id !== `tab-${name}`;
  });
  if (name === "model") loadModel();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    history.replaceState(null, "", `#${tab.dataset.tab}`);
    selectTab(tab.dataset.tab);
  });
});

/* ---------- Status ---------- */

let modelLoaded = false;

async function refreshStatus() {
  const status = $("#status");
  try {
    const health = await (await api("/health")).json();
    modelLoaded = health.model_loaded;
    status.className = `status ${modelLoaded ? "ok" : "bad"}`;
    $("#status-text").textContent = modelLoaded
      ? `Model v${health.model_version} live`
      : "No model loaded — run hc-train";
  } catch (error) {
    modelLoaded = false;
    status.className = "status bad";
    $("#status-text").textContent = "API offline";
  }
}

/* ---------- Single booking ---------- */

async function buildForm() {
  let properties = {};
  try {
    const spec = await (await api("/openapi.json")).json();
    properties = spec.components.schemas.Booking.properties;
  } catch { /* fall back to free inputs */ }

  const enumFor = (name) => {
    const prop = properties[name] ?? {};
    return prop.enum ?? prop.anyOf?.find((option) => option.enum)?.enum ?? null;
  };

  const container = $("#form-groups");

  for (const [title, fields] of FORM_GROUPS) {
    const grid = el("div", { class: "fields" });
    for (const field of fields) {
      const id = `f-${field.name}`;
      let input;
      if (field.type === "select") {
        const options = field.name === "reserved_room_type" ? ROOM_TYPES : enumFor(field.name) ?? [];
        input = el("select", { id, name: field.name },
          options.map((value) => el("option", { value }, OPTION_LABELS[field.name]?.[value] ?? value)));
      } else {
        input = el("input", { id, name: field.name, type: field.type });
        if (field.type === "number") {
          input.min = "0";
          input.step = field.step ?? "1";
          input.inputMode = field.step ? "decimal" : "numeric";
        }
        if (!field.optional) input.required = true;
        if (field.optional) input.placeholder = "None";
      }
      grid.append(el("div", { class: "field", "data-field": field.name },
        el("label", { for: id }, field.label), input, field.help ? el("small", {}, field.help) : null));
    }
    container.append(el("fieldset", {}, el("legend", {}, title), grid));
  }
  fillForm(BASE_BOOKING);
}

function fillForm(values) {
  const form = $("#booking-form");
  for (const [name, value] of Object.entries(values)) {
    if (form.elements[name]) form.elements[name].value = value ?? "";
  }
  form.querySelectorAll(".field.invalid").forEach((f) => f.classList.remove("invalid"));
}

function readForm() {
  const form = $("#booking-form");
  const booking = {};
  for (const [, fields] of FORM_GROUPS) {
    for (const field of fields) {
      const raw = form.elements[field.name].value.trim();
      if (field.type === "date") continue;
      if (field.type === "number" || field.name === "is_repeated_guest") {
        booking[field.name] = raw === "" ? (field.optional ? null : NaN) : Number(raw);
      } else {
        booking[field.name] = raw;
      }
    }
  }

  const missing = Object.entries(booking).filter(([, v]) => Number.isNaN(v)).map(([k]) => k);
  const dateText = form.elements.arrival_date.value;
  if (!dateText) missing.push("arrival_date");
  if (missing.length) throw new ApiError("Please fill in every required field.", missing);

  const [year, month, day] = dateText.split("-").map(Number);
  Object.assign(booking, {
    arrival_date_year: year,
    arrival_date_month: MONTHS[month - 1],
    arrival_date_day_of_month: day,
    arrival_date_week_number: isoWeek(new Date(year, month - 1, day)),
  });
  return booking;
}

function markInvalid(fields) {
  const form = $("#booking-form");
  form.querySelectorAll(".field.invalid").forEach((f) => f.classList.remove("invalid"));
  for (let name of fields) {
    if (name.startsWith("arrival_date")) name = "arrival_date";
    form.querySelector(`[data-field="${name}"]`)?.classList.add("invalid");
  }
}

function gaugePoint(fraction, radius) {
  const angle = Math.PI * (1 - fraction);
  return [60 + radius * Math.cos(angle), 64 - radius * Math.sin(angle)];
}

function renderPrediction(result) {
  $("#result-empty").hidden = true;
  $("#result-body").hidden = false;

  const p = result.cancel_probability;
  const fill = $("#gauge-fill");
  fill.style.strokeDasharray = `${Math.max(p * 100, 0.5)} 100`;
  fill.classList.toggle("risk", result.will_cancel);
  const [x1, y1] = gaugePoint(result.threshold, 42);
  const [x2, y2] = gaugePoint(result.threshold, 58);
  Object.entries({ x1, y1, x2, y2 }).forEach(([k, v]) => $("#gauge-threshold").setAttribute(k, v));

  $("#prob-value").textContent = riskPct(p);
  const verdict = $("#verdict");
  verdict.className = `verdict ${result.will_cancel ? "risk" : "safe"}`;
  verdict.textContent = result.will_cancel ? "Likely to cancel" : "Likely to show up";
  $("#threshold-note").textContent =
    `Flagged above ${pct(result.threshold, 0)} (the tick on the dial), the cost-optimal cut-off · model v${result.model_version}`;

  const rows = result.explanation ?? [];
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.contribution)), 0.01);
  $("#explanation").replaceChildren(...rows.map((row) => {
    const up = row.contribution > 0;
    const width = (Math.abs(row.contribution) / maxAbs) * 50;
    return el("div", { class: "explain-row", title: `${up ? "+" : ""}${fmt(row.contribution)} log-odds` },
      el("div", { class: "explain-label" },
        el("span", {}, FEATURE_LABELS[row.feature] ?? row.feature),
        el("span", { class: "value" }, formatValue(row.feature, row.value))),
      el("div", { class: "diverging" },
        el("span", { class: `bar ${up ? "up" : "down"}`, style: `width:${width}%` })));
  }));
}

$("#booking-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("#form-error", null);
  const button = $("#predict-btn");
  button.disabled = true;
  try {
    const booking = readForm();
    markInvalid([]);
    const response = await api("/predict?explain=true&top_k=8", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(booking),
    });
    renderPrediction(await response.json());
    // On narrow screens the result sits below the form; bring it into view after a click.
    if (event.submitter && window.innerWidth < 900) {
      $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    markInvalid(error.fields ?? []);
    showError("#form-error", error);
  } finally {
    button.disabled = false;
  }
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    fillForm(PRESETS[button.dataset.preset]);
    $("#booking-form").requestSubmit();
  });
});

/* ---------- File uploads ---------- */

function setupDropzone(zone, onFile) {
  const input = zone.querySelector("input");
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
}

async function uploadCsv(path, file, zone) {
  const body = new FormData();
  body.append("file", file);
  zone.classList.add("busy");
  try {
    return await api(path, { method: "POST", body });
  } finally {
    zone.classList.remove("busy");
  }
}

/* ---------- Batch scoring ---------- */

let batchFile = null;
let batchData = null;
const TABLE_LIMIT = 200;

function tile(label, value, sub) {
  return el("div", { class: "tile" },
    el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    sub ? el("div", { class: "sub" }, sub) : null);
}

function metricRows(metrics) {
  const names = [["roc_auc", "ROC-AUC"], ["pr_auc", "PR-AUC"], ["precision", "Precision"],
    ["recall", "Recall"], ["f1", "F1 score"], ["accuracy", "Accuracy"]];
  return el("div", { class: "metric-list" }, names.map(([key, label]) =>
    el("div", { class: "metric-row" },
      el("span", {}, label),
      el("div", { class: "meter" }, el("span", { style: `width:${metrics[key] * 100}%` })),
      el("b", {}, fmt(metrics[key])))));
}

function renderBatch(data) {
  batchData = data;
  $("#batch-result").hidden = false;
  const flaggedShare = data.n_predicted_cancel / data.n_rows;
  const tiles = [
    tile("Bookings scored", data.n_rows.toLocaleString()),
    tile("Flagged to cancel", data.n_predicted_cancel.toLocaleString(), `${pct(flaggedShare)} of bookings`),
    tile("Average risk", pct(data.mean_probability), `threshold ${pct(data.threshold, 0)}`),
  ];
  if (data.rows.length && "actual" in data.rows[0]) {
    const actual = data.rows.reduce((sum, row) => sum + row.actual, 0);
    tiles.push(tile("Actually cancelled", actual.toLocaleString(), `${pct(actual / data.n_rows)} of bookings`));
  }
  if (data.metrics) tiles.push(tile("ROC-AUC", fmt(data.metrics.roc_auc), "on this file"));
  if (data.metrics?.cost_saving != null) {
    tiles.push(tile("Cost saved", pct(data.metrics.cost_saving, 0),
      `€${fmt(data.metrics.cost_per_booking, 2)} per booking vs no model`));
  }
  $("#batch-tiles").replaceChildren(...tiles);

  // Histogram
  const maxCount = Math.max(...data.histogram, 1);
  const bins = data.histogram.map((count, i) => {
    const flagged = (i + 0.5) / data.histogram.length >= data.threshold;
    return el("div", {
      class: `bin ${flagged ? "flagged" : ""}`,
      style: `height:${(count / maxCount) * 100}%`,
      title: `${i * 5}–${(i + 1) * 5}%: ${count.toLocaleString()} bookings`,
    });
  });
  const line = el("div", { class: "threshold-line", style: `left:${data.threshold * 100}%` },
    el("span", {}, `threshold ${pct(data.threshold, 0)}`));
  $("#histogram").replaceChildren(...bins, line);

  $("#batch-metrics").replaceChildren(data.metrics
    ? metricRows(data.metrics)
    : el("p", { class: "muted" }, "Add an is_canceled column (0/1) to the file to compare predictions with what actually happened."));

  if (data.warnings.length) {
    $("#batch-metrics").append(el("p", { class: "muted small" }, `Data warnings: ${data.warnings.join("; ")}`));
  }
  renderBatchTable();
}

function renderBatchTable() {
  const onlyFlagged = $("#only-flagged").checked;
  let rows = [...batchData.rows].sort((a, b) => b.cancel_probability - a.cancel_probability);
  if (onlyFlagged) rows = rows.filter((row) => row.will_cancel);
  const total = rows.length;
  rows = rows.slice(0, TABLE_LIMIT);
  const hasActual = rows.length > 0 && "actual" in rows[0];

  const head = el("tr", {},
    el("th", { class: "num" }, "Row"), el("th", {}, "Hotel"), el("th", {}, "Arrival"),
    el("th", { class: "num" }, "Lead time"), el("th", { class: "num" }, "Daily rate"),
    el("th", {}, "Deposit"), el("th", {}, "Segment"), el("th", {}, "Risk"),
    el("th", {}, "Prediction"), hasActual ? el("th", {}, "Actual") : null);

  const body = rows.map((row) => {
    const risk = row.will_cancel;
    const arrival = row.arrival_date_month
      ? `${row.arrival_date_day_of_month} ${String(row.arrival_date_month).slice(0, 3)} ${row.arrival_date_year}`
      : "—";
    return el("tr", {},
      el("td", { class: "num" }, row.row),
      el("td", {}, row.hotel ?? "—"),
      el("td", {}, arrival),
      el("td", { class: "num" }, row.lead_time ?? "—"),
      el("td", { class: "num" }, row.adr == null ? "—" : `€${Number(row.adr).toFixed(0)}`),
      el("td", {}, row.deposit_type ?? "—"),
      el("td", {}, row.market_segment ?? "—"),
      el("td", {}, el("div", { class: "prob-cell" },
        el("div", { class: "meter" }, el("span", { class: risk ? "risk" : "safe", style: `width:${row.cancel_probability * 100}%` })),
        el("span", {}, riskPct(row.cancel_probability)))),
      el("td", {}, el("span", { class: `pill ${risk ? "risk" : "safe"}` }, risk ? "Will cancel" : "Will stay")),
      hasActual ? el("td", {}, el("span", { class: `pill ${row.actual ? "risk" : "neutral"}` },
        row.actual ? "Cancelled" : "Stayed")) : null);
  });
  $("#batch-table").replaceChildren(el("thead", {}, head), el("tbody", {}, body));
  $("#table-note").textContent = total > TABLE_LIMIT
    ? `Showing the ${TABLE_LIMIT} riskiest of ${total.toLocaleString()} bookings. Download the CSV for all of them.`
    : `${total.toLocaleString()} bookings.`;
}

setupDropzone(document.querySelector('[data-target="batch"]'), async (file) => {
  showError("#batch-error", null);
  $("#batch-file-name").textContent = file.name;
  batchFile = file;
  try {
    const response = await uploadCsv("/predict/csv", file, document.querySelector('[data-target="batch"]'));
    renderBatch(await response.json());
  } catch (error) {
    $("#batch-result").hidden = true;
    showError("#batch-error", error);
  }
});

$("#only-flagged").addEventListener("change", renderBatchTable);

$("#download-btn").addEventListener("click", async () => {
  if (!batchFile) return;
  const button = $("#download-btn");
  button.disabled = true;
  try {
    const response = await uploadCsv("/predict/csv?output=csv", batchFile, document.querySelector('[data-target="batch"]'));
    const url = URL.createObjectURL(await response.blob());
    const link = el("a", { href: url, download: batchFile.name.replace(/\.csv$/i, "") + "_predictions.csv" });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    showError("#batch-error", error);
  } finally {
    button.disabled = false;
  }
});

/* ---------- Drift ---------- */

function renderDrift(report) {
  $("#drift-result").hidden = false;
  const banner = $("#drift-banner");
  banner.className = `banner ${report.dataset_drift ? "drift" : "ok"}`;
  banner.replaceChildren(
    el("strong", {}, report.dataset_drift
      ? "Significant drift — consider retraining"
      : report.drifted_features.length ? "Some features shifted, dataset still within limits" : "No drift detected"),
    el("span", { class: "muted" },
      `${report.drifted_features.length} of ${report.n_features} features drifted ` +
      `(${pct(report.drift_share, 0)}; limit ${pct(report.drift_share_threshold, 0)}) · ` +
      `${report.n_rows.toLocaleString()} rows compared with the training data of model v${report.model_version}`));

  const features = Object.entries(report.features).sort((a, b) => b[1].psi - a[1].psi);
  const threshold = report.psi_threshold;
  // Cap the axis at PSI 1 so a few extreme features don't flatten the rest; longer bars clip.
  const scale = Math.max(Math.min(features[0]?.[1].psi ?? 0, 1), threshold * 2.5);
  $("#drift-legend").textContent = `Dashed line = drift threshold (PSI ${threshold})`;
  $("#drift-bars").replaceChildren(...features.map(([name, result]) => {
    const level = result.drifted ? "drifted" : result.psi > 0.1 ? "moderate" : "stable";
    const label = { drifted: "Drift", moderate: "Shift", stable: "Stable" }[level];
    const pill = { drifted: "risk", moderate: "warn", stable: "safe" }[level];
    return el("div", { class: "drift-row" },
      el("span", { title: name }, FEATURE_LABELS[name] ?? name),
      el("div", { class: "drift-track" },
        el("span", { class: `bar ${level}`, style: `width:${Math.min(Math.max((result.psi / scale) * 100, 0.5), 100)}%` }),
        el("span", { class: "limit", style: `left:${(threshold / scale) * 100}%` })),
      el("span", { class: "num" }, fmt(result.psi)),
      el("span", {}, el("span", { class: `pill ${pill}` }, label)));
  }));
}

setupDropzone(document.querySelector('[data-target="drift"]'), async (file) => {
  showError("#drift-error", null);
  $("#drift-file-name").textContent = file.name;
  try {
    const response = await uploadCsv("/monitoring/drift", file, document.querySelector('[data-target="drift"]'));
    renderDrift(await response.json());
  } catch (error) {
    $("#drift-result").hidden = true;
    showError("#drift-error", error);
  }
});

/* ---------- Model ---------- */

async function loadModel() {
  try {
    const [info, registry] = await Promise.all([
      api("/model").then((r) => r.json()),
      api("/model/versions").then((r) => r.json()),
    ]);
    const trained = info.trained_at ? new Date(info.trained_at).toLocaleString() : "unknown";
    $("#model-title").textContent = `Production model v${info.version}`;
    $("#model-sub").textContent =
      `Trained ${trained} · XGBoost, ${info.n_trees} trees, ${info.n_features} model features · MLflow registry`;
    const m = info.test_metrics ?? {};
    $("#model-tiles").replaceChildren(
      tile("ROC-AUC", fmt(m.roc_auc), "held-out test set"),
      tile("PR-AUC", fmt(m.pr_auc)),
      tile("F1 score", fmt(m.f1)),
      tile("Precision", fmt(m.precision), "flagged that did cancel"),
      tile("Recall", fmt(m.recall), "cancellations caught"),
      tile("Threshold", pct(info.threshold, 1), "minimises cost"),
      tile("Cost / booking", `€${fmt(m.cost_per_booking, 2)}`,
        info.costs ? `missed €${info.costs.false_negative} · false alarm €${info.costs.false_positive}` : ""));

    const head = el("tr", {}, ["Version", "Registered", "ROC-AUC", "PR-AUC", "F1", "Cost / booking", "Status"]
      .map((h, i) => el("th", { class: i >= 2 && i <= 5 ? "num" : "" }, h)));
    const metric = (v, key, digits = 3) => (v.test_metrics[key] == null ? "—" : fmt(v.test_metrics[key], digits));
    const rows = [...registry.versions].reverse().map((v) => el("tr", {},
      el("td", {}, el("b", {}, `v${v.version}`)),
      el("td", {}, new Date(v.registered_at).toLocaleString()),
      el("td", { class: "num" }, metric(v, "roc_auc")),
      el("td", { class: "num" }, metric(v, "pr_auc")),
      el("td", { class: "num" }, metric(v, "f1")),
      el("td", { class: "num" }, v.test_metrics.cost_per_booking == null ? "—" : `€${metric(v, "cost_per_booking", 2)}`),
      el("td", {}, v.is_production
        ? el("span", { class: "pill safe" }, "Production")
        : el("span", { class: "pill neutral" }, "Archived"))));
    $("#versions-table").replaceChildren(el("thead", {}, head), el("tbody", {}, rows));
  } catch (error) {
    $("#model-title").textContent = "No model loaded";
    $("#model-sub").textContent = error.message;
    $("#model-tiles").replaceChildren();
    $("#versions-table").replaceChildren();
  }
}

$("#reload-btn").addEventListener("click", async () => {
  const button = $("#reload-btn");
  button.disabled = true;
  try {
    await api("/model/reload", { method: "POST" });
  } catch (error) {
    $("#model-sub").textContent = error.message;
  } finally {
    button.disabled = false;
    await refreshStatus();
    if (modelLoaded) loadModel();
  }
});

/* ---------- Start ---------- */

(async function init() {
  await Promise.all([refreshStatus(), buildForm()]);
  const initial = location.hash.slice(1);
  if (["predict", "batch", "monitor", "model"].includes(initial)) selectTab(initial);
  if (modelLoaded) $("#booking-form").requestSubmit();
})();
