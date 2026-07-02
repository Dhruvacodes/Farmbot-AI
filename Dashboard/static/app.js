const sensorSpecs = [
  ["N", "Nitrogen", "mg/kg"],
  ["P", "Phosphorus", "mg/kg"],
  ["K", "Potassium", "mg/kg"],
  ["EC", "EC", "uS/cm"],
  ["pH", "pH", ""],
  ["moisture", "Moisture", "%"],
  ["temp", "Temperature", "C"],
];

const predictionSpecs = [
  ["delta_N", "Delta N", "mg/kg"],
  ["delta_P", "Delta P", "mg/kg"],
  ["delta_K", "Delta K", "mg/kg"],
  ["irrigation_ml", "Irrigation", "mL"],
  ["pH_adj", "pH Adj", ""],
];

let currentState = null;
let selectedMapKey = "N";
const thermalView = {
  yaw: -0.72,
  pitch: 0.76,
  dragging: false,
  dragX: 0,
  dragY: 0,
  hover: null,
};

const $ = (id) => document.getElementById(id);

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}

function statusForValue(key, value, ranges) {
  if (value === null || value === undefined || !ranges[key]) return "unknown";
  const [min, max] = ranges[key];
  if (key === "pH" && (value < 4.5 || value > 7.0)) return "critical";
  if (key === "EC" && value >= 1500) return "critical";
  if (key === "moisture" && value <= 15) return "critical";
  if (value < min) return "low";
  if (value > max) return "high";
  return "ok";
}

function renderSensors(state) {
  const grid = $("sensorGrid");
  grid.innerHTML = "";
  sensorSpecs.forEach(([key, label, unit]) => {
    const value = state.sensors?.[key];
    const range = state.optimal_ranges?.[key];
    const status = statusForValue(key, value, state.optimal_ranges || {});
    const card = document.createElement("article");
    card.className = `metric ${status}`;
    const digits = key === "pH" ? 2 : 1;
    card.innerHTML = `
      <div class="label-row"><span>${label}</span><span>${status.toUpperCase()}</span></div>
      <div class="value">${formatNumber(value, digits)} <small>${unit}</small></div>
      <div class="range">Optimal ${range ? range.join(" - ") : "--"}</div>
    `;
    grid.appendChild(card);
  });
}

function renderPredictions(state) {
  const prediction = state.predictions || {};
  const actions = prediction.npk_action || prediction;
  $("predictionSource").textContent = prediction.source || "unknown";
  const grid = $("predictionGrid");
  grid.innerHTML = "";

  predictionSpecs.forEach(([key, label, unit]) => {
    const value = actions?.[key];
    const item = document.createElement("div");
    const digits = key === "pH_adj" ? 3 : 2;
    item.innerHTML = `<span>${label}</span><strong>${formatNumber(value, digits)}</strong><small>${unit}</small>`;
    grid.appendChild(item);
  });
}

function renderVision(state) {
  const vision = state.vision || {};
  $("leafStatus").textContent = vision.leaf_status || "--";
  $("leafConfidence").textContent = vision.leaf_confidence !== null && vision.leaf_confidence !== undefined ? `${Math.round(vision.leaf_confidence * 100)}% confidence` : "--";
  $("leafSeverity").textContent = vision.leaf_severity || "--";
  $("leafDetectionCount").textContent = vision.leaf_detection_count ?? "--";
  $("fruitCount").textContent = vision.fruit_count ?? "--";
  $("ripeness").textContent = vision.ripeness || "--";
  $("ripenessConfidence").textContent = vision.ripeness_confidence !== null && vision.ripeness_confidence !== undefined ? `${Math.round(vision.ripeness_confidence * 100)}% confidence` : "--";
  $("weight").textContent = vision.estimated_weight_kg !== null && vision.estimated_weight_kg !== undefined ? `${formatNumber(vision.estimated_weight_kg, 2)} kg` : "--";

  const leafModel = state.model?.leaf || {};
  $("leafModelClasses").textContent = leafModel.classes?.length ? leafModel.classes.join(" / ") : "classes pending";
}

function renderSystem(state) {
  const connection = state.connection || "unknown";
  const pill = $("connectionPill");
  pill.textContent = connection.toUpperCase();
  pill.className = `pill ${connection}`;

  $("subtitle").textContent = `Updated ${state.updated_at || "--"} | Growth stage ${state.sensors?.growth_stage ?? "--"}`;
  $("updatedAt").textContent = state.updated_at || "--";
  $("jetsonName").textContent = state.system?.jetson_name || "--";
  $("latency").textContent = state.system?.latency_ms ? `${formatNumber(state.system.latency_ms, 1)} ms` : "--";
  $("modelStatus").textContent = state.model?.status || state.predictions?.model_status || "--";
  $("leafModelStatus").textContent = state.model?.leaf?.status || "--";
  $("usbFps").textContent = state.system?.fps_usb ? `${formatNumber(state.system.fps_usb, 1)} fps` : "-- fps";
  $("simulationButton").textContent = state.simulation_enabled ? "S" : "J";
  $("simulationButton").title = state.simulation_enabled ? "Simulation is on" : "Waiting for Jetson telemetry";
}

function setFeed(elementId, placeholderId, url) {
  const img = $(elementId);
  const frame = img.parentElement;
  if (url) {
    if (img.src !== url) img.src = url;
    frame.classList.add("has-feed");
  } else {
    img.removeAttribute("src");
    frame.classList.remove("has-feed");
    $(placeholderId).style.display = "";
  }
}

function renderStreams(state) {
  const storedUsb = localStorage.getItem("farmbot_usb_stream") || "";
  const usb = storedUsb || state.streams?.usb_cam || "";
  if (document.activeElement !== $("usbUrlInput")) {
    $("usbUrlInput").value = usb;
  }
  setFeed("usbFeed", "usbPlaceholder", usb);
}

function renderEvents(state) {
  const list = $("eventList");
  list.innerHTML = "";
  const events = [...(state.events || [])].reverse();
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "event";
    empty.textContent = "No events yet.";
    list.appendChild(empty);
    return;
  }
  events.slice(0, 20).forEach((event) => {
    const row = document.createElement("div");
    row.className = `event ${event.level || "info"}`;
    const level = document.createElement("strong");
    level.textContent = event.level || "info";
    const message = document.createTextNode(` ${event.message || ""}`);
    const breakLine = document.createElement("br");
    const time = document.createElement("small");
    time.textContent = event.time || "";
    row.append(level, message, breakLine, time);
    list.appendChild(row);
  });
}

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function lerp(start, end, amount) {
  return start + (end - start) * amount;
}

function heatColor(amount) {
  const stops = [
    [39, 98, 143],
    [40, 122, 67],
    [217, 155, 34],
    [180, 35, 24],
  ];
  const scaled = clamp(amount) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const from = stops[index];
  const to = stops[index + 1];
  return `rgb(${Math.round(lerp(from[0], to[0], local))}, ${Math.round(lerp(from[1], to[1], local))}, ${Math.round(lerp(from[2], to[2], local))})`;
}

function normalizedMapValue(key, value, ranges) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0.5;
  const numeric = Number(value);
  const range = ranges?.[key];
  if (!range) return clamp(numeric / 100);
  const [min, max] = range;
  const span = Math.max(1e-6, max - min);
  return clamp((numeric - min) / span);
}

function mapCellValue(base, row, col, rows, cols) {
  const x = col / Math.max(1, cols - 1);
  const y = row / Math.max(1, rows - 1);
  const wave = Math.sin((x * 2.8 + base * 1.6) * Math.PI) * 0.14;
  const ridge = Math.cos((y * 3.4 - base) * Math.PI) * 0.1;
  const diagonal = (x - y) * 0.12;
  return clamp(base + wave + ridge + diagonal);
}

function drawRoverPath(ctx, width, height) {
  ctx.save();
  ctx.strokeStyle = "rgba(24, 35, 24, 0.42)";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 7]);
  ctx.beginPath();
  ctx.moveTo(width * 0.08, height * 0.78);
  ctx.bezierCurveTo(width * 0.22, height * 0.34, width * 0.34, height * 0.86, width * 0.48, height * 0.46);
  ctx.bezierCurveTo(width * 0.61, height * 0.1, width * 0.76, height * 0.62, width * 0.92, height * 0.22);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#152016";
  ctx.beginPath();
  ctx.arc(width * 0.92, height * 0.22, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function thermalPoint(base, row, col, rows, cols) {
  const heat = mapCellValue(base, row, col, rows, cols);
  const nx = col / Math.max(1, cols - 1);
  const ny = row / Math.max(1, rows - 1);
  const mound = Math.sin(nx * Math.PI * 2.2) * Math.cos(ny * Math.PI * 2.6) * 0.11;
  return clamp(heat + mound);
}

function projectThermal(x, y, z, width, height, scale) {
  const yaw = thermalView.yaw;
  const pitch = thermalView.pitch;
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const rx = x * cosY - y * sinY;
  const ry = x * sinY + y * cosY;
  const rz = z;
  const sy = ry * Math.cos(pitch) - rz * Math.sin(pitch);
  return {
    x: width * 0.5 + rx * scale,
    y: height * 0.58 + sy * scale,
    depth: ry + rz * 0.35,
  };
}

function thermalGrid(state) {
  const key = selectedMapKey;
  const value = state.sensors?.[key];
  const base = normalizedMapValue(key, value, state.optimal_ranges || {});
  const rows = 18;
  const cols = 26;
  const points = [];

  for (let row = 0; row < rows; row += 1) {
    const line = [];
    for (let col = 0; col < cols; col += 1) {
      const amount = thermalPoint(base, row, col, rows, cols);
      const x = (col / (cols - 1) - 0.5) * 2.1;
      const y = (row / (rows - 1) - 0.5) * 1.45;
      const z = amount * 0.56;
      line.push({ row, col, amount, x, y, z });
    }
    points.push(line);
  }

  return { key, value, rows, cols, points };
}

function drawThermalSurface(state) {
  const canvas = $("thermalMapCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const grid = thermalGrid(state);
  const scale = Math.min(width / 3.0, height / 1.8);
  const cells = [];
  const projected = grid.points.map((line) =>
    line.map((point) => ({
      ...point,
      screen: projectThermal(point.x, point.y, point.z, width, height, scale),
      floor: projectThermal(point.x, point.y, 0, width, height, scale),
    }))
  );

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#1c1d19";
  ctx.fillRect(0, 0, width, height);

  for (let row = 0; row < grid.rows - 1; row += 1) {
    for (let col = 0; col < grid.cols - 1; col += 1) {
      const corners = [
        projected[row][col],
        projected[row][col + 1],
        projected[row + 1][col + 1],
        projected[row + 1][col],
      ];
      const amount = corners.reduce((sum, point) => sum + point.amount, 0) / corners.length;
      const depth = corners.reduce((sum, point) => sum + point.screen.depth, 0) / corners.length;
      cells.push({ row, col, corners, amount, depth });
    }
  }

  cells.sort((a, b) => a.depth - b.depth);
  cells.forEach((cell) => {
    const path = new Path2D();
    path.moveTo(cell.corners[0].screen.x, cell.corners[0].screen.y);
    cell.corners.slice(1).forEach((point) => path.lineTo(point.screen.x, point.screen.y));
    path.closePath();
    ctx.fillStyle = heatColor(cell.amount);
    ctx.fill(path);
    ctx.strokeStyle = "rgba(20, 24, 17, 0.28)";
    ctx.lineWidth = 1;
    ctx.stroke(path);
    cell.path = path;
  });

  const frontRows = [grid.rows - 1, 0];
  frontRows.forEach((row) => {
    for (let col = 0; col < grid.cols - 1; col += 1) {
      const topA = projected[row][col];
      const topB = projected[row][col + 1];
      const floorB = topB.floor;
      const floorA = topA.floor;
      const side = new Path2D();
      side.moveTo(topA.screen.x, topA.screen.y);
      side.lineTo(topB.screen.x, topB.screen.y);
      side.lineTo(floorB.x, floorB.y);
      side.lineTo(floorA.x, floorA.y);
      side.closePath();
      ctx.fillStyle = heatColor((topA.amount + topB.amount) / 2);
      ctx.globalAlpha = 0.72;
      ctx.fill(side);
      ctx.globalAlpha = 1;
    }
  });

  const hover = thermalView.hover;
  if (hover) {
    const match = cells.find((cell) => cell.row === hover.row && cell.col === hover.col);
    if (match) {
      ctx.strokeStyle = "#fffef9";
      ctx.lineWidth = 3;
      ctx.stroke(match.path);
    }
  }

  ctx.fillStyle = "rgba(255, 254, 249, 0.9)";
  ctx.fillRect(18, 18, 190, 52);
  ctx.fillStyle = "#182318";
  ctx.font = "700 18px sans-serif";
  ctx.fillText(`${grid.key}: ${formatNumber(grid.value, grid.key === "pH" ? 2 : 1)}`, 32, 50);

  const label = $("thermalMapMetricLabel");
  if (label) label.textContent = `${grid.key} thermal surface`;
  canvas._thermalCells = cells;
}

function renderFieldMap(state) {
  const canvas = $("fieldMapCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const key = selectedMapKey;
  const value = state.sensors?.[key];
  const base = normalizedMapValue(key, value, state.optimal_ranges || {});
  const rows = 9;
  const cols = 18;
  const gutter = 3;
  const cellW = width / cols;
  const cellH = height / rows;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#efe7d1";
  ctx.fillRect(0, 0, width, height);

  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const amount = mapCellValue(base, row, col, rows, cols);
      const x = col * cellW + gutter;
      const y = row * cellH + gutter;
      ctx.fillStyle = heatColor(amount);
      ctx.fillRect(x, y, cellW - gutter * 2, cellH - gutter * 2);
    }
  }

  ctx.fillStyle = "rgba(255, 254, 249, 0.34)";
  for (let col = 1; col < cols; col += 2) {
    ctx.fillRect(col * cellW - 1, 0, 2, height);
  }

  drawRoverPath(ctx, width, height);

  ctx.fillStyle = "rgba(255, 254, 249, 0.88)";
  ctx.fillRect(14, 14, 172, 48);
  ctx.strokeStyle = "rgba(24, 35, 24, 0.16)";
  ctx.strokeRect(14.5, 14.5, 172, 48);
  ctx.fillStyle = "#182318";
  ctx.font = "700 18px sans-serif";
  ctx.fillText(`${key}: ${formatNumber(value, key === "pH" ? 2 : 1)}`, 26, 42);

  const label = $("fieldMapMetricLabel");
  if (label) label.textContent = `${key} field gradient`;
}

function drawTrend(state) {
  const canvas = $("trendCanvas");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const history = state.history || [];
  if (history.length < 2) {
    ctx.fillStyle = "#627066";
    ctx.font = "14px sans-serif";
    ctx.fillText("Trend will appear after a few samples.", 24, 42);
    return;
  }

  const series = [
    ["moisture", "#1f7a4d", 0, 100],
    ["pH", "#b7791f", 0, 14],
    ["EC", "#2764a3", 400, 1500],
  ];
  ctx.strokeStyle = "#d8ded9";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = 24 + (i * (height - 48)) / 4;
    ctx.beginPath();
    ctx.moveTo(32, y);
    ctx.lineTo(width - 18, y);
    ctx.stroke();
  }

  series.forEach(([key, color, min, max]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    history.forEach((row, idx) => {
      const x = 32 + (idx * (width - 54)) / (history.length - 1);
      const raw = Number(row[key]);
      const normalized = Math.max(0, Math.min(1, (raw - min) / (max - min)));
      const y = height - 24 - normalized * (height - 52);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.font = "13px sans-serif";
  ctx.fillStyle = "#1f7a4d";
  ctx.fillText("Moisture", 38, 22);
  ctx.fillStyle = "#b7791f";
  ctx.fillText("pH", 118, 22);
  ctx.fillStyle = "#2764a3";
  ctx.fillText("EC", 152, 22);
}

function render(state) {
  currentState = state;
  renderSystem(state);
  renderStreams(state);
  renderSensors(state);
  renderPredictions(state);
  renderVision(state);
  renderFieldMap(state);
  drawThermalSurface(state);
  drawTrend(state);
}

async function refresh() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    render(await response.json());
  } catch (error) {
    $("connectionPill").textContent = "OFFLINE";
    $("connectionPill").className = "pill stale";
  }
}

$("applyStreams").addEventListener("click", () => {
  localStorage.setItem("farmbot_usb_stream", $("usbUrlInput").value.trim());
  if (currentState) renderStreams(currentState);
});

$("simulationButton").addEventListener("click", async () => {
  const next = !(currentState?.simulation_enabled);
  await fetch(`/api/simulation/${next}`, { method: "POST" });
  await refresh();
});

document.querySelectorAll(".map-toggle").forEach((button) => {
  button.addEventListener("click", () => {
    selectedMapKey = button.dataset.mapKey || "N";
    document.querySelectorAll(".map-toggle").forEach((item) => item.classList.toggle("active", item === button));
    if (currentState) {
      renderFieldMap(currentState);
      drawThermalSurface(currentState);
    }
  });
});

const thermalCanvas = $("thermalMapCanvas");
if (thermalCanvas) {
  thermalCanvas.addEventListener("pointerdown", (event) => {
    thermalView.dragging = true;
    thermalView.dragX = event.clientX;
    thermalView.dragY = event.clientY;
    thermalCanvas.setPointerCapture(event.pointerId);
  });

  thermalCanvas.addEventListener("pointermove", (event) => {
    if (thermalView.dragging) {
      const dx = event.clientX - thermalView.dragX;
      const dy = event.clientY - thermalView.dragY;
      thermalView.yaw += dx * 0.008;
      thermalView.pitch = clamp(thermalView.pitch + dy * 0.004, 0.38, 1.12);
      thermalView.dragX = event.clientX;
      thermalView.dragY = event.clientY;
      if (currentState) drawThermalSurface(currentState);
      return;
    }

    const rect = thermalCanvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * thermalCanvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * thermalCanvas.height;
    const ctx = thermalCanvas.getContext("2d");
    const cells = thermalCanvas._thermalCells || [];
    const hit = [...cells].reverse().find((cell) => ctx.isPointInPath(cell.path, x, y));
    thermalView.hover = hit ? { row: hit.row, col: hit.col, amount: hit.amount } : null;
    $("thermalReadout").textContent = hit
      ? `${selectedMapKey} zone ${hit.col + 1},${hit.row + 1}: ${Math.round(hit.amount * 100)}%`
      : `${selectedMapKey} surface`;
    if (currentState) drawThermalSurface(currentState);
  });

  thermalCanvas.addEventListener("pointerup", () => {
    thermalView.dragging = false;
  });

  thermalCanvas.addEventListener("pointerleave", () => {
    thermalView.dragging = false;
    thermalView.hover = null;
    $("thermalReadout").textContent = `${selectedMapKey} surface`;
    if (currentState) drawThermalSurface(currentState);
  });
}

refresh();
setInterval(refresh, 1000);
