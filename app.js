import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs";

const canvas = document.querySelector("#editorCanvas");
const ctx = canvas.getContext("2d");
const stage = document.querySelector("#canvasStage");
const imageInput = document.querySelector("#imageInput");
const recognizeButton = document.querySelector("#recognizeButton");
const exportButton = document.querySelector("#exportButton");
const clearButton = document.querySelector("#clearButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const resetViewButton = document.querySelector("#resetViewButton");
const zoomInButton = document.querySelector("#zoomInButton");
const emptyState = document.querySelector("#emptyState");
const statusText = document.querySelector("#statusText");
const unicodeButtons = [...document.querySelectorAll("[data-expression]")];

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm";

const EMOTION_LABELS = {
  happy: "开心",
  sad: "悲伤",
  angry: "生气",
  surprised: "惊讶",
  neutral: "中性",
};
const UNICODE_EMOJI = {
  happy: "😄",
  sad: "😢",
  angry: "😠",
  surprised: "😮",
  neutral: "😐",
};

const DEFAULT_EMOJI_DISPLAY_SIZE = 84;
const MIN_EMOJI_DISPLAY_SIZE = 24;
const BOUNDS_RATIO = 1.05;
const HANDLE_SIZE = 9;
const HANDLE_HIT_SIZE = 14;
const BORDER_HIT_SIZE = 10;
const STAGE_PADDING = 28;
const MIN_ZOOM_RATIO = 0.25;
const MAX_ZOOM_RATIO = 12;
const MAX_AUTO_FACES = 30;
const MAX_DETECTION_DIMENSION = 2400;
const DETECTION_TILE_SIZE = 960;
const FINE_DETECTION_TILE_SIZE = 640;
const DETECTION_TILE_OVERLAP = 0.25;
const DETECTION_TILE_RENDER_SIZE = 1280;
const MAX_DETECTION_TILES = 12;
const DETECTION_TIME_BUDGET_MS = 9000;
const MIN_DETECTED_FACE_SIZE = 18;
const FACE_DEDUPE_IOU = 0.35;
const FACE_DEDUPE_CENTER_RATIO = 0.45;

const state = {
  image: null,
  imageName: "",
  imageWidth: 0,
  imageHeight: 0,
  displayWidth: 0,
  displayHeight: 0,
  scale: 1,
  fitScale: 1,
  panX: 0,
  panY: 0,
  viewReady: false,
  dpr: Math.max(1, window.devicePixelRatio || 1),
  annotations: [],
  selectedId: null,
  lastEmojiSize: null,
  drag: null,
  detecting: false,
  faceLandmarker: null,
  faceLandmarkerPromise: null,
};

function setStatus(text) {
  statusText.textContent = text;
}

function imageStatusText() {
  if (!state.image) {
    return "未导入图片";
  }

  return (
    state.imageName +
    " · " +
    state.imageWidth +
    " × " +
    state.imageHeight +
    "px · " +
    state.annotations.length +
    " 个贴图 · " +
    Math.round(state.scale * 100) +
    "%"
  );
}

function updateControls() {
  const hasImage = Boolean(state.image);
  exportButton.disabled = !hasImage;
  clearButton.disabled = !hasImage || state.annotations.length === 0;
  recognizeButton.disabled = !hasImage || state.detecting;
  zoomOutButton.disabled = !hasImage;
  resetViewButton.disabled = !hasImage;
  zoomInButton.disabled = !hasImage;
  emptyState.classList.toggle("hidden", hasImage);
  const selected = getSelectedAnnotation();
  unicodeButtons.forEach((button) => {
    const active = Boolean(selected && selected.expression === button.dataset.expression);
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = !selected;
  });
}

function setDetecting(isDetecting) {
  state.detecting = isDetecting;
  recognizeButton.classList.toggle("is-loading", isDetecting);
  recognizeButton.innerHTML = isDetecting
    ? "正在识别…"
    : '<span class="button-icon" aria-hidden="true">✦</span>识别并贴图';
  updateControls();
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function makeId() {
  return Date.now() + "-" + Math.random().toString(16).slice(2);
}

function getSelectedAnnotation() {
  return (
    state.annotations.find((annotation) => annotation.id === state.selectedId) ||
    null
  );
}

function canvasToImage(point) {
  return {
    x: (point.x - state.panX) / state.scale,
    y: (point.y - state.panY) / state.scale,
  };
}

function isInsideImage(imagePoint) {
  return (
    imagePoint.x >= 0 &&
    imagePoint.x <= state.imageWidth &&
    imagePoint.y >= 0 &&
    imagePoint.y <= state.imageHeight
  );
}

function clampViewScale(scale) {
  const minScale = Math.max(0.01, state.fitScale * MIN_ZOOM_RATIO);
  const maxScale = Math.max(1, state.fitScale * MAX_ZOOM_RATIO);
  return clamp(scale, minScale, maxScale);
}

function clampPan() {
  if (!state.image) {
    return;
  }

  const imageDisplayWidth = state.imageWidth * state.scale;
  const imageDisplayHeight = state.imageHeight * state.scale;

  if (imageDisplayWidth <= state.displayWidth) {
    state.panX = (state.displayWidth - imageDisplayWidth) / 2;
  } else {
    state.panX = clamp(state.panX, state.displayWidth - imageDisplayWidth, 0);
  }
  if (imageDisplayHeight <= state.displayHeight) {
    state.panY = (state.displayHeight - imageDisplayHeight) / 2;
  } else {
    state.panY = clamp(state.panY, state.displayHeight - imageDisplayHeight, 0);
  }
}

function resetView(redraw) {
  if (!state.image) {
    return;
  }
  state.scale = state.fitScale;
  state.panX = (state.displayWidth - state.imageWidth * state.scale) / 2;
  state.panY = (state.displayHeight - state.imageHeight * state.scale) / 2;
  state.viewReady = true;
  clampPan();

  if (redraw !== false) {
    draw();
  }
}

function zoomAt(point, factor) {
  if (!state.image) {
    return;
  }

  const before = canvasToImage(point);
  const nextScale = clampViewScale(state.scale * factor);

  if (nextScale === state.scale) {
    return;
  }
  state.scale = nextScale;
  state.panX = point.x - before.x * state.scale;
  state.panY = point.y - before.y * state.scale;
  clampPan();
  draw();
}

function zoomFromCenter(factor) {
  zoomAt(
    { x: state.displayWidth / 2, y: state.displayHeight / 2 },
    factor,
  );
}

function fitCanvasToStage() {
  if (!state.image) {
    state.displayWidth = 1;
    state.displayHeight = 1;
    canvas.width = 1;
    canvas.height = 1;
    canvas.style.width = "0px";
    canvas.style.height = "0px";
    draw();
    return;
  }

  const previousCenter = state.viewReady
    ? {
        x: (state.displayWidth / 2 - state.panX) / state.scale,
        y: (state.displayHeight / 2 - state.panY) / state.scale,
      }
    : null;
  state.displayWidth = Math.max(1, Math.round(stage.clientWidth));
  state.displayHeight = Math.max(1, Math.round(stage.clientHeight));
  state.dpr = Math.max(1, window.devicePixelRatio || 1);
  state.fitScale = clamp(
    Math.min(
      (state.displayWidth - STAGE_PADDING * 2) / state.imageWidth,
      (state.displayHeight - STAGE_PADDING * 2) / state.imageHeight,
    ),
    0.01,
    2.5,
  );
  canvas.width = Math.round(state.displayWidth * state.dpr);
  canvas.height = Math.round(state.displayHeight * state.dpr);
  canvas.style.width = state.displayWidth + "px";
  canvas.style.height = state.displayHeight + "px";

  if (!state.viewReady) {
    resetView(false);
  } else {
    state.scale = clampViewScale(state.scale);
    state.panX = state.displayWidth / 2 - previousCenter.x * state.scale;
    state.panY = state.displayHeight / 2 - previousCenter.y * state.scale;
    clampPan();
  }
  draw();
}

function drawAnnotation(targetCtx, annotation, scale, offsetX, offsetY) {
  const size = Math.max(1, annotation.size * scale);
  const x = (offsetX || 0) + annotation.x * scale;
  const y = (offsetY || 0) + annotation.y * scale;

  targetCtx.save();
  targetCtx.font =
    "normal " +
    size +
    'px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
  targetCtx.textAlign = "center";
  targetCtx.textBaseline = "middle";
  targetCtx.translate(x, y);
  targetCtx.rotate(annotation.rotation || 0);
  targetCtx.fillText(annotation.emoji, 0, 0);
  targetCtx.restore();
}

function drawSelection(annotation) {
  const bounds = getBounds(annotation);
  const handles = getHandleCenters(bounds);

  ctx.save();
  ctx.strokeStyle = "#635bff";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
  ctx.setLineDash([]);
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#635bff";
  ctx.lineWidth = 2;
  handles.forEach((handle) => {
    ctx.beginPath();
    ctx.rect(
      handle.x - HANDLE_SIZE / 2,
      handle.y - HANDLE_SIZE / 2,
      HANDLE_SIZE,
      HANDLE_SIZE,
    );
    ctx.fill();
    ctx.stroke();
  });
  ctx.restore();
}

function draw() {
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  ctx.clearRect(0, 0, state.displayWidth, state.displayHeight);

  if (!state.image) {
    return;
  }
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    state.image,
    state.panX,
    state.panY,
    state.imageWidth * state.scale,
    state.imageHeight * state.scale,
  );
  ctx.restore();

  state.annotations.forEach((annotation) => {
    drawAnnotation(ctx, annotation, state.scale, state.panX, state.panY);
  });

  const selected = getSelectedAnnotation();
  if (selected) {
    drawSelection(selected);
  }
}

function getBounds(annotation, scale) {
  const currentScale = scale || state.scale;
  const side = annotation.size * BOUNDS_RATIO * currentScale;
  const centerX = state.panX + annotation.x * currentScale;
  const centerY = state.panY + annotation.y * currentScale;

  return {
    left: centerX - side / 2,
    top: centerY - side / 2,
    right: centerX + side / 2,
    bottom: centerY + side / 2,
    width: side,
    height: side,
    centerX,
    centerY,
  };
}

function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * state.displayWidth,
    y: ((event.clientY - rect.top) / rect.height) * state.displayHeight,
  };
}

function getHandleCenters(bounds) {
  return [
    { x: bounds.left, y: bounds.top },
    { x: bounds.right, y: bounds.top },
    { x: bounds.right, y: bounds.bottom },
    { x: bounds.left, y: bounds.bottom },
  ];
}

function hitHandle(bounds, point) {
  return getHandleCenters(bounds).some(
    (handle) =>
      Math.abs(point.x - handle.x) <= HANDLE_HIT_SIZE &&
      Math.abs(point.y - handle.y) <= HANDLE_HIT_SIZE,
  );
}

function hitBorder(bounds, point) {
  const inExpandedBox =
    point.x >= bounds.left - BORDER_HIT_SIZE &&
    point.x <= bounds.right + BORDER_HIT_SIZE &&
    point.y >= bounds.top - BORDER_HIT_SIZE &&
    point.y <= bounds.bottom + BORDER_HIT_SIZE;

  if (!inExpandedBox) {
    return false;
  }
  return (
    Math.abs(point.x - bounds.left) <= BORDER_HIT_SIZE ||
    Math.abs(point.x - bounds.right) <= BORDER_HIT_SIZE ||
    Math.abs(point.y - bounds.top) <= BORDER_HIT_SIZE ||
    Math.abs(point.y - bounds.bottom) <= BORDER_HIT_SIZE
  );
}

function hitInside(bounds, point) {
  return (
    point.x >= bounds.left &&
    point.x <= bounds.right &&
    point.y >= bounds.top &&
    point.y <= bounds.bottom
  );
}

function hitSelectedBorder(point) {
  const selected = getSelectedAnnotation();
  if (!selected) {
    return null;
  }

  const bounds = getBounds(selected);
  if (hitHandle(bounds, point) || hitBorder(bounds, point)) {
    return { annotation: selected, kind: "resize" };
  }

  return null;
}

function hitAnnotation(point) {
  for (let index = state.annotations.length - 1; index >= 0; index -= 1) {
    const annotation = state.annotations[index];
    if (hitInside(getBounds(annotation), point)) {
      return { annotation, kind: "move" };
    }
  }

  return null;
}

function loadImageFile(file) {
  if (!file) {
    return;
  }

  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(url);
    state.image = image;
    state.imageName = file.name || "image";
    state.imageWidth = image.naturalWidth;
    state.imageHeight = image.naturalHeight;
    state.annotations = [];
    state.selectedId = null;
    state.lastEmojiSize = null;
    state.panX = 0;
    state.panY = 0;
    state.viewReady = false;

    updateControls();
    fitCanvasToStage();
    setStatus(imageStatusText() + " · 点击“识别并贴图”开始");
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    setStatus("图片读取失败");
  };

  image.src = url;
}

function addManualEmojiAt(point) {
  const imagePoint = canvasToImage(point);
  if (!isInsideImage(imagePoint)) {
    setStatus("请在图片范围内右键添加贴图");
    return;
  }

  const expression = "happy";
  const size = state.lastEmojiSize || DEFAULT_EMOJI_DISPLAY_SIZE / state.scale;
  const annotation = {
    id: makeId(),
    expression,
    emoji: UNICODE_EMOJI[expression],
    confidence: 1,
    x: imagePoint.x,
    y: imagePoint.y,
    size,
    rotation: 0,
    source: "manual",
  };
  state.annotations.push(annotation);
  state.selectedId = annotation.id;
  updateControls();
  setStatus(imageStatusText());
  draw();
}

function startMove(annotation, point, pointerId) {
  state.drag = {
    kind: "move",
    id: annotation.id,
    pointerId,
    startPoint: point,
    startX: annotation.x,
    startY: annotation.y,
  };
}

function startResize(annotation, point, pointerId) {
  const bounds = getBounds(annotation);
  const distance = Math.max(
    Math.abs(point.x - bounds.centerX),
    Math.abs(point.y - bounds.centerY),
    (annotation.size * BOUNDS_RATIO * state.scale) / 2,
  );

  state.drag = {
    kind: "resize",
    id: annotation.id,
    pointerId,
    centerX: bounds.centerX,
    centerY: bounds.centerY,
    startDistance: distance,
  };
}

function startPan(point, pointerId) {
  state.drag = {
    kind: "pan",
    pointerId,
    startPoint: point,
    startPanX: state.panX,
    startPanY: state.panY,
  };
}

function updateDrag(point) {
  if (!state.drag) {
    return;
  }

  if (state.drag.kind === "pan") {
    state.panX = state.drag.startPanX + point.x - state.drag.startPoint.x;
    state.panY = state.drag.startPanY + point.y - state.drag.startPoint.y;
    clampPan();
    draw();
    return;
  }
  const annotation = state.annotations.find(
    (item) => item.id === state.drag.id,
  );
  if (!annotation) {
    state.drag = null;
    return;
  }

  if (state.drag.kind === "move") {
    const deltaX = (point.x - state.drag.startPoint.x) / state.scale;
    const deltaY = (point.y - state.drag.startPoint.y) / state.scale;
    annotation.x = clamp(state.drag.startX + deltaX, 0, state.imageWidth);
    annotation.y = clamp(state.drag.startY + deltaY, 0, state.imageHeight);
  }
  if (state.drag.kind === "resize") {
    const distance = Math.max(
      Math.abs(point.x - state.drag.centerX),
      Math.abs(point.y - state.drag.centerY),
    );
    const maxDisplaySize = Math.max(state.displayWidth, state.displayHeight) * 2;
    const displaySize = clamp(
      (distance * 2) / BOUNDS_RATIO,
      MIN_EMOJI_DISPLAY_SIZE,
      maxDisplaySize,
    );
    annotation.size = displaySize / state.scale;
    state.lastEmojiSize = annotation.size;
  }

  draw();
}

function updateCursor(event) {
  if (state.drag) {
    canvas.style.cursor =
      state.drag.kind === "resize"
        ? "nwse-resize"
        : state.drag.kind === "pan"
          ? "grabbing"
          : "move";
    return;
  }

  if (!state.image) {
    canvas.style.cursor = "default";
    return;
  }
  const point = getCanvasPoint(event);
  canvas.style.cursor =
    hitSelectedBorder(point) || hitAnnotation(point) ? "move" : "grab";
}

function finishDrag() {
  if (!state.drag) {
    return;
  }

  const annotation = state.annotations.find(
    (item) => item.id === state.drag.id,
  );
  if (annotation && state.drag.kind === "resize") {
    state.lastEmojiSize = annotation.size;
  }
  state.drag = null;
  setStatus(imageStatusText());
  draw();
}

function toScoreMap(categories) {
  return new Map(
    (categories || []).map((category) => [
      category.categoryName || category.displayName,
      Number(category.score || 0),
    ]),
  );
}

function average(scores, names) {
  if (!names.length) {
    return 0;
  }
  return (
    names.reduce((sum, name) => sum + (scores.get(name) || 0), 0) / names.length
  );
}

function classifyExpression(categories) {
  const scores = toScoreMap(categories);
  const smile = average(scores, ["mouthSmileLeft", "mouthSmileRight"]);
  const cheekSquint = average(scores, [
    "cheekSquintLeft",
    "cheekSquintRight",
  ]);
  const frown = average(scores, ["mouthFrownLeft", "mouthFrownRight"]);
  const browDown = average(scores, ["browDownLeft", "browDownRight"]);
  const eyeSquint = average(scores, ["eyeSquintLeft", "eyeSquintRight"]);
  const eyeWide = average(scores, ["eyeWideLeft", "eyeWideRight"]);
  const mouthPress = average(scores, ["mouthPressLeft", "mouthPressRight"]);
  const browInnerUp = scores.get("browInnerUp") || 0;
  const jawOpen = scores.get("jawOpen") || 0;

  const happyScore = smile * 0.8 + cheekSquint * 0.2;
  const surprisedScore = Math.min(jawOpen, eyeWide);
  const angryScore = Math.min(
    browDown,
    Math.max(mouthPress, eyeSquint),
  );
  const sadScore = Math.min(frown, browInnerUp);

  if (surprisedScore >= 0.42) {
    return { expression: "surprised", confidence: surprisedScore };
  }
  if (angryScore >= 0.34) {
    return { expression: "angry", confidence: angryScore };
  }
  if (sadScore >= 0.34) {
    return { expression: "sad", confidence: sadScore };
  }
  if (happyScore >= 0.2) {
    return { expression: "happy", confidence: happyScore };
  }

  // For this playful sticker workflow, an uncertain face defaults to happy
  // instead of showing a misleading negative emotion.
  return {
    expression: "happy",
    confidence: clamp(happyScore, 0.35, 0.75),
  };
}

function midpoint(first, second) {
  return {
    x: (first.x + second.x) / 2,
    y: (first.y + second.y) / 2,
  };
}

function getLandmarkBounds(landmarks, width, height) {
  const xs = landmarks.map((landmark) => landmark.x * width);
  const ys = landmarks.map((landmark) => landmark.y * height);
  const left = clamp(Math.min(...xs), 0, width);
  const right = clamp(Math.max(...xs), 0, width);
  const top = clamp(Math.min(...ys), 0, height);
  const bottom = clamp(Math.max(...ys), 0, height);

  return {
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
    area: Math.max(0, right - left) * Math.max(0, bottom - top),
    centerX: (left + right) / 2,
    centerY: (top + bottom) / 2,
  };
}

function createDetectionCanvas() {
  const scale = Math.min(
    1,
    MAX_DETECTION_DIMENSION / Math.max(state.imageWidth, state.imageHeight),
  );
  const detectionCanvas = document.createElement("canvas");
  detectionCanvas.width = Math.max(1, Math.round(state.imageWidth * scale));
  detectionCanvas.height = Math.max(1, Math.round(state.imageHeight * scale));
  const detectionContext = detectionCanvas.getContext("2d");
  detectionContext.imageSmoothingEnabled = true;
  detectionContext.imageSmoothingQuality = "high";
  detectionContext.drawImage(
    state.image,
    0,
    0,
    detectionCanvas.width,
    detectionCanvas.height,
  );
  return detectionCanvas;
}

function buildTileStarts(length, tileSize = DETECTION_TILE_SIZE) {
  if (length <= tileSize) {
    return [0];
  }

  const step = tileSize * (1 - DETECTION_TILE_OVERLAP);
  const starts = [0];
  while (true) {
    const next = starts[starts.length - 1] + step;
    if (next + tileSize >= length) {
      break;
    }
    starts.push(Math.round(next));
  }

  const finalStart = length - tileSize;
  if (finalStart > starts[starts.length - 1]) {
    starts.push(finalStart);
  }
  return starts;
}

function takeEvenlyDistributed(items, limit) {
  if (items.length <= limit) {
    return items;
  }
  if (limit <= 0) {
    return [];
  }
  if (limit === 1) {
    return [items[Math.floor(items.length / 2)]];
  }

  return Array.from({ length: limit }, (_, index) => {
    const itemIndex = Math.round((index * (items.length - 1)) / (limit - 1));
    return items[itemIndex];
  });
}

function buildDetectionTiles(
  width,
  height,
  tileSize = DETECTION_TILE_SIZE,
) {
  if (width <= tileSize && height <= tileSize) {
    return [];
  }

  const columns = buildTileStarts(width, tileSize);
  const rows = buildTileStarts(height, tileSize);
  const tiles = [];
  rows.forEach((y) => {
    columns.forEach((x) => {
      tiles.push({
        x,
        y,
        width: Math.min(tileSize, width - x),
        height: Math.min(tileSize, height - y),
      });
    });
  });
  return takeEvenlyDistributed(tiles, MAX_DETECTION_TILES);
}

function mapLandmarksToImage(landmarks, crop, sourceWidth, sourceHeight) {
  return landmarks.map((landmark) => ({
    ...landmark,
    x: clamp((crop.x + landmark.x * crop.width) / sourceWidth, 0, 1),
    y: clamp((crop.y + landmark.y * crop.height) / sourceHeight, 0, 1),
  }));
}

function collectDetections(result, crop, sourceWidth, sourceHeight) {
  return (result.faceLandmarks || [])
    .map((landmarks, index) => {
      const globalLandmarks = mapLandmarksToImage(
        landmarks,
        crop,
        sourceWidth,
        sourceHeight,
      );
      const bounds = getLandmarkBounds(
        globalLandmarks,
        state.imageWidth,
        state.imageHeight,
      );
      const localBounds = getLandmarkBounds(landmarks, 1, 1);

      if (
        Math.min(bounds.width, bounds.height) < MIN_DETECTED_FACE_SIZE
      ) {
        return null;
      }

      return {
        landmarks: globalLandmarks,
        blendshapeCategories: result.faceBlendshapes?.[index]?.categories || [],
        bounds,
        quality: localBounds.area,
      };
    })
    .filter(Boolean);
}

function intersectionOverUnion(first, second) {
  const intersectionLeft = Math.max(first.left, second.left);
  const intersectionTop = Math.max(first.top, second.top);
  const intersectionRight = Math.min(first.right, second.right);
  const intersectionBottom = Math.min(first.bottom, second.bottom);
  const intersectionWidth = Math.max(0, intersectionRight - intersectionLeft);
  const intersectionHeight = Math.max(0, intersectionBottom - intersectionTop);
  const intersectionArea = intersectionWidth * intersectionHeight;
  const unionArea = first.area + second.area - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

function isSameFace(first, second) {
  if (intersectionOverUnion(first.bounds, second.bounds) >= FACE_DEDUPE_IOU) {
    return true;
  }

  const distance = Math.hypot(
    first.bounds.centerX - second.bounds.centerX,
    first.bounds.centerY - second.bounds.centerY,
  );
  const minimumSize = Math.min(
    first.bounds.width,
    first.bounds.height,
    second.bounds.width,
    second.bounds.height,
  );
  return distance <= minimumSize * FACE_DEDUPE_CENTER_RATIO;
}

function mergeDetections(detections) {
  const merged = [];
  detections
    .sort((first, second) => second.quality - first.quality)
    .forEach((candidate) => {
      if (!merged.some((existing) => isSameFace(existing, candidate))) {
        merged.push(candidate);
      }
    });

  return merged
    .sort(
      (first, second) =>
        first.bounds.centerY - second.bounds.centerY ||
        first.bounds.centerX - second.bounds.centerX,
    )
    .slice(0, MAX_AUTO_FACES);
}

function createRenderedTile(sourceCanvas, tile) {
  const renderScale = Math.min(
    DETECTION_TILE_RENDER_SIZE / tile.width,
    DETECTION_TILE_RENDER_SIZE / tile.height,
  );
  const tileCanvas = document.createElement("canvas");
  tileCanvas.width = Math.max(1, Math.round(tile.width * renderScale));
  tileCanvas.height = Math.max(1, Math.round(tile.height * renderScale));
  const tileContext = tileCanvas.getContext("2d");
  tileContext.imageSmoothingEnabled = true;
  tileContext.imageSmoothingQuality = "high";
  tileContext.drawImage(
    sourceCanvas,
    tile.x,
    tile.y,
    tile.width,
    tile.height,
    0,
    0,
    tileCanvas.width,
    tileCanvas.height,
  );
  return tileCanvas;
}

function nextFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(resolve));
}

function faceToAnnotation(landmarks, blendshapeCategories) {
  const bounds = getLandmarkBounds(
    landmarks,
    state.imageWidth,
    state.imageHeight,
  );
  const width = bounds.width;
  const height = bounds.height;
  const expressionResult = classifyExpression(blendshapeCategories);
  const leftEye = midpoint(landmarks[33], landmarks[133]);
  const rightEye = midpoint(landmarks[263], landmarks[362]);
  const rotation = Math.atan2(
    (rightEye.y - leftEye.y) * state.imageHeight,
    (rightEye.x - leftEye.x) * state.imageWidth,
  );

  return {
    id: makeId(),
    expression: expressionResult.expression,
    confidence: expressionResult.confidence,
    emoji: UNICODE_EMOJI[expressionResult.expression],
    x: bounds.centerX,
    y: bounds.centerY + height * 0.035,
    size: clamp(
      Math.max(width * 1.9, height * 1.85),
      36,
      Math.max(state.imageWidth, state.imageHeight) * 0.6,
    ),
    rotation: clamp(rotation, -0.21, 0.21),
    source: "auto",
  };
}

async function ensureFaceLandmarker() {
  if (state.faceLandmarker) {
    return state.faceLandmarker;
  }
  if (state.faceLandmarkerPromise) {
    return state.faceLandmarkerPromise;
  }

  state.faceLandmarkerPromise = (async () => {
    setStatus("正在加载表情识别模型…");
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
      },
      runningMode: "IMAGE",
      numFaces: MAX_AUTO_FACES,
      minFaceDetectionConfidence: 0.35,
      minFacePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    state.faceLandmarker = landmarker;
    state.modelError = false;
    return landmarker;
  })().catch((error) => {
    state.faceLandmarkerPromise = null;
    state.modelError = true;
    throw error;
  });

  return state.faceLandmarkerPromise;
}

async function recognizeFaces() {
  if (!state.image || state.detecting) {
    return;
  }

  setDetecting(true);
  setStatus("正在识别人脸和表情…");
  try {
    const landmarker = await ensureFaceLandmarker();
    const detectionCanvas = createDetectionCanvas();
    const fullCrop = {
      x: 0,
      y: 0,
      width: detectionCanvas.width,
      height: detectionCanvas.height,
    };
    const tiles = buildDetectionTiles(
      detectionCanvas.width,
      detectionCanvas.height,
    );
    const totalPasses = 1 + tiles.length;
    const startedAt = performance.now();
    const detections = [];

    setStatus("正在检测整张图片（1/" + totalPasses + "）…");
    await nextFrame();
    detections.push(
      ...collectDetections(
        landmarker.detect(detectionCanvas),
        fullCrop,
        detectionCanvas.width,
        detectionCanvas.height,
      ),
    );

    let timedOut = false;
    for (let index = 0; index < tiles.length; index += 1) {
      if (performance.now() - startedAt >= DETECTION_TIME_BUDGET_MS) {
        timedOut = true;
        break;
      }

      const tile = tiles[index];
      setStatus(
        "正在检测分块 " + (index + 2) + "/" + totalPasses + "…",
      );
      await nextFrame();
      const tileCanvas = createRenderedTile(detectionCanvas, tile);
      detections.push(
        ...collectDetections(
          landmarker.detect(tileCanvas),
          tile,
          detectionCanvas.width,
          detectionCanvas.height,
        ),
      );
    }

    if (
      !detections.length &&
      !timedOut &&
      performance.now() - startedAt < DETECTION_TIME_BUDGET_MS
    ) {
      const fineTiles = takeEvenlyDistributed(
        buildDetectionTiles(
          detectionCanvas.width,
          detectionCanvas.height,
          FINE_DETECTION_TILE_SIZE,
        ),
        Math.max(0, MAX_DETECTION_TILES - tiles.length),
      );

      for (let index = 0; index < fineTiles.length; index += 1) {
        if (performance.now() - startedAt >= DETECTION_TIME_BUDGET_MS) {
          timedOut = true;
          break;
        }

        setStatus(
          "正在放大检测细分区域 " + (index + 1) + "/" + fineTiles.length + "…",
        );
        await nextFrame();
        const tile = fineTiles[index];
        const tileCanvas = createRenderedTile(detectionCanvas, tile);
        detections.push(
          ...collectDetections(
            landmarker.detect(tileCanvas),
            tile,
            detectionCanvas.width,
            detectionCanvas.height,
          ),
        );
      }
    }

    const faces = mergeDetections(detections);
    if (!faces.length) {
      state.annotations = [];
      state.selectedId = null;
      updateControls();
      draw();
      setStatus("未检测到可识别的人脸");
      return;
    }

    state.annotations = faces.map((face) =>
      faceToAnnotation(
        face.landmarks,
        face.blendshapeCategories,
      ),
    );
    state.selectedId = state.annotations[0]?.id || null;
    updateControls();
    draw();
    const labels = state.annotations
      .map((annotation) => EMOTION_LABELS[annotation.expression])
      .join("、");
    setStatus(
      "已识别 " +
        state.annotations.length +
        " 张脸 · " +
        labels +
        " · 分块检测" +
        (timedOut ? "（已达到时间上限）" : "") +
        " · Unicode Emoji",
    );
  } catch (error) {
    console.error(error);
    setStatus("识别服务加载失败，请检查网络后重试；手动贴图仍可用");
  } finally {
    setDetecting(false);
  }
}

function selectExpression(expression) {
  if (!UNICODE_EMOJI[expression]) {
    return;
  }

  const selected = getSelectedAnnotation();
  if (!selected) {
    setStatus("请先选中一个贴图，再修改它的表情");
    return;
  }

  selected.expression = expression;
  selected.emoji = UNICODE_EMOJI[expression];
  updateControls();
  draw();
  setStatus(
    imageStatusText() + " · 已将选中贴图改为" + EMOTION_LABELS[expression],
  );
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportEditedImage() {
  if (!state.image) {
    return;
  }

  const output = document.createElement("canvas");
  output.width = state.imageWidth;
  output.height = state.imageHeight;
  const outputCtx = output.getContext("2d");
  outputCtx.imageSmoothingEnabled = true;
  outputCtx.imageSmoothingQuality = "high";
  outputCtx.drawImage(state.image, 0, 0, state.imageWidth, state.imageHeight);
  state.annotations.forEach((annotation) => {
    drawAnnotation(outputCtx, annotation, 1, 0, 0);
  });
  const baseName = state.imageName.replace(/\.[^.]+$/, "") || "image";
  output.toBlob((blob) => {
    if (blob) {
      downloadBlob(blob, baseName + "-emoji.png");
      setStatus("已导出 " + state.imageWidth + " × " + state.imageHeight + "px");
      return;
    }
    const link = document.createElement("a");
    link.href = output.toDataURL("image/png");
    link.download = baseName + "-emoji.png";
    document.body.append(link);
    link.click();
    link.remove();
    setStatus("已导出 " + state.imageWidth + " × " + state.imageHeight + "px");
  }, "image/png");
}

imageInput.addEventListener("change", (event) => {
  loadImageFile(event.target.files?.[0]);
  event.target.value = "";
});

recognizeButton.addEventListener("click", recognizeFaces);
exportButton.addEventListener("click", exportEditedImage);
clearButton.addEventListener("click", () => {
  state.annotations = [];
  state.selectedId = null;
  updateControls();
  setStatus(imageStatusText());
  draw();
});

unicodeButtons.forEach((button) => {
  button.addEventListener("click", () =>
    selectExpression(button.dataset.expression),
  );
});

zoomOutButton.addEventListener("click", () => {
  zoomFromCenter(0.82);
  setStatus(imageStatusText());
});

zoomInButton.addEventListener("click", () => {
  zoomFromCenter(1.22);
  setStatus(imageStatusText());
});

resetViewButton.addEventListener("click", () => {
  resetView();
  setStatus(imageStatusText());
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
  if (!state.image) {
    setStatus("请先导入图片");
    return;
  }

  const point = getCanvasPoint(event);
  if (
    point.x >= 0 &&
    point.x <= state.displayWidth &&
    point.y >= 0 &&
    point.y <= state.displayHeight
  ) {
    addManualEmojiAt(point);
  }
});

canvas.addEventListener(
  "wheel",
  (event) => {
    if (!state.image) {
      return;
    }
    event.preventDefault();
    zoomAt(getCanvasPoint(event), Math.exp(-event.deltaY * 0.0015));
    setStatus(imageStatusText());
  },
  { passive: false },
);

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0 || !state.image) {
    return;
  }

  const point = getCanvasPoint(event);
  const resizeHit = hitSelectedBorder(point);
  const hit = resizeHit || hitAnnotation(point);
  if (!hit) {
    event.preventDefault();
    state.selectedId = null;
    startPan(point, event.pointerId);
    canvas.setPointerCapture(event.pointerId);
    updateControls();
    draw();
    return;
  }

  event.preventDefault();
  state.selectedId = hit.annotation.id;
  if (hit.kind === "resize") {
    startResize(hit.annotation, point, event.pointerId);
  } else {
    startMove(hit.annotation, point, event.pointerId);
  }
  canvas.setPointerCapture(event.pointerId);
  updateControls();
  draw();
});

canvas.addEventListener("pointermove", (event) => {
  const point = getCanvasPoint(event);
  if (state.drag) {
    event.preventDefault();
    updateDrag(point);
    return;
  }
  updateCursor(event);
});

canvas.addEventListener("pointerup", (event) => {
  if (state.drag?.pointerId === event.pointerId) {
    canvas.releasePointerCapture(event.pointerId);
    finishDrag();
  }
});

canvas.addEventListener("pointercancel", (event) => {
  if (state.drag?.pointerId === event.pointerId) {
    finishDrag();
  }
});

document.addEventListener("keydown", (event) => {
  if (
    !state.selectedId ||
    (event.key !== "Delete" && event.key !== "Backspace")
  ) {
    return;
  }

  const before = state.annotations.length;
  state.annotations = state.annotations.filter(
    (annotation) => annotation.id !== state.selectedId,
  );
  if (state.annotations.length !== before) {
    state.selectedId = null;
    updateControls();
    setStatus(imageStatusText());
    draw();
  }
});

if ("ResizeObserver" in window) {
  const resizeObserver = new ResizeObserver(fitCanvasToStage);
  resizeObserver.observe(stage);
} else {
  window.addEventListener("resize", fitCanvasToStage);
}

updateControls();
fitCanvasToStage();
