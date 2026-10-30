import {
  FaceDetector,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/vision_bundle.mjs";

const canvas = document.querySelector("#editorCanvas");
const ctx = canvas.getContext("2d");
const stage = document.querySelector("#canvasStage");
const imageInput = document.querySelector("#imageInput");
const recognizeButton = document.querySelector("#recognizeButton");
const recognizeLabel = document.querySelector("#recognizeLabel");
const processingModeSelect = document.querySelector("#processingMode");
const maskButtons = [...document.querySelectorAll("[data-mask-type]")];
const unicodePicker = document.querySelector("#unicodePicker");
const repairButton = document.querySelector("#repairButton");
const repairLabel = document.querySelector("#repairLabel");
const scanStats = document.querySelector("#scanStats");
const exportButton = document.querySelector("#exportButton");
const clearButton = document.querySelector("#clearButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const resetViewButton = document.querySelector("#resetViewButton");
const zoomInButton = document.querySelector("#zoomInButton");
const emptyState = document.querySelector("#emptyState");
const statusText = document.querySelector("#statusText");
const selectionHint = document.querySelector("#selectionHint");
const unicodeButtons = [...document.querySelectorAll("[data-expression]")];

const FACE_LANDMARKER_MODEL_URL = new URL(
  "./assets/models/face_landmarker.task",
  import.meta.url,
).href;
const FACE_DETECTOR_MODEL_URL = new URL(
  "./assets/models/blaze_face_full_range.tflite",
  import.meta.url,
).href;
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
const MASK_LABELS = {
  emoji: "Emoji",
  blur: "模糊",
  pixelate: "像素",
  solid: "纯色",
};
const DEFAULT_MASK_COLOR = "#171914";
const PROCESSING_MODE_LABELS = {
  "auto-expression": "自动表情",
  "all-happy": "全部开心",
  random: "随机表情",
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
const MAX_AUTO_FACES = 80;
const MAX_EXPRESSION_FACES = 6;
const MAX_DETECTION_DIMENSION = 2400;
const DETECTION_TILE_SCALES = [0.62, 0.4, 0.26, 0.17];
const DETECTION_TILE_OVERLAP = 0.5;
const DETECTION_TILE_RENDER_SIZE = 512;
const DETECTION_TIME_BUDGET_MS = 9000;
const DETECTOR_MIN_CONFIDENCE = 0.25;
const FULL_IMAGE_MIN_SCORE = 0.55;
const TILE_MIN_SCORE = 0.6;
const STRONG_SINGLE_SCORE = 0.85;
const MIN_DETECTED_FACE_SIZE = 10;
const FACE_DEDUPE_IOU = 0.35;
const FACE_DEDUPE_IOMIN = 0.7;
const EXPRESSION_CROP_SCALE = 2.2;
const EXPRESSION_CROP_SIZE = 256;

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
  processingMode: "auto-expression",
  maskType: "emoji",
  maskColor: DEFAULT_MASK_COLOR,
  repairMode: false,
  batchFiles: [],
  batchIndex: 0,
  batchTotal: 0,
  batchRunning: false,
  detectedFaces: 0,
  visionPromise: null,
  faceDetector: null,
  faceDetectorPromise: null,
  faceLandmarker: null,
  faceLandmarkerPromise: null,
};

function setStatus(text) {
  const hasBatchPrefix = text.startsWith("批量处理");
  const prefix = state.batchRunning && !hasBatchPrefix
    ? `批量处理 ${state.batchIndex}/${state.batchTotal} · `
    : "";
  statusText.textContent = prefix + text;
  if (text.includes("正在")) {
    statusText.dataset.state = "loading";
  } else if (text.includes("失败") || text.includes("不可用")) {
    statusText.dataset.state = "error";
  } else if (text.includes("未检测") || text.includes("请先")) {
    statusText.dataset.state = "warning";
  } else if (text.includes("已") || text.includes("检测到")) {
    statusText.dataset.state = "success";
  } else {
    statusText.dataset.state = "neutral";
  }
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
  const busy = state.detecting || state.batchRunning;
  imageInput.disabled = busy;
  exportButton.disabled = !hasImage || busy;
  clearButton.disabled = !hasImage || busy || state.annotations.length === 0;
  recognizeButton.disabled = !hasImage || busy;
  repairButton.disabled = !hasImage || busy;
  processingModeSelect.disabled = busy;
  zoomOutButton.disabled = !hasImage;
  resetViewButton.disabled = !hasImage;
  zoomInButton.disabled = !hasImage;
  emptyState.classList.toggle("hidden", hasImage);
  const selected = getSelectedAnnotation();
  scanStats.textContent = state.detectedFaces
    ? `检测到 ${state.detectedFaces} 张脸 · 当前 ${state.annotations.length} 个遮挡`
    : (state.annotations.length ? `当前 ${state.annotations.length} 个遮挡` : "尚未检测");
  selectionHint.textContent = selected
    ? `已选中“${EMOTION_LABELS[selected.expression] || "遮挡"}”贴图 · ${MASK_LABELS[selected.maskType || "emoji"]}`
    : `先选中贴图，或选择${MASK_LABELS[state.maskType]}作为新贴图默认效果`;
  const currentMaskType = selected?.maskType || state.maskType;
  maskButtons.forEach((button) => {
    const active = currentMaskType === button.dataset.maskType;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = busy;
  });
  unicodePicker.hidden = currentMaskType !== "emoji";
  unicodeButtons.forEach((button) => {
    const active = Boolean(selected && selected.expression === button.dataset.expression);
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = busy || !selected || currentMaskType !== "emoji";
  });
  repairButton.classList.toggle("is-active", state.repairMode);
  repairLabel.textContent = state.repairMode ? "完成补脸" : "补脸";
  recognizeLabel.textContent = state.detecting
    ? (state.batchRunning ? "批量处理中…" : "正在识别…")
    : (state.batchFiles.length > 1 ? "批量识别并下载" : "识别并贴图");
}

function setDetecting(isDetecting) {
  state.detecting = isDetecting;
  recognizeButton.classList.toggle("is-loading", isDetecting);
  recognizeLabel.textContent = isDetecting ? (state.batchRunning ? "批量处理中…" : "正在识别…") : "识别并贴图";
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

function drawAnnotation(targetCtx, annotation, scale, offsetX, offsetY, source = state.image) {
  const size = Math.max(1, annotation.size * scale);
  const x = (offsetX || 0) + annotation.x * scale;
  const y = (offsetY || 0) + annotation.y * scale;
  const maskType = annotation.maskType || "emoji";

  targetCtx.save();
  targetCtx.translate(x, y);
  targetCtx.rotate(annotation.rotation || 0);

  if (maskType === "emoji") {
    targetCtx.font =
      "normal " +
      size +
      'px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
    targetCtx.textAlign = "center";
    targetCtx.textBaseline = "middle";
    targetCtx.fillText(annotation.emoji || UNICODE_EMOJI.happy, 0, 0);
  } else {
    targetCtx.beginPath();
    targetCtx.ellipse(0, 0, size * 0.49, size * 0.49, 0, 0, Math.PI * 2);
    targetCtx.clip();
    if (maskType === "solid") {
      targetCtx.fillStyle = annotation.maskColor || DEFAULT_MASK_COLOR;
      targetCtx.fillRect(-size / 2, -size / 2, size, size);
    } else if (source) {
      const sourceSize = Math.max(1, annotation.size);
      const sourceX = annotation.x - sourceSize / 2;
      const sourceY = annotation.y - sourceSize / 2;
      if (maskType === "blur") {
        targetCtx.filter = `blur(${Math.max(5, size * 0.08)}px)`;
        targetCtx.drawImage(
          source,
          sourceX,
          sourceY,
          sourceSize,
          sourceSize,
          -size / 2,
          -size / 2,
          size,
          size,
        );
      } else {
        const pixelCanvas = document.createElement("canvas");
        pixelCanvas.width = 14;
        pixelCanvas.height = 14;
        const pixelContext = pixelCanvas.getContext("2d");
        pixelContext.imageSmoothingEnabled = false;
        pixelContext.drawImage(
          source,
          sourceX,
          sourceY,
          sourceSize,
          sourceSize,
          0,
          0,
          pixelCanvas.width,
          pixelCanvas.height,
        );
        targetCtx.imageSmoothingEnabled = false;
        targetCtx.drawImage(
          pixelCanvas,
          -size / 2,
          -size / 2,
          size,
          size,
        );
      }
    }
  }
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
  if (state.drag?.kind === "repair") {
    drawRepairSelection(state.drag.startPoint, state.drag.currentPoint);
  }
}

function drawRepairSelection(startPoint, currentPoint) {
  if (!startPoint || !currentPoint) {
    return;
  }
  const left = Math.min(startPoint.x, currentPoint.x);
  const top = Math.min(startPoint.y, currentPoint.y);
  const width = Math.abs(currentPoint.x - startPoint.x);
  const height = Math.abs(currentPoint.y - startPoint.y);
  ctx.save();
  ctx.fillStyle = "rgba(99, 91, 255, 0.14)";
  ctx.fillRect(left, top, width, height);
  ctx.strokeStyle = "#635bff";
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 5]);
  ctx.strokeRect(left, top, width, height);
  ctx.restore();
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

function loadImageFile(file, options = {}) {
  if (!file) {
    return Promise.reject(new Error("没有图片文件"));
  }
  if (!file.type.startsWith("image/")) {
    setStatus("请选择 JPG、PNG、WebP 等图片文件");
    return Promise.reject(new Error("不是图片文件"));
  }

  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      state.image = image;
      state.imageName = file.name || "image";
      state.imageFile = file;
      state.imageWidth = image.naturalWidth;
      state.imageHeight = image.naturalHeight;
      state.annotations = [];
      state.detectedFaces = 0;
      state.selectedId = null;
      state.lastEmojiSize = null;
      state.panX = 0;
      state.panY = 0;
      state.viewReady = false;
      state.repairMode = false;

      updateControls();
      fitCanvasToStage();
      if (!options.silent) {
        setStatus(imageStatusText() + " · 点击“识别并贴图”开始");
      }
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      setStatus("图片读取失败");
      reject(new Error("图片读取失败"));
    };
    image.src = url;
  });
}

function createAnnotation({
  expression = "happy",
  confidence = 1,
  x,
  y,
  size,
  rotation = 0,
  source = "manual",
  maskType = state.maskType,
  maskColor = state.maskColor,
}) {
  return {
    id: makeId(),
    expression,
    emoji: UNICODE_EMOJI[expression] || UNICODE_EMOJI.happy,
    confidence,
    x,
    y,
    size,
    rotation,
    source,
    maskType,
    maskColor,
  };
}

function addManualEmojiAt(point) {
  const imagePoint = canvasToImage(point);
  if (!isInsideImage(imagePoint)) {
    setStatus("请在图片范围内右键添加贴图");
    return;
  }

  const expression = "happy";
  const size = state.lastEmojiSize || DEFAULT_EMOJI_DISPLAY_SIZE / state.scale;
  const annotation = createAnnotation({
    expression,
    confidence: 1,
    x: imagePoint.x,
    y: imagePoint.y,
    size,
    source: "manual",
  });
  state.annotations.push(annotation);
  state.selectedId = annotation.id;
  updateControls();
  setStatus(imageStatusText());
  draw();
}

function addManualAnnotationFromRect(startPoint, endPoint) {
  const first = canvasToImage(startPoint);
  const second = canvasToImage(endPoint);
  if (!isInsideImage(first) && !isInsideImage(second)) {
    setStatus("请在图片范围内拖拽补脸区域");
    return;
  }
  const left = clamp(Math.min(first.x, second.x), 0, state.imageWidth);
  const top = clamp(Math.min(first.y, second.y), 0, state.imageHeight);
  const right = clamp(Math.max(first.x, second.x), 0, state.imageWidth);
  const bottom = clamp(Math.max(first.y, second.y), 0, state.imageHeight);
  const width = right - left;
  const height = bottom - top;
  if (Math.max(width, height) < 8 / state.scale) {
    setStatus("补脸区域太小，请拖动更大的矩形");
    return;
  }
  const size = Math.max(
    width,
    height,
    MIN_EMOJI_DISPLAY_SIZE / state.scale,
  );
  const annotation = createAnnotation({
    x: clamp((left + right) / 2, 0, state.imageWidth),
    y: clamp((top + bottom) / 2, 0, state.imageHeight),
    size,
    source: "manual",
  });
  state.annotations.push(annotation);
  state.selectedId = annotation.id;
  state.lastEmojiSize = size;
  updateControls();
  setStatus(imageStatusText() + " · 已补充 1 个遮挡");
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

function startRepair(point, pointerId) {
  state.drag = {
    kind: "repair",
    pointerId,
    startPoint: point,
    currentPoint: point,
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
  if (state.drag.kind === "repair") {
    state.drag.currentPoint = point;
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
  if (state.repairMode) {
    canvas.style.cursor = "crosshair";
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

  const drag = state.drag;
  if (drag.kind === "repair") {
    addManualAnnotationFromRect(drag.startPoint, drag.currentPoint);
    state.drag = null;
    draw();
    return;
  }
  const annotation = state.annotations.find((item) => item.id === drag.id);
  if (annotation && drag.kind === "resize") {
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

function buildDetectorAxisStarts(length, tileSize) {
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

function buildDetectorJobs(width, height) {
  const shortSide = Math.min(width, height);
  const jobs = [];

  DETECTION_TILE_SCALES.forEach((fraction) => {
    const tileSize = Math.round(shortSide * fraction);
    if (tileSize < 96 || tileSize >= Math.max(width, height)) {
      return;
    }
    const columns = buildDetectorAxisStarts(width, tileSize);
    const rows = buildDetectorAxisStarts(height, tileSize);
    rows.forEach((y) => {
      columns.forEach((x) => {
        jobs.push({ x, y, width: tileSize, height: tileSize });
      });
    });
  });

  return jobs;
}

function mapDetectorPoint(point, crop, sourceWidth, sourceHeight) {
  return {
    x: ((crop.x + point.x * crop.width) / sourceWidth) * state.imageWidth,
    y: ((crop.y + point.y * crop.height) / sourceHeight) * state.imageHeight,
  };
}

function hasPlausibleKeypoints(keypoints, box) {
  if (keypoints.length < 4) {
    return false;
  }
  const eyes = [keypoints[0], keypoints[1]].sort(
    (first, second) => first.x - second.x,
  );
  const nose = keypoints[2];
  const mouth = keypoints[3];
  const eyeDistance = Math.hypot(
    eyes[1].x - eyes[0].x,
    eyes[1].y - eyes[0].y,
  );
  const eyeMidpoint = midpoint(eyes[0], eyes[1]);
  const faceSize = Math.max(box.width, box.height);

  return (
    eyeDistance > faceSize * 0.12 &&
    eyeDistance < faceSize * 0.9 &&
    Math.hypot(mouth.x - eyeMidpoint.x, mouth.y - eyeMidpoint.y) >
      eyeDistance * 0.4 &&
    Math.hypot(nose.x - eyeMidpoint.x, nose.y - eyeMidpoint.y) <
      eyeDistance * 1.5
  );
}

function collectDetectorCandidates(
  result,
  crop,
  sourceWidth,
  sourceHeight,
  isFullImage,
) {
  const scaleX = state.imageWidth / sourceWidth;
  const scaleY = state.imageHeight / sourceHeight;
  const minimumScore = isFullImage
    ? FULL_IMAGE_MIN_SCORE
    : TILE_MIN_SCORE;

  return (result.detections || [])
    .map((detection) => {
      const score = Number(detection.categories?.[0]?.score || 0);
      const sourceBox = detection.boundingBox;
      if (!sourceBox || score < minimumScore) {
        return null;
      }

      const box = {
        x: (crop.x + sourceBox.originX * (crop.width / crop.inputWidth)) * scaleX,
        y: (crop.y + sourceBox.originY * (crop.height / crop.inputHeight)) * scaleY,
        width: sourceBox.width * (crop.width / crop.inputWidth) * scaleX,
        height: sourceBox.height * (crop.height / crop.inputHeight) * scaleY,
      };
      if (Math.min(box.width, box.height) < MIN_DETECTED_FACE_SIZE) {
        return null;
      }

      const keypoints = (detection.keypoints || []).map((point) =>
        mapDetectorPoint(point, crop, sourceWidth, sourceHeight),
      );
      if (!hasPlausibleKeypoints(keypoints, box)) {
        return null;
      }

      const eyes = [keypoints[0], keypoints[1]].sort(
        (first, second) => first.x - second.x,
      );
      return {
        box,
        score,
        hits: 1,
        fullHit: isFullImage,
        eyeLeft: eyes[0],
        eyeRight: eyes[1],
      };
    })
    .filter(Boolean);
}

function boxArea(box) {
  return Math.max(0, box.width) * Math.max(0, box.height);
}

function boxIntersection(first, second) {
  const intersectionLeft = Math.max(first.x, second.x);
  const intersectionTop = Math.max(first.y, second.y);
  const intersectionRight = Math.min(
    first.x + first.width,
    second.x + second.width,
  );
  const intersectionBottom = Math.min(
    first.y + first.height,
    second.y + second.height,
  );
  const intersectionWidth = Math.max(0, intersectionRight - intersectionLeft);
  const intersectionHeight = Math.max(0, intersectionBottom - intersectionTop);
  return intersectionWidth * intersectionHeight;
}

function isSameDetectorFace(first, second) {
  const intersection = boxIntersection(first.box, second.box);
  const firstArea = boxArea(first.box);
  const secondArea = boxArea(second.box);
  const union = firstArea + secondArea - intersection;
  const iou = union > 0 ? intersection / union : 0;
  const ioMinimum =
    Math.min(firstArea, secondArea) > 0
      ? intersection / Math.min(firstArea, secondArea)
      : 0;
  return iou > FACE_DEDUPE_IOU || ioMinimum > FACE_DEDUPE_IOMIN;
}

function mergeDetectorFaces(candidates) {
  const merged = candidates.map((candidate) => ({
    ...candidate,
    box: { ...candidate.box },
  }));
  let changed = true;

  while (changed) {
    changed = false;
    outer: for (let firstIndex = 0; firstIndex < merged.length; firstIndex += 1) {
      for (
        let secondIndex = firstIndex + 1;
        secondIndex < merged.length;
        secondIndex += 1
      ) {
        const first = merged[firstIndex];
        const second = merged[secondIndex];
        if (!isSameDetectorFace(first, second)) {
          continue;
        }

        const left = Math.min(first.box.x, second.box.x);
        const top = Math.min(first.box.y, second.box.y);
        const right = Math.max(
          first.box.x + first.box.width,
          second.box.x + second.box.width,
        );
        const bottom = Math.max(
          first.box.y + first.box.height,
          second.box.y + second.box.height,
        );
        const best = first.score >= second.score ? first : second;
        merged[firstIndex] = {
          ...best,
          box: {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
          },
          score: Math.max(first.score, second.score),
          hits: first.hits + second.hits,
          fullHit: first.fullHit || second.fullHit,
        };
        merged.splice(secondIndex, 1);
        changed = true;
        break outer;
      }
    }
  }

  return merged
    .filter(
      (face) =>
        face.fullHit || face.score >= STRONG_SINGLE_SCORE || face.hits >= 2,
    )
    .sort((first, second) => boxArea(second.box) - boxArea(first.box))
    .slice(0, MAX_AUTO_FACES);
}

function createDetectorTile(sourceCanvas, tile) {
  const tileCanvas = document.createElement("canvas");
  tileCanvas.width = DETECTION_TILE_RENDER_SIZE;
  tileCanvas.height = DETECTION_TILE_RENDER_SIZE;
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
    DETECTION_TILE_RENDER_SIZE,
    DETECTION_TILE_RENDER_SIZE,
  );
  return tileCanvas;
}

function drawReplicatedExpressionCrop(targetContext, crop) {
  const source = state.image;
  const size = EXPRESSION_CROP_SIZE;
  const scale = size / crop.side;
  const left = clamp(crop.x, 0, state.imageWidth);
  const top = clamp(crop.y, 0, state.imageHeight);
  const right = clamp(crop.x + crop.side, 0, state.imageWidth);
  const bottom = clamp(crop.y + crop.side, 0, state.imageHeight);
  const sourceWidth = Math.max(1, right - left);
  const sourceHeight = Math.max(1, bottom - top);
  const targetX = (left - crop.x) * scale;
  const targetY = (top - crop.y) * scale;
  const targetWidth = sourceWidth * scale;
  const targetHeight = sourceHeight * scale;

  targetContext.fillStyle = "#808080";
  targetContext.fillRect(0, 0, size, size);
  targetContext.drawImage(
    source,
    left,
    top,
    sourceWidth,
    sourceHeight,
    targetX,
    targetY,
    targetWidth,
    targetHeight,
  );

  if (targetX > 0) {
    targetContext.drawImage(
      source,
      left,
      top,
      1,
      sourceHeight,
      0,
      targetY,
      targetX,
      targetHeight,
    );
  }
  if (targetX + targetWidth < size) {
    targetContext.drawImage(
      source,
      Math.max(0, right - 1),
      top,
      1,
      sourceHeight,
      targetX + targetWidth,
      targetY,
      size - targetX - targetWidth,
      targetHeight,
    );
  }
  if (targetY > 0) {
    targetContext.drawImage(
      source,
      left,
      top,
      sourceWidth,
      1,
      targetX,
      0,
      targetWidth,
      targetY,
    );
  }
  if (targetY + targetHeight < size) {
    targetContext.drawImage(
      source,
      left,
      Math.max(0, bottom - 1),
      sourceWidth,
      1,
      targetX,
      targetY + targetHeight,
      targetWidth,
      size - targetY - targetHeight,
    );
  }
  if (targetX > 0 && targetY > 0) {
    targetContext.drawImage(source, left, top, 1, 1, 0, 0, targetX, targetY);
  }
  if (targetX + targetWidth < size && targetY > 0) {
    targetContext.drawImage(
      source,
      Math.max(0, right - 1),
      top,
      1,
      1,
      targetX + targetWidth,
      0,
      size - targetX - targetWidth,
      targetY,
    );
  }
  if (targetX > 0 && targetY + targetHeight < size) {
    targetContext.drawImage(
      source,
      left,
      Math.max(0, bottom - 1),
      1,
      1,
      0,
      targetY + targetHeight,
      targetX,
      size - targetY - targetHeight,
    );
  }
  if (
    targetX + targetWidth < size &&
    targetY + targetHeight < size
  ) {
    targetContext.drawImage(
      source,
      Math.max(0, right - 1),
      Math.max(0, bottom - 1),
      1,
      1,
      targetX + targetWidth,
      targetY + targetHeight,
      size - targetX - targetWidth,
      size - targetY - targetHeight,
    );
  }
}

function createExpressionCrop(face) {
  const centerX = face.box.x + face.box.width / 2;
  const centerY = face.box.y + face.box.height / 2;
  const side = Math.max(face.box.width, face.box.height) * EXPRESSION_CROP_SCALE;
  const crop = {
    x: centerX - side / 2,
    y: centerY - side / 2,
    side,
  };
  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = EXPRESSION_CROP_SIZE;
  cropCanvas.height = EXPRESSION_CROP_SIZE;
  const cropContext = cropCanvas.getContext("2d");
  cropContext.imageSmoothingEnabled = true;
  cropContext.imageSmoothingQuality = "high";
  drawReplicatedExpressionCrop(cropContext, crop);
  return { canvas: cropCanvas, crop };
}

function nextFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(resolve));
}

function detectorFaceToAnnotation(face) {
  const rotation =
    face.eyeLeft && face.eyeRight
      ? Math.atan2(
          face.eyeRight.y - face.eyeLeft.y,
          face.eyeRight.x - face.eyeLeft.x,
        )
      : 0;
  return createAnnotation({
    expression: "happy",
    confidence: face.score,
    x: face.box.x + face.box.width / 2,
    y: face.box.y + face.box.height / 2 + face.box.height * 0.025,
    size: clamp(
      Math.max(face.box.width * 1.55, face.box.height * 1.4),
      36,
      Math.max(state.imageWidth, state.imageHeight) * 0.6,
    ),
    rotation: clamp(rotation, -0.21, 0.21),
    source: "auto",
  });
}

function selectExpressionLandmarks(result) {
  return (result.faceLandmarks || [])
    .map((landmarks, index) => ({
      landmarks,
      categories: result.faceBlendshapes?.[index]?.categories || [],
      bounds: getLandmarkBounds(landmarks, 1, 1),
    }))
    .filter(
      (face) =>
        Math.abs(face.bounds.centerX - 0.5) < 0.2 &&
        Math.abs(face.bounds.centerY - 0.5) < 0.22 &&
        face.bounds.width > 0.2 &&
        face.bounds.width < 0.85,
    )
    .sort((first, second) => second.bounds.area - first.bounds.area)[0];
}

function mapCropLandmarksToImage(landmarks, crop) {
  return landmarks.map((landmark) => ({
    ...landmark,
    x: clamp((crop.x + landmark.x * crop.side) / state.imageWidth, 0, 1),
    y: clamp((crop.y + landmark.y * crop.side) / state.imageHeight, 0, 1),
  }));
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

  return createAnnotation({
    expression: expressionResult.expression,
    confidence: expressionResult.confidence,
    x: bounds.centerX,
    y: bounds.centerY + height * 0.035,
    size: clamp(
      Math.max(width * 1.9, height * 1.85),
      36,
      Math.max(state.imageWidth, state.imageHeight) * 0.6,
    ),
    rotation: clamp(rotation, -0.21, 0.21),
    source: "auto",
  });
}

function ensureVisionFileset() {
  if (!state.visionPromise) {
    state.visionPromise = FilesetResolver.forVisionTasks(WASM_URL).catch(
      (error) => {
        state.visionPromise = null;
        throw error;
      },
    );
  }
  return state.visionPromise;
}

async function ensureFaceDetector() {
  if (state.faceDetector) {
    return state.faceDetector;
  }
  if (state.faceDetectorPromise) {
    return state.faceDetectorPromise;
  }

  state.faceDetectorPromise = (async () => {
    setStatus("正在加载人脸检测模型…");
    const vision = await ensureVisionFileset();
    const detector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_DETECTOR_MODEL_URL,
      },
      runningMode: "IMAGE",
      minDetectionConfidence: DETECTOR_MIN_CONFIDENCE,
      minSuppressionThreshold: 0.3,
    });
    state.faceDetector = detector;
    return detector;
  })().catch((error) => {
    state.faceDetectorPromise = null;
    throw error;
  });

  return state.faceDetectorPromise;
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
    const vision = await ensureVisionFileset();
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_LANDMARKER_MODEL_URL,
      },
      runningMode: "IMAGE",
      numFaces: 1,
      minFaceDetectionConfidence: 0.35,
      minFacePresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    state.faceLandmarker = landmarker;
    return landmarker;
  })().catch((error) => {
    state.faceLandmarkerPromise = null;
    throw error;
  });

  return state.faceLandmarkerPromise;
}

async function detectFacesWithDetector(detector) {
  const detectionCanvas = createDetectionCanvas();
  const sourceWidth = detectionCanvas.width;
  const sourceHeight = detectionCanvas.height;
  const candidates = [];
  const startedAt = performance.now();
  const fullCrop = {
    x: 0,
    y: 0,
    width: sourceWidth,
    height: sourceHeight,
    inputWidth: sourceWidth,
    inputHeight: sourceHeight,
  };

  setStatus("正在检测整张图片…");
  await nextFrame();
  candidates.push(
    ...collectDetectorCandidates(
      detector.detect(detectionCanvas),
      fullCrop,
      sourceWidth,
      sourceHeight,
      true,
    ),
  );

  const jobs = buildDetectorJobs(sourceWidth, sourceHeight);
  let timedOut = false;
  for (let index = 0; index < jobs.length; index += 1) {
    if (performance.now() - startedAt >= DETECTION_TIME_BUDGET_MS) {
      timedOut = true;
      break;
    }

    if (index % 8 === 0) {
      setStatus("正在检测人脸 " + (index + 1) + "/" + jobs.length + "…");
      await nextFrame();
    }
    const job = jobs[index];
    const tileCanvas = createDetectorTile(detectionCanvas, job);
    candidates.push(
      ...collectDetectorCandidates(
        detector.detect(tileCanvas),
        {
          ...job,
          inputWidth: DETECTION_TILE_RENDER_SIZE,
          inputHeight: DETECTION_TILE_RENDER_SIZE,
        },
        sourceWidth,
        sourceHeight,
        false,
      ),
    );
  }

  return {
    faces: mergeDetectorFaces(candidates),
    startedAt,
    timedOut,
  };
}

async function fallbackToWholeImageLandmarker() {
  setStatus("人脸检测器不可用，正在使用兼容模式…");
  const landmarker = await ensureFaceLandmarker();
  await landmarker.setOptions({ numFaces: MAX_EXPRESSION_FACES });
  try {
    const detectionCanvas = createDetectionCanvas();
    const result = landmarker.detect(detectionCanvas);
    return (result.faceLandmarks || [])
      .slice(0, MAX_EXPRESSION_FACES)
      .map((landmarks, index) =>
        faceToAnnotation(
          landmarks,
          result.faceBlendshapes?.[index]?.categories || [],
        ),
      );
  } finally {
    await landmarker.setOptions({ numFaces: 1 });
  }
}

async function refineExpressionAnnotations(
  faces,
  annotations,
  landmarker,
  startedAt,
) {
  const count = Math.min(faces.length, MAX_EXPRESSION_FACES);
  let recognized = 0;
  let timedOut = false;

  for (let index = 0; index < count; index += 1) {
    if (performance.now() - startedAt >= DETECTION_TIME_BUDGET_MS) {
      timedOut = true;
      break;
    }

    setStatus("正在识别表情 " + (index + 1) + "/" + count + "…");
    await nextFrame();
    try {
      const { canvas: cropCanvas, crop } = createExpressionCrop(faces[index]);
      const result = landmarker.detect(cropCanvas);
      const selectedFace = selectExpressionLandmarks(result);
      if (!selectedFace) {
        continue;
      }
      const globalLandmarks = mapCropLandmarksToImage(
        selectedFace.landmarks,
        crop,
      );
      const refined = faceToAnnotation(
        globalLandmarks,
        selectedFace.categories,
      );
      const originalId = annotations[index].id;
      Object.assign(annotations[index], refined, {
        id: originalId,
        source: "auto",
        maskType: annotations[index].maskType || "emoji",
        maskColor: annotations[index].maskColor || DEFAULT_MASK_COLOR,
      });
      recognized += 1;
      draw();
    } catch (error) {
      console.warn("单张人脸表情识别失败", error);
    }
  }

  return { recognized, timedOut };
}

function applyRandomExpressions(annotations) {
  const expressions = Object.keys(UNICODE_EMOJI);
  annotations.forEach((annotation) => {
    const expression = expressions[Math.floor(Math.random() * expressions.length)];
    annotation.expression = expression;
    annotation.emoji = UNICODE_EMOJI[expression];
    annotation.confidence = 1;
  });
}

async function recognizeFaces(options = {}) {
  const internal = options.internal === true;
  if (!state.image || (state.detecting && !internal)) {
    return;
  }

  if (!internal) {
    state.repairMode = false;
    setDetecting(true);
  }
  setStatus("正在识别人脸和表情…");
  try {
    let detectorResult;
    try {
      const detector = await ensureFaceDetector();
      detectorResult = await detectFacesWithDetector(detector);
    } catch (detectorError) {
      console.warn("人脸检测器不可用，尝试兼容模式", detectorError);
      const fallbackAnnotations = await fallbackToWholeImageLandmarker();
      if (state.processingMode === "all-happy") {
        fallbackAnnotations.forEach((annotation) => {
          annotation.expression = "happy";
          annotation.emoji = UNICODE_EMOJI.happy;
        });
      } else if (state.processingMode === "random") {
        applyRandomExpressions(fallbackAnnotations);
      }
      state.annotations = fallbackAnnotations;
      state.detectedFaces = fallbackAnnotations.length;
      state.selectedId = fallbackAnnotations[0]?.id || null;
      updateControls();
      draw();
      setStatus(
        fallbackAnnotations.length
          ? "已回退识别 " + fallbackAnnotations.length + " 张脸 · Unicode Emoji"
          : "未检测到可识别的人脸",
      );
      return;
    }

    const faces = detectorResult.faces;
    if (!faces.length) {
      state.annotations = [];
      state.detectedFaces = 0;
      state.selectedId = null;
      updateControls();
      draw();
      setStatus("未检测到可识别的人脸");
      return;
    }

    state.annotations = faces.map(detectorFaceToAnnotation);
    state.detectedFaces = faces.length;
    state.selectedId = state.annotations[0]?.id || null;
    updateControls();
    draw();
    await nextFrame();

    let expressionResult = { recognized: 0, timedOut: false };
    let landmarkerFailed = false;
    if (state.processingMode === "random") {
      applyRandomExpressions(state.annotations);
      expressionResult.recognized = state.annotations.length;
    } else if (
      state.processingMode === "auto-expression" &&
      performance.now() - detectorResult.startedAt <
      DETECTION_TIME_BUDGET_MS
    ) {
      try {
        const landmarker = await ensureFaceLandmarker();
        expressionResult = await refineExpressionAnnotations(
          faces,
          state.annotations,
          landmarker,
          detectorResult.startedAt,
        );
      } catch (landmarkerError) {
        landmarkerFailed = true;
        console.warn("表情识别模型不可用，保留开心贴图", landmarkerError);
      }
    } else if (state.processingMode === "auto-expression") {
      expressionResult.timedOut = true;
    }

    updateControls();
    draw();
    const defaultHappyCount = faces.length - expressionResult.recognized;
    const modeText = PROCESSING_MODE_LABELS[state.processingMode];
    setStatus(
      "检测到 " +
        faces.length +
        " 张脸 · 表情识别 " +
        expressionResult.recognized +
        " 张 · 其余 " +
        defaultHappyCount +
        " 张默认开心" +
        (detectorResult.timedOut || expressionResult.timedOut
          ? "（已达到时间上限）"
          : "") +
        (landmarkerFailed ? "（表情模型不可用）" : "") +
        " · " + modeText + " · Unicode Emoji",
    );
  } catch (error) {
    console.error(error);
    setStatus("识别服务加载失败，请检查网络后重试；手动贴图仍可用");
  } finally {
    if (!internal) {
      setDetecting(false);
    }
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
  selected.maskType = "emoji";
  state.maskType = "emoji";
  updateControls();
  draw();
  setStatus(
    imageStatusText() + " · 已将选中贴图改为" + EMOTION_LABELS[expression],
  );
}

function selectMaskType(maskType) {
  if (!MASK_LABELS[maskType]) {
    return;
  }
  state.maskType = maskType;
  const selected = getSelectedAnnotation();
  if (selected) {
    selected.maskType = maskType;
    selected.maskColor = state.maskColor;
  }
  updateControls();
  draw();
  setStatus(
    imageStatusText() +
      (selected ? " · 当前贴图已切换为" : " · 新贴图默认效果：") +
      MASK_LABELS[maskType],
  );
}

function setRepairMode(enabled) {
  state.repairMode = Boolean(enabled) && Boolean(state.image);
  if (state.repairMode) {
    state.selectedId = null;
    setStatus("补脸模式已开启 · 在图片上拖拽矩形区域");
  } else {
    setStatus(imageStatusText());
  }
  updateControls();
  draw();
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

async function exportEditedImage(options = {}) {
  if (!state.image) {
    return false;
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
  const baseName = (options.fileName || state.imageName).replace(/\.[^.]+$/, "") || "image";
  const fileName = baseName.endsWith("-emoji") ? baseName + ".png" : baseName + "-emoji.png";
  const blob = await new Promise((resolve) => output.toBlob(resolve, "image/png"));
  if (blob) {
    downloadBlob(blob, fileName);
  } else {
    const link = document.createElement("a");
    link.href = output.toDataURL("image/png");
    link.download = fileName;
    document.body.append(link);
    link.click();
    link.remove();
  }
  if (!options.silent) {
    setStatus("已导出 " + state.imageWidth + " × " + state.imageHeight + "px");
  }
  return true;
}

function handleInputFiles(files) {
  const imageFiles = [...files].filter((file) => file.type.startsWith("image/"));
  if (!imageFiles.length) {
    setStatus("请选择 JPG、PNG、WebP 等图片文件");
    return;
  }
  state.batchFiles = imageFiles;
  state.batchIndex = 0;
  state.batchTotal = imageFiles.length;
  loadImageFile(imageFiles[0], { silent: true })
    .then(() => {
      updateControls();
      setStatus(
        imageStatusText() +
          (imageFiles.length > 1
            ? ` · 已选择 ${imageFiles.length} 张照片，点击“批量识别并下载”`
            : " · 点击“识别并贴图”开始"),
      );
    })
    .catch(() => {
      state.batchFiles = [];
    });
}

async function processBatchFiles(files) {
  if (state.batchRunning || !files.length) {
    return;
  }
  state.batchRunning = true;
  state.batchTotal = files.length;
  state.batchIndex = 0;
  state.repairMode = false;
  setDetecting(true);
  let success = 0;
  const failures = [];
  const fileNameCounts = new Map();
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      state.batchIndex = index + 1;
      const previousCount = fileNameCounts.get(file.name) || 0;
      fileNameCounts.set(file.name, previousCount + 1);
      const fileName = previousCount
        ? file.name.replace(/(\.[^.]+)$/, `-${previousCount + 1}$1`)
        : file.name;
      setStatus(`正在处理 ${index + 1}/${files.length}：${file.name}`);
      try {
        await loadImageFile(file, { silent: true });
        await recognizeFaces({ internal: true });
        await exportEditedImage({ silent: true, fileName });
        success += 1;
      } catch (error) {
        console.error("批量处理失败", file.name, error);
        failures.push(file.name);
      }
    }
  } finally {
    state.batchRunning = false;
    setDetecting(false);
  }
  setStatus(
    `批量完成 · 成功 ${success}/${files.length} 张` +
      (failures.length ? ` · 失败：${failures.join("、")}` : "") +
      " · 每张已单独下载",
  );
}

imageInput.addEventListener("change", (event) => {
  handleInputFiles(event.target.files || []);
  event.target.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  stage.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = "copy";
    }
    stage.classList.add("is-dragover");
  });
}

stage.addEventListener("dragleave", (event) => {
  if (!stage.contains(event.relatedTarget)) {
    stage.classList.remove("is-dragover");
  }
});

stage.addEventListener("drop", (event) => {
  event.preventDefault();
  stage.classList.remove("is-dragover");
  handleInputFiles(event.dataTransfer?.files || []);
});

recognizeButton.addEventListener("click", () => {
  if (state.batchFiles.length > 1) {
    processBatchFiles(state.batchFiles);
  } else {
    recognizeFaces();
  }
});
exportButton.addEventListener("click", exportEditedImage);
repairButton.addEventListener("click", () => setRepairMode(!state.repairMode));
processingModeSelect.addEventListener("change", (event) => {
  state.processingMode = event.target.value;
  setStatus(imageStatusText() + " · 已选择" + PROCESSING_MODE_LABELS[state.processingMode]);
});
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

maskButtons.forEach((button) => {
  button.addEventListener("click", () => selectMaskType(button.dataset.maskType));
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
  if (state.repairMode && isInsideImage(canvasToImage(point))) {
    event.preventDefault();
    startRepair(point, event.pointerId);
    canvas.setPointerCapture(event.pointerId);
    draw();
    return;
  }
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
setStatus("未导入图片");
fitCanvasToStage();
