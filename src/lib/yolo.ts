import type { InferenceSession, Tensor } from "onnxruntime-web";
import * as ort from "onnxruntime-web";

// Target classes to detect: COCO class 0 = person, class 1 = face
const TARGET_CLASSES: Record<number, string> = { 0: "Person", 1: "Face" };

const MAX_DETECTIONS = 30;

export type ExecutionProvider = "webgpu" | "wasm";

export interface ModelLoadResult {
  session: InferenceSession;
  ep: ExecutionProvider;
}

export interface Detection {
  /** Normalized [0, 1] coordinates */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  label: string;
}

// Reusable buffers
let preprocessBuffer: Float32Array | null = null;
let preprocessCanvas: HTMLCanvasElement | null = null;
let preprocessCtx: CanvasRenderingContext2D | null = null;
let outputDimsLogged = false;

// Model loading
export async function initModel(
  modelPath: string,
  preferredBackend?: "webgpu" | "wasm",
): Promise<{
  session: ort.InferenceSession;
  ep: ExecutionProvider;
}> {
  const providers: string[] = [];

  if (preferredBackend !== "wasm" && "gpu" in navigator) {
    providers.push("webgpu");
  }
  providers.push("wasm");

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: providers,
    graphOptimizationLevel: "all",
  });

  for (const name of session.outputNames) {
    console.log(`Output "${name}"`);
  }

  const ep: ExecutionProvider = providers[0] === "webgpu" ? "webgpu" : "wasm";

  return { session, ep };
}

function getModelInputSize(session: InferenceSession): [number, number] {
  const meta = session.inputMetadata[0];
  if (meta.isTensor) {
    const { shape } = meta;
    const w = typeof shape[3] === "number" ? shape[3] : 640;
    const h = typeof shape[2] === "number" ? shape[2] : 640;
    return [w, h];
  }
  return [640, 640];
}

function preprocessFrame(
  video: HTMLVideoElement,
  modelWidth: number,
  modelHeight: number,
): Tensor {
  if (
    !preprocessCanvas ||
    preprocessCanvas.width !== modelWidth ||
    preprocessCanvas.height !== modelHeight
  ) {
    preprocessCanvas = document.createElement("canvas");
    preprocessCanvas.width = modelWidth;
    preprocessCanvas.height = modelHeight;
    preprocessCtx = preprocessCanvas.getContext("2d");
  }
  const ctx = preprocessCtx;
  if (!ctx) throw new Error("Canvas 2D context unavailable");

  const scale = Math.min(
    modelWidth / video.videoWidth,
    modelHeight / video.videoHeight,
  );
  const scaledW = video.videoWidth * scale;
  const scaledH = video.videoHeight * scale;
  const offsetX = (modelWidth - scaledW) / 2;
  const offsetY = (modelHeight - scaledH) / 2;

  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, modelWidth, modelHeight);
  ctx.drawImage(video, offsetX, offsetY, scaledW, scaledH);

  const { data } = ctx.getImageData(0, 0, modelWidth, modelHeight);
  const pixels = modelWidth * modelHeight;

  if (!preprocessBuffer || preprocessBuffer.length !== 3 * pixels) {
    preprocessBuffer = new Float32Array(3 * pixels);
  }

  for (let i = 0; i < pixels; i++) {
    preprocessBuffer[i] = data[i * 4] / 255; // R
    preprocessBuffer[i + pixels] = data[i * 4 + 1] / 255; // G
    preprocessBuffer[i + 2 * pixels] = data[i * 4 + 2] / 255; // B
  }

  return new ort.Tensor("float32", preprocessBuffer, [
    1,
    3,
    modelHeight,
    modelWidth,
  ]);
}

function postprocessEndToEnd(
  data: Float32Array,
  numBoxes: number,
  confThreshold: number,
  modelWidth: number,
  modelHeight: number,
): Detection[] {
  const results: Detection[] = [];

  for (let i = 0; i < numBoxes; i++) {
    if (results.length >= MAX_DETECTIONS) break;

    const offset = i * 6;
    const confidence = data[offset + 4];
    if (confidence <= 0 || confidence < confThreshold) continue;

    const classId = Math.round(data[offset + 5]);
    if (!(classId in TARGET_CLASSES)) continue;

    results.push({
      x1: data[offset] / modelWidth,
      y1: data[offset + 1] / modelHeight,
      x2: data[offset + 2] / modelWidth,
      y2: data[offset + 3] / modelHeight,
      score: confidence,
      classId,
      label: TARGET_CLASSES[classId],
    });
  }

  return results;
}

export async function detectPersons(
  session: InferenceSession,
  video: HTMLVideoElement,
  confThreshold: number,
  _iouThreshold: number,
): Promise<Detection[]> {
  const [modelWidth, modelHeight] = getModelInputSize(session);
  const inputTensor = preprocessFrame(video, modelWidth, modelHeight);

  const results = await session.run({
    [session.inputNames[0]]: inputTensor,
  });

  const output = results[session.outputNames[0]] as Tensor;
  const data = output.data as Float32Array;
  const dims = output.dims;

  if (!outputDimsLogged) {
    console.log("Output dims:", dims.join(", "));
    outputDimsLogged = true;
  }

  const [, dim1] = dims;
  return postprocessEndToEnd(
    data,
    dim1,
    confThreshold,
    modelWidth,
    modelHeight,
  );
}
