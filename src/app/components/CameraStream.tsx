"use client";

import type { InferenceSession } from "onnxruntime-web";
import { useCallback, useEffect, useRef, useState } from "react";
import { prepareVideoStream } from "@/lib/camera";
import {
  type Detection,
  detectPersons,
  type ExecutionProvider,
  initModel,
} from "@/lib/yolo";

const MODEL_PATH = "/models/yolo26n.onnx";
const CONF_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;

function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const det of detections) {
    const x = det.x1 * canvas.width;
    const y = det.y1 * canvas.height;
    const w = (det.x2 - det.x1) * canvas.width;
    const h = (det.y2 - det.y1) * canvas.height;

    // Person → green
    // Face → blue
    const color = det.classId === 0 ? "#22c55e" : "#3b82f6";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
    ctx.font = "bold 13px sans-serif";
    const textY = y > 22 ? y - 6 : y + h + 16;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x, textY - 14, ctx.measureText(label).width + 6, 18);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 3, textY);
  }
}

export default function CameraStream() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<InferenceSession | null>(null);
  const rafRef = useRef<number>(0);
  const isRunningRef = useRef(false);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [ep, setEp] = useState<ExecutionProvider | null>(null);

  const runInferenceLoop = useCallback(async () => {
    if (!isRunningRef.current) return;

    const video = videoRef.current;
    const session = sessionRef.current;

    if (
      !video ||
      !session ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      rafRef.current = requestAnimationFrame(runInferenceLoop);
      return;
    }

    try {
      const detections = await detectPersons(
        session,
        video,
        CONF_THRESHOLD,
        IOU_THRESHOLD,
      );
      if (isRunningRef.current && canvasRef.current) {
        drawDetections(canvasRef.current, detections);
      }
    } catch {
      // continue loop even on transient inference error
    }

    if (isRunningRef.current) {
      rafRef.current = requestAnimationFrame(runInferenceLoop);
    }
  }, []);

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    setError(null);
    setIsLoading(true);

    try {
      const [mediaStream, { session, ep: activeEp }] = await Promise.all([
        prepareVideoStream(videoRef.current),
        initModel(MODEL_PATH),
      ]);

      sessionRef.current = session;
      isRunningRef.current = true;
      setEp(activeEp);
      setStream(mediaStream);
      setIsLoading(false);
      runInferenceLoop();
    } catch (err) {
      setError(err instanceof Error ? err.message : "初期化に失敗しました");
      setIsLoading(false);
    }
  }, [runInferenceLoop]);

  const stopCamera = useCallback(() => {
    isRunningRef.current = false;
    cancelAnimationFrame(rafRef.current);

    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      setStream(null);
    }

    sessionRef.current?.release().catch(() => {});
    sessionRef.current = null;
    setEp(null);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, [stream]);

  useEffect(() => {
    return () => {
      isRunningRef.current = false;
      cancelAnimationFrame(rafRef.current);
      sessionRef.current?.release().catch(() => {});
    };
  }, []);

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-2xl">
      <div className="relative w-full aspect-video bg-zinc-900 rounded-xl overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />
        {!stream && (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-500 text-sm" />
        )}
        {ep && (
          <span
            className={`absolute top-3 right-3 text-xs font-semibold px-2 py-1 rounded-full ${
              ep === "webgpu"
                ? "bg-purple-600/80 text-white"
                : "bg-yellow-500/80 text-black"
            }`}
          >
            {ep === "webgpu" ? "WebGPU" : "WASM"}
          </span>
        )}
      </div>

      {error && (
        <p className="text-red-500 text-sm bg-red-50 dark:bg-red-950 px-4 py-2 rounded-lg w-full text-center">
          {error}
        </p>
      )}

      {!stream ? (
        <button
          type="button"
          onClick={startCamera}
          disabled={isLoading}
          className="px-8 py-3 bg-blue-600 text-white font-medium rounded-full hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? "読み込み中..." : "カメラ開始"}
        </button>
      ) : (
        <button
          type="button"
          onClick={stopCamera}
          className="px-8 py-3 bg-red-600 text-white font-medium rounded-full hover:bg-red-700 transition-colors"
        >
          カメラ停止
        </button>
      )}
    </div>
  );
}
