"use client";

import { useCallback, useRef, useState } from "react";
import { prepareVideoStream } from "@/lib/camera";

export default function CameraStream() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    setError(null);
    try {
      const mediaStream = await prepareVideoStream(videoRef.current);
      setStream(mediaStream);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "カメラへのアクセスに失敗しました",
      );
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      setStream(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  }, [stream]);

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
        {!stream && (
          <div className="absolute inset-0 flex items-center justify-center text-zinc-500" />
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
          className="px-8 py-3 bg-blue-600 text-white font-medium rounded-full hover:bg-blue-700 transition-colors"
        >
          カメラ開始
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
