import CameraStream from "./components/CameraStream";

export default function Home() {
  return (
    <div className="flex flex-col flex-1 items-center justify-center bg-zinc-50 dark:bg-zinc-950 p-8">
      <main className="flex flex-col items-center gap-8 w-full max-w-2xl">
        <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
          ONNX Web YOLO PoC
        </h1>
        <CameraStream />
      </main>
    </div>
  );
}
