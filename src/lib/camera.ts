export async function prepareVideoStream(
  video: HTMLVideoElement,
): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });
  video.srcObject = stream;
  video.loop = false;
  return stream;
}
