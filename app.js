import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const { ImageSegmenter, FilesetResolver } = vision;

// Demo: Continuously grab image from webcam stream and segmented it.

// Get DOM elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
let enableWebcamButton;
let webcamRunning = false;
let runningMode = "VIDEO";

let imageSegmenter;
let lastWebcamTime = -1;

const createImageSegmenter = async () => {
  const audio = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  imageSegmenter = await ImageSegmenter.createFromOptions(audio, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
};

function getFaceImage(mask) {
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  const imageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  ).data;

  let j = 0;

  for (let i = 0; i < mask.length; ++i) {
    if (mask[i] > 0) {
      imageData[j] = 0;
      imageData[j + 1] = 0;
      imageData[j + 2] = 0;
      imageData[j + 3] = 0;
    }

    j += 4;
  }
  const uint8Array = new Uint8ClampedArray(imageData.buffer);
  const image = new ImageData(uint8Array, video.videoWidth, video.videoHeight);

  return image;
}

function callbackForVideo(result) {
  const mask = result.categoryMask.getAsFloat32Array();
  const faceImage = getFaceImage(mask);
  const image = video;
  // canvasCtx.putImageData(faceImage, 0, 0)
  canvasCtx.save();
  canvasCtx.fillStyle = "white";
  canvasCtx.clearRect(0, 0, video.videoWidth, video.videoHeight);
  canvasCtx.filter = "blur(0)";
  canvasCtx.drawImage(image, 0, 0, video.videoWidth, video.videoHeight);
  // Only overwrite existing pixels.
  canvasCtx.globalCompositeOperation = "destination-atop";

  canvasCtx.filter = "blur(24px)";

  // canvasCtx.drawImage(faceImage, 0, 0, video.videoWidth, video.videoHeight)
  canvasCtx.putImageData(faceImage, 0, 0);
  // Only overwrite missing pixels.
  canvasCtx.globalCompositeOperation = "destination-over";

  canvasCtx.drawImage(image, 0, 0, video.videoWidth, video.videoHeight);
  canvasCtx.restore();
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Get segmentation from the webcam
async function predictWebcam() {
  if (video.currentTime === lastWebcamTime) {
    if (webcamRunning === true) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  lastWebcamTime = video.currentTime;
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  // Do not segmented if imageSegmenter hasn't loaded
  if (imageSegmenter === undefined) {
    return;
  }
  // if image mode is initialized, create a new segmented with video runningMode
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await imageSegmenter.setOptions({
      runningMode: runningMode,
    });
  }
  let startTimeMs = performance.now();

  // Start segmenting the stream.
  imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
}

// Enable the live webcam view and start imageSegmentation.
async function enableCam(event) {
  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "ENABLE SEGMENTATION";
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE SEGMENTATION";
  }

  // getUsermedia parameters.
  const constraints = {
    video: true,
  };

  // Activate the webcam stream.
  video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  video.addEventListener("loadeddata", predictWebcam);
}

// App Starts
createImageSegmenter();

// If webcam supported, add event listener to button.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
