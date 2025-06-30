import {
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

// Demo: Continuously grab image from webcam stream and segmented it.

// Get DOM elements
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const demosSection = document.getElementById("demos");

let webcamRunning = false;
let runningMode = "VIDEO";
let enableWebcamButton, imageSegmenter, labels;

const createImageSegmenter = async () => {
  const audio = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  imageSegmenter = await ImageSegmenter.createFromOptions(audio, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
  labels = imageSegmenter.getLabels();
  demosSection.classList.remove("invisible");
};

function callbackForVideo(result) {
  // Step 1: Capture current video frame from canvas
  let imageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const videoPixels = imageData.data;

  // Step 2: Draw background to an offscreen canvas to capture its pixels
  const offscreenCanvas = document.createElement("canvas");
  offscreenCanvas.width = video.videoWidth;
  offscreenCanvas.height = video.videoHeight;
  const offscreenCtx = offscreenCanvas.getContext("2d");
  offscreenCtx.drawImage(
    backgroundImage,
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const bgImageData = offscreenCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const bgPixels = bgImageData.data;

  // Step 3: Get the mask
  const mask = result.categoryMask.getAsFloat32Array();

  let j = 0;
  for (let i = 0; i < mask.length; ++i) {
    const maskVal = mask[i];

    if (maskVal < 0.5) {
      // Background pixel: replace with background image pixel
      videoPixels[j] = bgPixels[j]; // Red
      videoPixels[j + 1] = bgPixels[j + 1]; // Green
      videoPixels[j + 2] = bgPixels[j + 2]; // Blue
      // Keep alpha as is
    } else {
      // Foreground pixel: keep webcam pixel as is
      // Explicitly keep original webcam pixel (this line is optional but good for clarity)
      videoPixels[j] = videoPixels[j]; // Red
      videoPixels[j + 1] = videoPixels[j + 1]; // Green
      videoPixels[j + 2] = videoPixels[j + 2]; // Blue
      // Keep alpha as is
    }

    j += 4;
  }

  // Step 4: Draw final frame
  canvasCtx.putImageData(imageData, 0, 0);

  // Continue the loop
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// Get segmentation from the webcam
let lastWebcamTime = -1;
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
  if (imageSegmenter === undefined) {
    return;
  }

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

// Demo Starts
const backgroundImage = new Image();
backgroundImage.src = "./background.jpg"; // Replace with your actual image path
backgroundImage.onload = () => {
  console.log("Background image loaded");
};

createImageSegmenter();

// If webcam supported, add event listener to button.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
