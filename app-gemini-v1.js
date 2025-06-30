import {
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

// --- DOM and Global Variables ---
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("canvas");
const canvasCtx = canvasElement.getContext("2d");
const demosSection = document.getElementById("demos");

let webcamRunning = false;
let runningMode = "VIDEO";
let imageSegmenter;
let enableWebcamButton;

// --- Load the Background Image ---
const backgroundImage = new Image();
backgroundImage.src = "./background.jpg"; // Using a placeholder background image

// Create a canvas for the background image to get its pixel data
const backgroundCanvas = document.createElement("canvas");
const backgroundCtx = backgroundCanvas.getContext("2d");

backgroundImage.onload = () => {
  backgroundCanvas.width = video.videoWidth;
  backgroundCanvas.height = video.videoHeight;
  backgroundCtx.drawImage(
    backgroundImage,
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  console.log("Background image loaded and drawn to canvas");
};

// --- Image Segmenter Initialization ---
const createImageSegmenter = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  });
  demosSection.classList.remove("invisible");
  console.log("Image segmenter created");
};
createImageSegmenter();

// --- Main Processing Callback ---
function callbackForVideo(result) {
  // Get the pixel data from the video frame
  const videoImageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const videoData = videoImageData.data;

  // Get the background image pixel data
  const backgroundData = backgroundCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  ).data;

  // Get the segmentation mask
  const mask = result.categoryMask.getAsFloat32Array();

  let j = 0;
  for (let i = 0; i < mask.length; ++i) {
    // If the mask value is 0, it's the background. Otherwise, it's the person.
    const isPerson = mask[i] !== 0;

    if (!isPerson) {
      // If it's the background, replace the video pixel with the background image pixel
      videoData[j] = backgroundData[j]; // Red
      videoData[j + 1] = backgroundData[j + 1]; // Green
      videoData[j + 2] = backgroundData[j + 2]; // Blue
      // videoData[j + 3] is the alpha channel, we can leave it as is
    }
    j += 4;
  }

  // Draw the modified pixel data back to the main canvas
  canvasCtx.putImageData(videoImageData, 0, 0);

  // Continue the loop
  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// --- Webcam Handling ---
async function predictWebcam() {
  if (video.currentTime === lastWebcamTime) {
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
    return;
  }
  lastWebcamTime = video.currentTime;

  // Draw video frame to canvas to get pixel data
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  if (imageSegmenter) {
    const startTimeMs = performance.now();
    imageSegmenter.segmentForVideo(video, startTimeMs, callbackForVideo);
  }
}

let lastWebcamTime = -1;

async function enableCam(event) {
  if (!imageSegmenter) {
    console.log("Wait! Image segmenter not loaded yet.");
    return;
  }

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning
    ? "DISABLE WEBCAM"
    : "ENABLE WEBCAM";

  if (webcamRunning) {
    const constraints = { video: true };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  } else {
    video.srcObject.getTracks().forEach((track) => track.stop());
  }
}

// --- Event Listeners ---
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}
