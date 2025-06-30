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
backgroundImage.src = "./background.jpg"; // Path to your background image

// Create a canvas for the background image
const backgroundCanvas = document.createElement("canvas");
const backgroundCtx = backgroundCanvas.getContext("2d");

// Background image load handler
backgroundImage.onload = () => {
  console.log("Background image has loaded.");
};

// --- Image Segmenter Initialization ---
const createImageSegmenter = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );

  imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite",
      delegate: "GPU",
    },
    runningMode: runningMode,
    outputCategoryMask: false, // IMPORTANT: disable category mask
    outputConfidenceMasks: true, // Enable soft mask for blending
  });

  demosSection.classList.remove("invisible");
};
createImageSegmenter();

// --- Main Processing Callback with Soft Edge Blending ---
function callbackForVideo(result) {
  const videoImageData = canvasCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  );
  const videoData = videoImageData.data;

  const backgroundData = backgroundCtx.getImageData(
    0,
    0,
    video.videoWidth,
    video.videoHeight
  ).data;

  // Extract confidence mask for person (index 1)
  const confidenceMasks = result.confidenceMasks;
  const personMask = confidenceMasks[0].getAsFloat32Array();

  let j = 0;
  for (let i = 0; i < personMask.length; ++i) {
    let maskVal = personMask[i]; // 0.0 = background, 1.0 = person

    // Clamp mask value for safety
    maskVal = Math.min(Math.max(maskVal, 0), 1);

    // Soft edge blending (weighted average)
    videoData[j] = maskVal * videoData[j] + (1 - maskVal) * backgroundData[j]; // Red
    videoData[j + 1] =
      maskVal * videoData[j + 1] + (1 - maskVal) * backgroundData[j + 1]; // Green
    videoData[j + 2] =
      maskVal * videoData[j + 2] + (1 - maskVal) * backgroundData[j + 2]; // Blue
    // Keep alpha as is

    j += 4;
  }

  canvasCtx.putImageData(videoImageData, 0, 0);

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// --- Webcam Handling ---
async function predictWebcam() {
  canvasCtx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

  if (imageSegmenter) {
    imageSegmenter.segmentForVideo(video, performance.now(), callbackForVideo);
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

async function enableCam() {
  if (!imageSegmenter) return;

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning
    ? "DISABLE WEBCAM"
    : "ENABLE WEBCAM";

  if (webcamRunning) {
    const constraints = { video: true };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;

    video.addEventListener("loadeddata", () => {
      // Webcam is loaded, resize the background canvas
      backgroundCanvas.width = video.videoWidth;
      backgroundCanvas.height = video.videoHeight;
      backgroundCtx.drawImage(
        backgroundImage,
        0,
        0,
        video.videoWidth,
        video.videoHeight
      );

      // Start processing loop
      predictWebcam();
    });
  } else {
    // Stop the loop and the webcam stream
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
