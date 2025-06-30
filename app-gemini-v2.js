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

// Create a canvas for the background image to get its pixel data
const backgroundCanvas = document.createElement("canvas");
const backgroundCtx = backgroundCanvas.getContext("2d");

// We only need to know when the image is loaded. The drawing will happen later.
backgroundImage.onload = () => {
  console.log("Background image has loaded.");
};

// --- Image Segmenter Initialization ---
const createImageSegmenter = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  // https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite
  // https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
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
};
createImageSegmenter();

// --- Main Processing Callback ---
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

  const mask = result.categoryMask.getAsFloat32Array();

  let j = 0;
  for (let i = 0; i < mask.length; ++i) {
    const isPerson = mask[i] !== 0;

    if (!isPerson) {
      videoData[j] = backgroundData[j];
      videoData[j + 1] = backgroundData[j + 1];
      videoData[j + 2] = backgroundData[j + 2];
    }
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

async function enableCam(event) {
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
      // **FIX APPLIED HERE**
      // Now that the video has loaded, we know its dimensions.
      // Set the background canvas to the same size and draw the image.
      backgroundCanvas.width = video.videoWidth;
      backgroundCanvas.height = video.videoHeight;
      backgroundCtx.drawImage(
        backgroundImage,
        0,
        0,
        video.videoWidth,
        video.videoHeight
      );
      // Now start the prediction loop.
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
