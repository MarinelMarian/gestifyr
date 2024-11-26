import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

// DOM elements
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("output");
const canvasCtx = canvasElement.getContext("2d");
const yawElement = document.getElementById("yaw");
const pitchElement = document.getElementById("pitch");
const shakeStatusElement = document.getElementById("shake-status");
const nodStatusElement = document.getElementById("nod-status");

// Constants for head pose estimation
const shakeThreshold = 15; // degrees
const nodThreshold = 10; // degrees

// Variables
let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastYaw = 0;
let lastPitch = 0;

// Initialize FaceLandmarker
async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode,
    numFaces: 1,
  });

  startWebcam();
}

// Enable Webcam
function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      videoElement.srcObject = stream;
      videoElement.addEventListener("loadeddata", predictWebcam);
    });
}

// Calculate head pose using landmarks
function calculateHeadPose(landmarks) {
  // Select key landmarks
  const noseTip = landmarks[1];
  const leftEyeOuter = landmarks[33];
  const rightEyeOuter = landmarks[263];
  const chin = landmarks[152];

  // Calculate vectors
  const horizontalVec = [
    rightEyeOuter.x - leftEyeOuter.x,
    rightEyeOuter.y - leftEyeOuter.y,
    rightEyeOuter.z - leftEyeOuter.z,
  ];
  const verticalVec = [
    chin.x - noseTip.x,
    chin.y - noseTip.y,
    chin.z - noseTip.z,
  ];

  // Normalize vectors
  const normHorizontal = Math.sqrt(
    horizontalVec[0] ** 2 + horizontalVec[1] ** 2 + horizontalVec[2] ** 2
  );
  const normVertical = Math.sqrt(
    verticalVec[0] ** 2 + verticalVec[1] ** 2 + verticalVec[2] ** 2
  );

  const normalizedHorizontal = horizontalVec.map((v) => v / normHorizontal);
  const normalizedVertical = verticalVec.map((v) => v / normVertical);

  // Calculate angles
  const yaw = Math.atan2(
    normalizedHorizontal[1],
    normalizedHorizontal[0]
  );
  const pitch = Math.asin(normalizedVertical[2]);

  return {
    yaw: (yaw * 180) / Math.PI, // Convert to degrees
    pitch: (pitch * 180) / Math.PI, // Convert to degrees
  };
}

// Process Webcam Frames
async function predictWebcam() {
  if (!faceLandmarker) return;

  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  const startTimeMs = performance.now();
  const results = faceLandmarker.detectForVideo(videoElement, startTimeMs);

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    const landmarks = results.faceLandmarks[0];

    // Calculate head pose
    const { yaw, pitch } = calculateHeadPose(landmarks);

    // Detect gestures
    const isShake = Math.abs(yaw - lastYaw) > shakeThreshold;
    const isNod = Math.abs(pitch - lastPitch) > nodThreshold;

    // Update last values
    lastYaw = yaw;
    lastPitch = pitch;

    // Update UI
    yawElement.textContent = yaw.toFixed(2);
    pitchElement.textContent = pitch.toFixed(2);
    shakeStatusElement.textContent = isShake.toString();
    nodStatusElement.textContent = isNod.toString();

    // Draw landmarks and head pose
    const drawingUtils = new DrawingUtils(canvasCtx);
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#C0C0C070", lineWidth: 1 }
    );
    drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 1 });
  }

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}

// Start Application
createFaceLandmarker();
