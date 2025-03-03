import DeviceDetector from "https://cdn.skypack.dev/device-detector-js@2.2.10";

testSupport([{ client: "Chrome" }]);
function testSupport(supportedDevices) {
    const deviceDetector = new DeviceDetector();
    const detectedDevice = deviceDetector.parse(navigator.userAgent);
    let isSupported = false;
    for (const device of supportedDevices) {
        if (device.client !== undefined) {
            const re = new RegExp(`^${device.client}$`);
            if (!re.test(detectedDevice.client.name)) {
                continue;
            }
        }
        if (device.os !== undefined) {
            const re = new RegExp(`^${device.os}$`);
            if (!re.test(detectedDevice.os.name)) {
                continue;
            }
        }
        isSupported = true;
        break;
    }
    if (!isSupported) {
        alert(`This demo, running on ${detectedDevice.client.name}/${detectedDevice.os.name}, ` +
            `is not well supported at this time, continue at your own risk.`);
    }
}
const controls = window;
const drawingUtils = window;
const mpFaceMesh = window;
const config = {
    locateFile: (file) => {
        return (`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@` +
            `${mpFaceMesh.VERSION}/${file}`);
    }
};
// Our input frames will come from here.
const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
const canvasCtx = canvasElement.getContext("2d");
/**
 * Solution options.
 */
const solutionOptions = {
    soundEnabled: false,
    selfieMode: true,
    enableFaceGeometry: false,
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
};
// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new controls.FPS();
// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
    spinner.style.display = "none";
};


let maxRoll = 0.5;
let maxPpitch = 0.83;
let maxYaw = 0.66;
let minRoll = -0.5; 
let minPpitch = -0.89;
let minYaw = -0.86;

// Detection thresholds
const nodThreshold = 0.2; // Adjust for sensitivity
const shakeThreshold = 0.4; // Adjust for sensitivity
const rollThreshold = 0.5; // Adjust for sensitivity
const mouthOpenThreshold = 0.03; // Adjust based on sensitivity
const eyebrowRaiseThreshold = 0.05; // Threshold for eyebrow raise detection
const eyeOpenThreshold = 0.02; // Threshold for eye openness detection
let isHeadNod = false;
let isHeadShake = false;
let isHeadRoll = false;
let areBothEyebrowsRaised = false;
let isMouthOpen = false;
const bufferSize = 15; // Number of frames to store
let windowSize = 2;
let distance = 1;
let isSoundEnabled = true; // Sound is enabled by default


// Buffers for storing pitch and yaw history
const pitchBuffer = [];
const yawBuffer = [];
const rollBuffer = [];

// Global state for audio playback
let isAudioPlaying = false;
let currentAudio = null;

// Function to setup a new audio file with its configuration
function setupAudio({ fileName, gestureName }) {
    const audio = new Audio(fileName);

    // Event listener to reset the playback state after audio finishes
    audio.addEventListener("ended", () => {
        isAudioPlaying = false;
        currentAudio = null;
    });

    return { audio, gestureName };
}

// Function to play audio if the specified gesture is detected
function playAudioIfDetected(audioConfig, isGestureDetected) {
    if (isGestureDetected && isSoundEnabled && !isAudioPlaying) {
        isAudioPlaying = true; // Set flag to prevent overlapping
        currentAudio = audioConfig.audio;

        currentAudio.play();
    }
}

// Setup nod and shake sounds
const yesConfig = setupAudio({ fileName: "yes.mp3", gestureName: "Nod" });
const noConfig = setupAudio({ fileName: "no.mp3", gestureName: "Shake" });
const imHungryConfig = setupAudio({ fileName: "im_hungry.mp3", gestureName: "Hungry" });
const imHappyConfig = setupAudio({ fileName: "im_happy.mp3", gestureName: "Happy" });

// Compute sliding window statistics (average, min, max) for better trend analysis
function computeSlidingWindowStats(buffer, windowSize) {
    const stats = [];
    for (let i = 0; i <= buffer.length - windowSize; i++) {
        const window = buffer.slice(i, i + windowSize);
        const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
        stats.push(avg);
    }
    return stats;
}

function updateBuffer(buffer, value, maxSize) {
    if (buffer.length >= maxSize) {
        buffer.shift(); // Remove the oldest value if the buffer is full
    }
    buffer.push(value);
}

function analyzeNod(data) {
    let buffer = computeSlidingWindowStats(data, windowSize);
    // Calculate the difference between the max and min pitch in the buffer
    const maxPitch = Math.max(...buffer);
    const minPitch = Math.min(...buffer);
    return {flag:maxPitch - minPitch > nodThreshold, value:maxPitch - minPitch};
}

function analyzeShake(data) {
    let buffer = computeSlidingWindowStats(data, windowSize);
    // Calculate the difference between the max and min yaw in the buffer
    const maxYaw = Math.max(...buffer);
    const minYaw = Math.min(...buffer);
    return {flag:maxYaw - minYaw > shakeThreshold, value:maxYaw - minYaw};
}

function analyzeRoll(data) {
    let buffer = computeSlidingWindowStats(data, windowSize);
    // Calculate the difference between the max and min yaw in the buffer
    const maxRoll = Math.max(...buffer);
    const minRoll = Math.min(...buffer);
    return {flag:maxRoll - minRoll > rollThreshold, value:maxRoll - minRoll};
}

let actualValues = {};

function onResults(results) {
    // Hide the spinner.
    document.body.classList.add("loaded");
    // Update the frame rate.
    fpsControl.tick();
    var face_2d = [];
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj      
    // https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    var points = [1, 33, 263, 61, 291, 199];
    /*
    var pointsObj = [ 0.0,
        -3.406404,
        5.979507,
        -2.266659,
        -7.425768,
        4.389812,
        2.266659,
        -7.425768,
        4.389812,
        -0.729766,
        -1.593712,
        5.833208,
        0.729766,
        -1.593712,
        5.833208,
        //0.000000, 1.728369, 6.316750];
        -1.246815,
        0.230297,
        5.681036];
  */
    var pointsObj = [0, -1.126865, 7.475604,
        -4.445859, 2.663991, 3.173422,
        4.445859, 2.663991, 3.173422,
        -2.456206, -4.342621, 4.283884,
        2.456206, -4.342621, 4.283884,
        0, -9.403378, 4.264492]; //chin
    // Draw the overlays
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    var width = results.image.width; //canvasElement.width; //
    var height = results.image.height; //canvasElement.height; //results.image.height;
    var roll = 0, pitch = 0, yaw = 0;
    var x, y, z;
    // Camera internals
    var normalizedFocaleY = 1.28; // Logitech 922
    var focalLength = height * normalizedFocaleY;
    var s = 0; //0.953571;
    var cx = width / 2;
    var cy = height / 2;
    var cam_matrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
        focalLength,
        s,
        cx,
        0,
        focalLength,
        cy,
        0,
        0,
        1
    ]);
    //The distortion parameters
    //var dist_matrix = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    var k1 = 0.1318020374;
    var k2 = -0.1550007612;
    var p1 = -0.0071350401;
    var p2 = -0.0096747708;
    var dist_matrix = cv.matFromArray(4, 1, cv.CV_64FC1, [k1, k2, p1, p2]);
    var message = "";
    if (results.multiFaceLandmarks) {
        for (const landmarks of results.multiFaceLandmarks) {
            const noseTip = landmarks[1];
            distance = noseTip.z;

            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_RIGHT_IRIS, { color: "#FF3030" });
            drawingUtils.drawConnectors(canvasCtx, landmarks, mpFaceMesh.FACEMESH_LEFT_IRIS, { color: "#30FF30" });

            const upperLip = landmarks[13]; // Center of the upper lip
            const lowerLip = landmarks[14]; // Center of the lower lip

            // Calculate vertical distance
            const mouthOpenDistance = Math.abs(lowerLip.y - upperLip.y);
            actualValues["mouth"] = Math.abs(lowerLip.y - upperLip.y);

            // Check if the mouth is open
            isMouthOpen = mouthOpenDistance > mouthOpenThreshold;
            playAudioIfDetected(imHungryConfig, isMouthOpen);


            // Calculate eyebrow-raise status
            const rightEyebrowOuter = landmarks[70];
            const rightEyebrowInner = landmarks[105];
            const rightEyeCenter = landmarks[159];

            const leftEyebrowOuter = landmarks[336];
            const leftEyebrowInner = landmarks[334];
            const leftEyeCenter = landmarks[386];

            const isRightEyebrowRaised =
                Math.min(rightEyebrowOuter.y, rightEyebrowInner.y) < rightEyeCenter.y - eyebrowRaiseThreshold;

            const isLeftEyebrowRaised =
                Math.min(leftEyebrowOuter.y, leftEyebrowInner.y) < leftEyeCenter.y - eyebrowRaiseThreshold;

            // Calculate eye-open status
            const rightEyeOpenDistance = Math.abs(rightEyeCenter.y - landmarks[145].y);
            const leftEyeOpenDistance = Math.abs(leftEyeCenter.y - landmarks[374].y);            

            const isRightEyeOpen = rightEyeOpenDistance > eyeOpenThreshold;
            const isLeftEyeOpen = leftEyeOpenDistance > eyeOpenThreshold;

            // Combine checks for eyebrow raise and eye openness
            areBothEyebrowsRaised = (isRightEyebrowRaised || isLeftEyebrowRaised) && isRightEyeOpen && isLeftEyeOpen;
            actualValues["eyebrows"] = Math.max(
                rightEyeCenter.y - Math.min(rightEyebrowOuter.y, rightEyebrowInner.y),
                leftEyeCenter.y - Math.min(leftEyebrowOuter.y, leftEyebrowInner.y)
            );

            playAudioIfDetected(imHappyConfig, areBothEyebrowsRaised);

            for (const point of points) {
                var point0 = landmarks[point];
                var point0 = landmarks[point];
                //console.log("landmarks : " + landmarks.landmark.data64F);
                drawingUtils.drawLandmarks(canvasCtx, [point0], { color: "#FFFFFF" }); // expects normalized landmark
                var x = point0.x * width;
                var y = point0.y * height;
                //var z = point0.z; 
                // Get the 2D Coordinates
                face_2d.push(x);
                face_2d.push(y);
            }
        }
    }

    if (face_2d.length > 0) {
        // Initial guess
        //Rotation in axis-angle form
        var rvec = new cv.Mat(); // = cv.matFromArray(1, 3, cv.CV_64FC1, [0, 0, 0]); //new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // Output rotation vector
        var tvec = new cv.Mat(); // = cv.matFromArray(1, 3, cv.CV_64FC1, [-100, 100, 1000]); //new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // Output translation vector
        const numRows = points.length;
        const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, face_2d);
        var modelPointsObj = cv.matFromArray(6, 3, cv.CV_64FC1, pointsObj);
        //console.log("modelPointsObj : " + modelPointsObj.data64F);
        //console.log("imagePoints : " + imagePoints.data64F);
        // https://docs.opencv.org/4.6.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
        // https://docs.opencv.org/4.6.0/d5/d1f/calib3d_solvePnP.html
        var success = cv.solvePnP(modelPointsObj, //modelPoints,
        imagePoints, cam_matrix, dist_matrix, rvec, // Output rotation vector
        tvec, false, //  uses the provided rvec and tvec values as initial approximations
        cv.SOLVEPNP_ITERATIVE //SOLVEPNP_EPNP //SOLVEPNP_ITERATIVE (default but pose seems unstable)
        );
        if (success) {
            var rmat = cv.Mat.zeros(3, 3, cv.CV_64FC1);
            const jaco = new cv.Mat();
            // console.log("rvec", rvec.data64F[0], rvec.data64F[1], rvec.data64F[2]);
            // console.log("tvec", tvec.data64F[0], tvec.data64F[1], tvec.data64F[2]);
            // Get rotational matrix rmat
            cv.Rodrigues(rvec, rmat, jaco); // jacobian	Optional output Jacobian matrix
            var sy = Math.sqrt(rmat.data64F[0] * rmat.data64F[0] + rmat.data64F[3] * rmat.data64F[3]);
            var singular = sy < 1e-6;
            // we need decomposeProjectionMatrix
            if (!singular) {
                //console.log("!singular");
                x = Math.atan2(rmat.data64F[7], rmat.data64F[8]);
                y = Math.atan2(-rmat.data64F[6], sy);
                z = Math.atan2(rmat.data64F[3], rmat.data64F[0]);
            }
            else {
                // console.log("singular");
                x = Math.atan2(-rmat.data64F[5], rmat.data64F[4]);
                //  x = Math.atan2(rmat.data64F[1], rmat.data64F[2]);
                y = Math.atan2(-rmat.data64F[6], sy);
                z = 0;
            }
            roll = z;
            pitch = x;
            yaw = y;
            var worldPoints = cv.matFromArray(9, 3, cv.CV_64FC1, [
                modelPointsObj.data64F[0] + 3,
                modelPointsObj.data64F[1],
                modelPointsObj.data64F[2],
                modelPointsObj.data64F[0],
                modelPointsObj.data64F[1] + 3,
                modelPointsObj.data64F[2],
                modelPointsObj.data64F[0],
                modelPointsObj.data64F[1],
                modelPointsObj.data64F[2] - 3,
                modelPointsObj.data64F[0],
                modelPointsObj.data64F[1],
                modelPointsObj.data64F[2],
                modelPointsObj.data64F[3],
                modelPointsObj.data64F[4],
                modelPointsObj.data64F[5],
                modelPointsObj.data64F[6],
                modelPointsObj.data64F[7],
                modelPointsObj.data64F[8],
                modelPointsObj.data64F[9],
                modelPointsObj.data64F[10],
                modelPointsObj.data64F[11],
                modelPointsObj.data64F[12],
                modelPointsObj.data64F[13],
                modelPointsObj.data64F[14],
                modelPointsObj.data64F[15],
                modelPointsObj.data64F[16],
                modelPointsObj.data64F[17] //
            ]);
            //console.log("worldPoints : " + worldPoints.data64F);
            var imagePointsProjected = new cv.Mat({ width: 9, height: 2 }, cv.CV_64FC1);
            cv.projectPoints(worldPoints, // TODO object points that never change !
            rvec, tvec, cam_matrix, dist_matrix, imagePointsProjected, jaco);
            // Draw pose
            canvasCtx.lineWidth = 5;
            var scaleX = canvasElement.width / width;
            var scaleY = canvasElement.height / height;
            canvasCtx.strokeStyle = "red";
            canvasCtx.beginPath();
            canvasCtx.moveTo(imagePointsProjected.data64F[6] * scaleX, imagePointsProjected.data64F[7] * scaleX);
            canvasCtx.lineTo(imagePointsProjected.data64F[0] * scaleX, imagePointsProjected.data64F[1] * scaleY);
            canvasCtx.closePath();
            canvasCtx.stroke();
            canvasCtx.strokeStyle = "green";
            canvasCtx.beginPath();
            canvasCtx.moveTo(imagePointsProjected.data64F[6] * scaleX, imagePointsProjected.data64F[7] * scaleX);
            canvasCtx.lineTo(imagePointsProjected.data64F[2] * scaleX, imagePointsProjected.data64F[3] * scaleY);
            canvasCtx.closePath();
            canvasCtx.stroke();
            canvasCtx.strokeStyle = "blue";
            canvasCtx.beginPath();
            canvasCtx.moveTo(imagePointsProjected.data64F[6] * scaleX, imagePointsProjected.data64F[7] * scaleX);
            canvasCtx.lineTo(imagePointsProjected.data64F[4] * scaleX, imagePointsProjected.data64F[5] * scaleY);
            canvasCtx.closePath();
            canvasCtx.stroke();
            // https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
            canvasCtx.fillStyle = "aqua";
            for (var i = 6; i <= 6 + 6 * 2; i += 2) {
                canvasCtx.rect(imagePointsProjected.data64F[i] * scaleX - 5, imagePointsProjected.data64F[i + 1] * scaleY - 5, 10, 10);
                canvasCtx.fill();
            }
            jaco.delete();
            imagePointsProjected.delete();
        }
        let ppitch = pitch < 0 ? -4 - pitch + 0.9 : 4 - pitch - 0.9;

        // Update buffers with current pitch and yaw
        updateBuffer(pitchBuffer, ppitch, bufferSize);
        updateBuffer(yawBuffer, yaw, bufferSize);
        updateBuffer(rollBuffer, roll, bufferSize);

        // Analyze buffers for nod and shake patterns
        let analysis = {};
        analysis = analyzeRoll(rollBuffer);
        isHeadRoll = analysis.flag;
        actualValues["roll"] = analysis.value;
        analysis = analyzeNod(pitchBuffer);
        isHeadNod = analysis.flag;
        actualValues["nod"] = analysis.value;
        analysis =  analyzeShake(yawBuffer);
        isHeadShake = analysis.flag;
        actualValues["shake"] = analysis.value;

        // Play sounds for detected gestures
        playAudioIfDetected(yesConfig, isHeadNod);
        playAudioIfDetected(noConfig, isHeadShake);

        maxRoll = Math.max(maxRoll, roll).toFixed(2); 
        maxPpitch = Math.max(maxPpitch, ppitch).toFixed(2); 
        maxYaw = Math.max(maxYaw, yaw).toFixed(2); 
        minRoll = Math.min(minRoll, roll).toFixed(2); 
        minPpitch = Math.min(minPpitch, ppitch).toFixed(2); 
        minYaw = Math.min(minYaw, yaw).toFixed(2); 


        const textXPos = canvasElement.width-125*scaleX;
        const textRow = 20*scaleX; 

        canvasCtx.fillStyle = "black";
        canvasCtx.font = `bold ${16*scaleX}px Arial`;
        // Display detection results
        let barWidth = 100;
        let totalBarWidth = barWidth * 1.2;
        let barHeight = 16;
        canvasCtx.fillText(`Distance: ${distance.toFixed(2)}`, textXPos, 1*textRow);
        canvasCtx.fillText(`Mouth: ${isMouthOpen}`, textXPos, 2*textRow);
        canvasCtx.fillText(`Eyebrows: ${areBothEyebrowsRaised}`, textXPos, 3*textRow);
        canvasCtx.fillText(`Nod: ${isHeadNod}`, textXPos, 4*textRow);
        canvasCtx.fillText(`Shake: ${isHeadShake}`, textXPos, 5*textRow);
        // canvasCtx.fillText(`Roll: ${isHeadRoll}`, textXPos, 6*textRow);
        // canvasCtx.fillText(//"roll: " + (180.0 * (roll / Math.PI)).toFixed(2), 
        // `roll: ${roll.toFixed(1)} \t\t(${minRoll},${maxRoll})`,
        // // "roll: " + roll.toFixed(2),
        // textXPos, 7*textRow);
        // canvasCtx.fillText(//"pitch: " + (180.0 * (pitch / Math.PI)).toFixed(2), 
        // `pitch: ${ppitch.toFixed(1)} \t\t(${minPpitch},${maxPpitch})`,
        // //     "pitch: " + pitch.toFixed(2),
        // textXPos, 8*textRow);
        // canvasCtx.fillText(//"yaw: " + (180.0 * (yaw / Math.PI)).toFixed(2), 
        // // "yaw: " + yaw.toFixed(2),
        // `yaw: ${yaw.toFixed(1)} \t\t(${minYaw},${maxYaw})`,
        // textXPos, 9*textRow);

        let offsetY = -15;

        drawProgressBar(canvasCtx, textXPos-totalBarWidth, 2*textRow + offsetY, actualValues["mouth"], mouthOpenThreshold, barWidth, barHeight);
        drawProgressBar(canvasCtx, textXPos-totalBarWidth, 3*textRow + offsetY, actualValues["eyebrows"], eyebrowRaiseThreshold, barWidth, barHeight);
        drawProgressBar(canvasCtx, textXPos-totalBarWidth, 4*textRow + offsetY, actualValues["nod"], nodThreshold, barWidth, barHeight);
        drawProgressBar(canvasCtx, textXPos-totalBarWidth, 5*textRow + offsetY, actualValues["shake"], shakeThreshold, barWidth, barHeight);
        // drawProgressBar(canvasCtx, textXPos-totalBarWidth, 6*textRow + offsetY, actualValues["roll"], rollThreshold, barWidth, barHeight);



        // canvasCtx.fillText(//"roll: " + (180.0 * (roll / Math.PI)).toFixed(2), 
        // `roll: ${roll.toFixed(1)} \t\t(${minRoll},${maxRoll})`,
        // // "roll: " + roll.toFixed(2),
        // width * 0.8, 50);
        // canvasCtx.fillText(//"pitch: " + (180.0 * (pitch / Math.PI)).toFixed(2), 
        // `pitch: ${ppitch.toFixed(1)} \t\t(${minPpitch},${maxPpitch})`,
        // //     "pitch: " + pitch.toFixed(2),
        // width * 0.8, 100);
        // canvasCtx.fillText(//"yaw: " + (180.0 * (yaw / Math.PI)).toFixed(2), 
        // // "yaw: " + yaw.toFixed(2),
        // `yaw: ${yaw.toFixed(1)} \t\t(${minYaw},${maxYaw})`,
        // width * 0.8, 150);
        // // Display detection results
        // canvasCtx.fillText(`Nod: ${isHeadNod}`, width * 0.8, 200);
        // canvasCtx.fillText(`Shake: ${isHeadShake}`, width * 0.8, 250);

        // console.log("pose %f %f %f", (180.0 * (roll / Math.PI)).toFixed(2), (180.0 * (pitch / Math.PI)).toFixed(2), (180.0 * (yaw / Math.PI)).toFixed(2));
        rvec.delete();
        tvec.delete();
    }
    canvasCtx.restore();
}
const faceMesh = new mpFaceMesh.FaceMesh(config);
faceMesh.setOptions(solutionOptions);
faceMesh.onResults(onResults);
// Present a control panel through which the user can manipulate the solution
// options.
new controls.ControlPanel(controlsElement, solutionOptions)
    .add([
    new controls.StaticText({ title: "Gestifyr" }),
    fpsControl,
    new controls.Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new controls.Toggle({ title: "Sound", field: "soundEnabled" }),
    new controls.SourcePicker({
        onFrame: async (input, size) => {
            const aspect = size.height / size.width;
            let width, height;
            if (window.innerWidth > window.innerHeight) {
                height = window.innerHeight;
                width = height / aspect;
            }
            else {
                width = window.innerWidth;
                height = width * aspect;
            }
            canvasElement.width = width;
            canvasElement.height = height;
            await faceMesh.send({ image: input });
        }
    }),
    new controls.Slider({
        title: "Max Number of Faces",
        field: "maxNumFaces",
        range: [1, 4],
        step: 1
    }),
    new controls.Toggle({
        title: "Refine Landmarks",
        field: "refineLandmarks"
    }),
    new controls.Slider({
        title: "Min Detection Confidence",
        field: "minDetectionConfidence",
        range: [0, 1],
        step: 0.01
    }),
    new controls.Slider({
        title: "Min Tracking Confidence",
        field: "minTrackingConfidence",
        range: [0, 1],
        step: 0.01
    })
])
    .on((x) => {
    const options = x;
    videoElement.classList.toggle("selfie", options.selfieMode);
    videoElement.classList.toggle("sound", options.soundEnabled);
    isSoundEnabled = options.soundEnabled;
    faceMesh.setOptions(options);
});

/**
 * Draw a horizontal progress bar on a canvas.
 * @param {CanvasRenderingContext2D} ctx - Canvas 2D rendering context.
 * @param {number} x - X coordinate of the progress bar.
 * @param {number} y - Y coordinate of the progress bar.
 * @param {number} actualValue - The current value.
 * @param {number} target - The target value.
 * @param {number} width - The width of the progress bar.
 * @param {number} height - The height of the progress bar.
 */
function drawProgressBar(ctx, x, y, actualValue, target, width = 200, height = 20) {
    // Normalize values
    const normalizedValue = (actualValue / target) * 100;
    const progressWidth = Math.min(normalizedValue, 100); // Clamp to 100% for main bar
    const overflowWidth = Math.max(0, normalizedValue - 100); // Render overflow if needed

    // Draw background bar
    ctx.fillStyle = "#ddd";
    ctx.fillRect(x, y, width, height);

    // Draw progress (yellow if below target, green if meeting/exceeding target)
    ctx.fillStyle = normalizedValue >= 100 ? "green" : "yellow";
    ctx.fillRect(x, y, (progressWidth / 100) * width, height);

    // Draw overflow in red
    if (overflowWidth > 0) {
        ctx.fillStyle = "red";
        ctx.fillRect(x + (width * progressWidth) / 100, y, (overflowWidth / 100) * width, height);
    }

    // // Draw text label
    // ctx.fillStyle = "black";
    // ctx.font = "14px Arial";
    // ctx.fillText(
    //     `${Math.min(actualValue, target).toFixed(2)} / ${target.toFixed(2)} (${normalizedValue.toFixed(0)}%)`,
    //     x + width + 10,
    //     y + height - 5
    // );
}