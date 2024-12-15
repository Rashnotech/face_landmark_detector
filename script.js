import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const demosSection = document.getElementById("demos");
const webcamButton = document.getElementById("webcamButton");
const cameraContainer = document.getElementById("camera-container");
const actionInstructions = document.getElementById("action-instructions");
const currentAction = document.getElementById("current-action");
const capturedPhotoCanvas = document.getElementById("captured-photo");
const proceedButton = document.getElementById("proceed-button");

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

let faceLandmarker;
let runningMode = "IMAGE";
let webcamRunning = false;
const videoWidth = 480;

// Face detection states
const STATES = {
    MOUTH_OPEN: 'mouth_open',
    EYE_BLINK: 'eye_blink',
    COMPLETE: 'complete'
};
let currentState = STATES.MOUTH_OPEN;

async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
    demosSection.classList.remove("invisible");
}
createFaceLandmarker();

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    webcamButton.addEventListener("click", enableCam);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}

function enableCam(event) {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }

    webcamButton.style.display = 'none';
    cameraContainer.style.display = 'block';
    actionInstructions.style.display = 'block';

    const constraints = { video: true };
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam() {
    const radio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * radio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * radio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: runningMode });
    }

    // Clear previous drawings
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    // Draw face landmarks
    if (results && results.faceLandmarks && results.faceLandmarks.length > 0) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
        }
    }

    // Check face blendshapes
    if (results && results.faceBlendshapes && results.faceBlendshapes.length > 0) {
        const blendshapes = results.faceBlendshapes[0];
        
        // Check for mouth open
        if (currentState === STATES.MOUTH_OPEN) {
            const mouthOpenLeft = blendshapes.categories.find(category => category.categoryName === 'mouthLowerDownLeft')?.score || 0;
            const mouthOpenRight = blendshapes.categories.find(category => category.categoryName === 'mouthLowerDownRight')?.score || 0;
            if (mouthOpenLeft > 0.6 && mouthOpenRight) {
                currentState = STATES.EYE_BLINK;
                currentAction.textContent = 'Now blink both eyes';
            }
        }
        
        // Check for eye blink
        if (currentState === STATES.EYE_BLINK) {
            const leftEyeBlinkScore = blendshapes.categories.find(category => category.categoryName === 'browOuterUpLeft')?.score || 0;
            const rightEyeBlinkScore = blendshapes.categories.find(category => category.categoryName === 'browOuterUpRight')?.score || 0;
            console.log(leftEyeBlinkScore, rightEyeBlinkScore)
            if (leftEyeBlinkScore > 0.6 && rightEyeBlinkScore > 0.6) {
                currentState = STATES.COMPLETE;
                stopWebcam();
            }
        }
    }

    window.requestAnimationFrame(predictWebcam);
}

function stopWebcam() {
    const stream = video.srcObject;
    const tracks = stream.getTracks();

    tracks.forEach(track => track.stop());
    video.srcObject = null;

    // Create a temporary canvas to capture the image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasElement.width;
    tempCanvas.height = canvasElement.height;
    const ctx = tempCanvas.getContext('2d');
    
    // Flip the image horizontally to match the video view
    ctx.translate(tempCanvas.width, 0);
    ctx.scale(-1, 1);
    
    // Draw the video frame
    ctx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
    
    // Convert canvas to data URL and set as img src
    const imageDataUrl = tempCanvas.toDataURL('image/png');
    capturedPhotoImg.src = imageDataUrl;
    capturedPhotoImg.style.width = `${videoWidth}px`; // Set consistent width
    
    cameraContainer.style.display = 'none';
    actionInstructions.style.display = 'none';
    capturedPhotoImg.style.display = 'block';
    proceedButton.style.display = 'block';
}


proceedButton.addEventListener('click', () => {
        // Here you would typically send the captured photo to the backend
        alert('Photo captured! Proceeding to next step...');
        
        // If you want to actually send the image
        capturedPhotoCanvas.toBlob((blob) => {
            const formData = new FormData();
            const url = window.location.href;
            const token = url.split('=')[1];
            formData.append('photo', blob, `${token}.png`);
            formData.append('token', token);
            // Example fetch call (you'd replace with your actual backend endpoint)
            fetch('/image_capture', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data =>{
              alert("Your registration completed")
              window.location.href = `https://res-net-face-recognition.vercel.app/` 
            })
            .catch(error => console.error('Error:', error));
        });
});
