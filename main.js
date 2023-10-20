import "@babel/polyfill";
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 350. 
const IMAGE_SIZE = 350;
// K value for KNN
const TOPK = 10;

function Main() {
    // Initiate variables
    const infoTexts = [];
    let training = -1; // -1 when no class is being trained
    let videoPlaying = false;
    const buttonTexts = new Array(NUM_CLASSES).fill("");
    let timer; // Define timer here

    // Initiate deeplearn.js math and knn classifier objects
    bindPage();

    // Create video element that will contain the webcam image
    const video = document.createElement('video');
    // console.log(video);
    video.setAttribute('autoplay', '');
    video.setAttribute('playsinline', '');

    // Add video element to the DOM
    document.body.appendChild(video);

    // Create training buttons and info texts    
    for (let i = 0; i < NUM_CLASSES; i++) {
        const div = document.createElement('div');
        document.body.appendChild(div);
        div.style.marginBottom = '10px';

        // Create an input field for the button name
        const input = document.createElement('input');
        input.setAttribute('type', 'text');
        input.placeholder = 'Enter button name';
        div.appendChild(input);

        // Create training button
        const button = document.createElement('button');
        button.innerText = "Train";
        div.appendChild(button);

        button.addEventListener('mousedown', () => {
            training = i;
            buttonTexts[i] = input.value || `Train ${i}`;
            button.innerText = buttonTexts[i];
        });

        button.addEventListener('mouseup', () => training = -1);

        input.addEventListener('blur', () => {
            button.innerText = input.value || `Train ${i}`;
        });

        const infoText = document.createElement('span');
        infoText.innerText = " No examples added";
        div.appendChild(infoText);
        infoTexts.push(infoText);
    }

    // Setup webcam
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then((stream) => {
            video.srcObject = stream;
            video.width = IMAGE_SIZE;
            video.height = IMAGE_SIZE;

            video.addEventListener('playing', () => videoPlaying = true);
            video.addEventListener('paused', () => videoPlaying = false);
        });

    const apiEndpoint = 'http://159.203.20.26:8085/api/v1/lmIuwb6jRcLZ0v3Y2pfz/telemetry';

    const toggleButton = document.createElement('button');
    toggleButton.innerText = 'Start Detection';
    document.body.appendChild(toggleButton);

    toggleButton.addEventListener('click', () => toggleDetection());

    async function sendToServer(data) {
        try {
            await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error('An error occurred while sending data to the server:', error);
        }
    }

    async function bindPage() {
        const knn = knnClassifier.create();
        const mobilenet = await mobilenetModule.load();

        start();

        function start() {
            if (timer) {
                stop();
            }
            video.play();
            timer = requestAnimationFrame(animate);
        }

        function stop() {
            video.pause();
            cancelAnimationFrame(timer);
        }

        async function animate() {
            if (videoPlaying) {
                const image = tf.fromPixels(video);
                // console.log(image);
                let logits;
                const infer = () => mobilenet.infer(image, 'conv_preds');

                if (training != -1) {
                    logits = infer();
                    try {
                        knn.addExample(logits, training);
                    } catch (error) {
                        console.error('Error adding example:', error);
                    }
                }

                const numClasses = knn.getNumClasses();
                if (numClasses > 0) {
                    logits = infer();
                    const res = await knn.predictClass(logits, TOPK);
                    // console.log(res);
                    for (let i = 0; i < NUM_CLASSES; i++) {
                        const exampleCount = knn.getClassExampleCount();

                        if (res.classIndex == i) {
                            infoTexts[i].style.fontWeight = 'bold';
                        } else {
                            infoTexts[i].style.fontWeight = 'normal';
                        }

                        if (exampleCount[i] > 0) {
                            infoTexts[i].innerText = `${exampleCount[i]} examples - ${(res.confidences[i] * 100) || 0}%`;

                            function extractNumberWithPercentage(input) {
                                var match = input.match(/\d+(\.\d+)?%/);
                                if (match) {
                                    return parseFloat(match[0]);
                                }
                                return null;
                            }
                            var percentage = extractNumberWithPercentage(infoTexts[i].innerText);

                            if (percentage >= 60) {
                                const buttonText = buttonTexts[i];
                                const dataToSend = { [buttonText]: buttonText };
                                sendToServer(dataToSend);
                            }
                        } else {
                            infoTexts[i].innerText = `No examples added`;
                        }
                    }
                }

                image.dispose();
                if (logits != null) {
                    logits.dispose();
                }
            }
            timer = requestAnimationFrame(animate);
            // console.log(timer);
        }
    }

    function toggleDetection() {
        if (detectionRunning) {
            detectionRunning = false;
            stop();
            toggleButton.innerText = 'Start Detection';
            console.log('Detection stopped.');
        } else {
            detectionRunning = true;
            start();
            toggleButton.innerText = 'Stop Detection';
            console.log('Detection Started.');
        }
    }
}

window.addEventListener('load', Main);
