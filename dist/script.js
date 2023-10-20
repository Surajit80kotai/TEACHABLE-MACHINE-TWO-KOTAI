document.addEventListener('DOMContentLoaded', function () {
    const NUM_CLASSES = 3;
    const exampleCount = new Array(NUM_CLASSES).fill(0);
    const TOPK = 10;
    let training = -1;
    let timer;

    const trainButtons = document.querySelectorAll('.train-btn');
    const knn = knnClassifier.create();

    async function sendToServer(data) {
        try {
            const apiEndpoint = 'http://159.203.20.26:8085/api/v1/lmIuwb6jRcLZ0v3Y2pfz/telemetry';
            await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                mode: 'cors',  // This enables CORS mode
                credentials: 'include',  // This includes cookies in the request
                body: JSON.stringify(data),
            });
        } catch (error) {
            console.error('An error occurred while sending data to the server:', error);
        }
    }


    async function bindPage() {
        const mobiLenet = await mobilenet.load();
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

        async function trainClass(classIndex, videoElement) {
            training = classIndex;

            const numExamples = 50;
            for (let i = 0; i < numExamples; i++) {
                if (training === classIndex) {
                    const image = tf.browser.fromPixels(videoElement);
                    const logits = mobiLenet.infer(image, 'conv_preds');
                    knn.addExample(logits, classIndex);
                    image.dispose();
                    logits.dispose();

                    const progressBar = document.getElementById(`progress-bar-${classIndex}`);
                    progressBar.style.width = `${(i / numExamples) * 100}%`;
                    await tf.nextFrame();
                } else {
                    break;
                }
            }

            const progressBar = document.getElementById(`progress-bar-${classIndex}`);
            progressBar.style.width = '0%';
            training = -1;
        }

        trainButtons.forEach((button, index) => {
            button.addEventListener('mousedown', function () {
                trainClass(index, video);
            });

            button.addEventListener('mouseup', function () {
                training = -1;
            });
        });

        async function animate() {
            const image = tf.browser.fromPixels(video);
            let logits = null;
            const infer = () => mobiLenet.infer(image, 'conv_preds');

            if (training !== -1) {
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

                for (let i = 0; i < NUM_CLASSES; i++) {
                    const exampleCount = knn.getClassExampleCount();

                    if (res.classIndex === i) {
                        // Change traffic light color based on detected object
                        $('#status-fp').text('');
                        if (i === 0) {
                            $('#boje-3').css('background', 'green');
                            $('#boje-1').css('background', '');
                            $('#boje-2').css('background', '');
                            $('#status-fp').text('Pass');
                            $('#status-bg').addClass('bg-success');
                            $('#status-bg').removeClass('bg-danger');
                            $('#status-bg').removeClass('bg-warning');
                            const buttonText1 = $("#btn-1").text();
                            sendToServer({ [buttonText1]: buttonText1 });
                        } else if (i === 1) {
                            $('#boje-2').css('background', 'yellow');
                            $('#boje-1').css('background', '');
                            $('#boje-3').css('background', '');
                            $('#status-fp').text('Reject');
                            $('#status-bg').addClass('bg-warning');
                            $('#status-bg').removeClass('bg-danger');
                            $('#status-bg').removeClass('bg-success');
                            const buttonText2 = $("#btn-2").text();
                            sendToServer({ [buttonText2]: buttonText2 });
                        } else if (i === 2) {
                            $('#boje-1').css('background', 'red');
                            $('#boje-2').css('background', '');
                            $('#boje-3').css('background', '');
                            $('#status-fp').text('Reject');
                            $('#status-bg').addClass('bg-danger');
                            $('#status-bg').removeClass('bg-warning');
                            $('#status-bg').removeClass('bg-warning');
                            const buttonText3 = $("#btn-3").text();
                            sendToServer({ [buttonText3]: buttonText3 });
                        } else {
                            $('#boje-1').css('background', '');
                            $('#boje-2').css('background', '');
                            $('#boje-3').css('background', '');
                        }
                    }
                }
            }

            image.dispose();
            if (logits !== null) {
                logits.dispose();
            }
            timer = requestAnimationFrame(animate);
        }
    }

    bindPage();
});