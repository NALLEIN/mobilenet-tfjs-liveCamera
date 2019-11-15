import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet_classes';
import Stats from 'stats.js';

const MOBILENET_MODEL_PATH =
  'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;
const videoWidth = 224;
const videoHeight = 224;
const stats = new Stats();

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

/*
  Loads a the camera to be used in the demo
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  //const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: 'user',
      width: videoWidth,
      height: videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('fps').appendChild(stats.dom);
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

async function initVideo() {
  let videoElement;
  try {
    videoElement = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }
  if(document.getElementById('videoContainer') === null) {
    const videoContainer = document.createElement('div');
    videoContainer.appendChild(videoElement);
    videoContainer.id='videoContainer';
    predictionContainer.appendChild(videoContainer);
  }
}

function showVideo(videoElement){
  if(document.getElementById('videoContainer') === null) {
    const videoContainer = document.createElement('div');
    videoContainer.appendChild(videoElement);
    videoContainer.id='videoContainer';
    predictionContainer.appendChild(videoContainer);
  }
}
function showResults(classes) {
  const probsContainer = document.createElement('div');
  probsContainer.id='probsContainer';
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }

  if(document.getElementById('probsContainer') === null) {
    predictionContainer.appendChild(probsContainer);
  }
  else {
    predictionContainer.replaceChild(probsContainer,document.getElementById('probsContainer'));
  }
}

async function mobilenetRealTime() {
  status('Infering realtime')
  let videoInput;
  try {
    videoInput = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  const img = tf.browser.fromPixels(videoInput).toFloat();
  const offset = tf.scalar(127.5);
  const normalized = img.sub(offset).div(offset);
  const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  //tf.webgl.forceHalfFloat();
  //console.info('WEBGL_FORCE_F16_TEXTURES : ',tf.ENV.getBool('WEBGL_FORCE_F16_TEXTURES'));
  console.info('floatPercision : ',tf.backend().floatPrecision());

  stats.begin();
  const results = mobilenet.predict(batched);
  stats.end();
  
  const classes = await getTopKClasses(results, TOPK_PREDICTIONS);
  showVideo(videoInput);
  showResults(classes);

  requestAnimationFrame(mobilenetRealTime);
}


/*
Main
*/

let mobilenet;
let init=false;
const predictionContainer = document.getElementById('predictions');
predictionContainer.className = 'pred-container';
navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

async function main() {
  status('Loadinf Model');
  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
  status(' ');
  setupFPS();
  mobilenetRealTime();
}

main();

