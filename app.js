const input = document.querySelector('#image-input');
const dropzone = document.querySelector('#dropzone');
const selected = document.querySelector('#selected-file');
const preview = document.querySelector('#preview');
const analyzeButton = document.querySelector('#analyze-button');
const maxBytes = 10 * 1024 * 1024;
const acceptedTypes = ['image/jpeg', 'image/png', 'image/webp'];
let selectedFile = null;
let objectUrl = null;
let busy = false;

function showState(state) {
  for (const name of ['empty-result', 'pending-result', 'error-result', 'result-content']) {
    document.querySelector(`#${name}`).hidden = name !== state;
  }
  const hints = {
    'empty-result': selectedFile ? 'Image selected. Start analysis when ready.' : 'Your model prediction will appear here.',
    'pending-result': 'The model is processing your image.',
    'error-result': 'Analysis stopped. Your image is ready to retry.',
    'result-content': 'Prediction returned by the research model.'
  };
  document.querySelector('#result-hint').textContent = hints[state];
}

function clearFileError() {
  document.querySelector('#file-error').hidden = true;
}

function showFileError(title, message) {
  document.querySelector('#file-error-title').textContent = title;
  document.querySelector('#file-error-message').textContent = message;
  document.querySelector('#file-error').hidden = false;
}

function resetResult() {
  document.querySelector('#error-title').textContent = 'Prediction couldn’t be completed';
  document.querySelector('#error-message').textContent = 'Prediction service unavailable. Try again.';
  showState('empty-result');
}

function readImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => { URL.revokeObjectURL(url); resolve(); };
    image.onerror = () => { URL.revokeObjectURL(url); reject(new Error('unreadable')); };
    image.src = url;
  });
}

async function displayFile(file) {
  if (!file) return;
  clearFileError();
  if (!acceptedTypes.includes(file.type)) {
    showFileError('Unsupported file type', 'Choose a JPEG, PNG, or WebP image.');
    input.value = '';
    return;
  }
  if (file.size > maxBytes) {
    showFileError('Image is too large', 'Choose an image that is 10 MiB or smaller.');
    input.value = '';
    return;
  }
  try {
    await readImage(file);
  } catch {
    showFileError('Image can’t be read', 'This file may be damaged. Choose a different JPEG, PNG, or WebP image.');
    input.value = '';
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  document.querySelector('#file-name').textContent = file.name;
  document.querySelector('#file-size').textContent = `${(file.size / 1024).toFixed(0)} KB · ready`;
  selected.hidden = false;
  dropzone.hidden = true;
  analyzeButton.hidden = false;
  analyzeButton.disabled = false;
  document.querySelector('#sample-button').disabled = false;
  resetResult();
}

function clearImage() {
  selectedFile = null;
  selected.hidden = true;
  dropzone.hidden = false;
  analyzeButton.hidden = true;
  analyzeButton.disabled = false;
  input.value = '';
  clearFileError();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  preview.removeAttribute('src');
  resetResult();
}

async function runAnalysis() {
  console.log("Run analysis clicked", {
  selectedFile,
  busy
});
  if (!selectedFile || busy) return;
  busy = true;
  analyzeButton.disabled = true;
  analyzeButton.textContent = 'Analyzing…';
  document.querySelector('#replace-image').disabled = true;
  document.querySelector('#remove-image').disabled = true;
  showState('pending-result');
  try {
    const body = new FormData();
    body.append('image', selectedFile);
    console.log("Sending request to Flask API...");
    const response = await fetch(
    'http://127.0.0.1:5000/api/v1/predictions',
    {
      method: 'POST',
      body
    }
    );
    console.log("Flask response status:", response.status);
    let data;
    try { data = await response.json(); } catch { throw new Error('unavailable'); }
    if (!response.ok) {
      const message = data?.error?.message || 'Prediction service unavailable. Try again.';
      document.querySelector('#error-title').textContent = response.status === 503 ? 'Model service unavailable' : 'Prediction couldn’t be completed';
      document.querySelector('#error-message').textContent = message;
      showState('error-result');
      return;
    }
    if (!['Cancer', 'Non-Cancer'].includes(data.predicted_class) || typeof data.score !== 'number' || data.score < 0 || data.score > 1 || typeof data.model_version !== 'string' || !data.model_version.trim()) throw new Error('unavailable');
    document.querySelector('#result-label').textContent = 'Prediction complete';
    document.querySelector('#prediction-class').textContent = data.predicted_class;
    document.querySelector('#score-value').textContent = data.score.toFixed(2);
    document.querySelector('#score-fill').style.width = `${data.score * 100}%`;
    document.querySelector('#model-version').textContent = data.model_version;
    const gradcam = typeof data.grad_cam_url === 'string' && data.grad_cam_url.trim() ? data.grad_cam_url : null;
    document.querySelector('#explanation').hidden = !gradcam;
    document.querySelector('#no-explanation').hidden = !!gradcam;
    document.querySelector('#no-explanation span').textContent = 'The model returned a prediction without a Grad-CAM visualization.';
    if (gradcam) document.querySelector('#gradcam-image').src = gradcam;
    else document.querySelector('#gradcam-image').removeAttribute('src');
    showState('result-content');
  } catch {
    document.querySelector('#error-title').textContent = 'Prediction service unavailable';
    document.querySelector('#error-message').textContent = 'Prediction service unavailable. Try again.';
    showState('error-result');
  } finally {
    busy = false;
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = 'Run analysis <span>→</span>';
    document.querySelector('#replace-image').disabled = false;
    document.querySelector('#remove-image').disabled = false;
  }
}

input.addEventListener('change', () => displayFile(input.files[0]));
document.querySelector('#replace-image').addEventListener('click', () => input.click());
document.querySelector('#remove-image').addEventListener('click', clearImage);
document.querySelector('#file-error-retry').addEventListener('click', () => input.click());
document.querySelector('#retry-button').addEventListener('click', runAnalysis);
document.querySelector('#gradcam-image').addEventListener('error', () => {
  document.querySelector('#explanation').hidden = true;
  document.querySelector('#no-explanation').hidden = false;
  document.querySelector('#no-explanation span').textContent = 'The explanation image could not be loaded.';
});
for (const event of ['dragenter', 'dragover']) dropzone.addEventListener(event, (e) => {
  e.preventDefault();
  dropzone.classList.add('dragging');
});
for (const event of ['dragleave', 'drop']) dropzone.addEventListener(event, (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragging');
});
dropzone.addEventListener('drop', (e) => displayFile(e.dataTransfer.files[0]));
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
});
analyzeButton.addEventListener('click', runAnalysis);

// Keep the demo case clearly illustrative and never present a fabricated explanation image.
function loadSample() {
  clearImage();
  document.querySelector('#empty-result').hidden = true;
  document.querySelector('#result-content').hidden = false;
  document.querySelector('#result-hint').textContent = 'Illustrative output using the dataset’s two class names.';
  document.querySelector('#result-label').textContent = 'Illustrative sample output';
  document.querySelector('#prediction-class').textContent = 'Cancer';
  document.querySelector('#score-value').textContent = '0.82';
  document.querySelector('#score-fill').style.width = '82%';
  document.querySelector('#model-version').textContent = 'v0.1 · example';
  document.querySelector('#explanation').hidden = true;
  document.querySelector('#no-explanation').hidden = false;
  document.querySelector('#no-explanation span').textContent = 'Sample output only. No model inference or explanation image is available.';
}
document.querySelector('#sample-button').addEventListener('click', loadSample);
document.querySelector('#sample-arrow').addEventListener('click', loadSample);

const pageTitles = { overview: 'Overview', analysis: 'New analysis', history: 'Case history', 'model-info': 'Model info' };
function showPage(page) {
  if (!pageTitles[page]) return;
  if (page === 'history') return;
  document.querySelectorAll('[data-view]').forEach((section) => {
    section.hidden = !section.dataset.view.split(' ').includes(page);
  });
  document.querySelectorAll('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.page === page));
  document.querySelector('#breadcrumb-page').textContent = pageTitles[page];
  document.querySelector('.main-content').scrollTop = 0;
  window.scrollTo(0, 0);
}
document.querySelectorAll('.nav-item[data-page]:not([aria-disabled="true"])').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault();
  const page = link.dataset.page;
  history.replaceState(null, '', `#${page}`);
  showPage(page);
}));
document.querySelectorAll('[data-go]').forEach((button) => button.addEventListener('click', () => {
  const page = button.dataset.go;
  history.replaceState(null, '', `#${page}`);
  showPage(page);
}));
const initialPage = location.hash.slice(1);
showPage(['overview', 'analysis', 'model-info'].includes(initialPage) ? initialPage : 'overview');
