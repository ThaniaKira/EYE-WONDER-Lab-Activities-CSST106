(() => {
  const $ = (s, root = document) => root.querySelector(s);
  const sidebar = $('#sidebar');
  const menu = $('#menuToggle');
  const dim = $('#mobileDim');
  function closeSidebar() { sidebar?.classList.remove('open'); dim?.classList.remove('show'); }
  menu?.addEventListener('click', () => { sidebar?.classList.toggle('open'); dim?.classList.toggle('show'); });
  dim?.addEventListener('click', closeSidebar);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSidebar(); });

  const search = $('#labSearch');
  search?.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    let visible = 0;
    document.querySelectorAll('.lab-card').forEach(card => {
      const matches = card.dataset.search.includes(q);
      card.hidden = !matches;
      if (matches) visible++;
    });
    if ($('#noSearch')) $('#noSearch').hidden = visible > 0;
  });

  const labPage = $('.lab-page');
  if (!labPage) return;
  const labId = labPage.dataset.lab;
  const form = $('#processForm');
  const fileInput = $('#imageInput');
  const drop = $('#dropZone');
  const sample = $('#sampleButton');
  const chosenFile = $('#chosenFile');
  const chosenFileName = $('#chosenFileName');
  const removeFile = $('#removeFile');
  const runButton = $('#runButton');
  const mode = $('#mode');
  const inputType = $('#inputType');
  let currentFile = null;
  let activeCameraStream = null;

  function chooseFile(file) {
    if (!file) return;
    if (!file.type.startsWith('image/')) { showError('Please select a valid image file.'); return; }
    currentFile = file;
    chosenFile.hidden = false;
    chosenFileName.textContent = file.name;
    drop.querySelector('strong').textContent = 'Image added successfully!';
    drop.querySelector('small').textContent = 'Click to select a different image';
  }
  fileInput?.addEventListener('change', () => chooseFile(fileInput.files[0]));
  removeFile?.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    chosenFile.hidden = true;
    drop.querySelector('strong').textContent = 'Drop your image here';
    drop.querySelector('small').textContent = 'or click to browse files';
  });
  ['dragenter', 'dragover'].forEach(evt => drop?.addEventListener(evt, e => { e.preventDefault(); drop.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(evt => drop?.addEventListener(evt, e => { e.preventDefault(); drop.classList.remove('drag-over'); }));
  drop?.addEventListener('drop', e => chooseFile(e.dataTransfer?.files?.[0]));
  sample?.addEventListener('click', async () => {
    try {
      const response = await fetch('/static/assets/sample-flower.png');
      if (!response.ok) throw new Error('Sample file not found.');
      const blob = await response.blob();
      chooseFile(new File([blob], 'sample-flower.png', { type: 'image/png' }));
    } catch (error) { showError('Could not load the included flower sample. ' + error.message); }
  });
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const output = document.getElementById(slider.id + 'Value');
    if (!output) return;
    slider.addEventListener('input', () => { output.textContent = slider.value + (slider.dataset.unit || ''); });
  });
  function updateMode() {
    if (!mode) return;
    const lfw = mode.value === 'faces';
    $('#uploadWrapper').hidden = lfw;
    if ($('#components')) {
      const s = $('#components');
      s.max = lfw ? '150' : '150';
      s.value = lfw ? '150' : '28';
      $('#componentsValue').textContent = s.value;
    }
  }
  mode?.addEventListener('change', updateMode);
  function updateInputType() {
    if (!inputType) return;
    const video = inputType.value === 'video';
    $('#uploadWrapper').hidden = video;
    $('#videoWrapper').hidden = !video;
  }
  inputType?.addEventListener('change', updateInputType);
  $('#videoInput')?.addEventListener('change', e => { $('#videoName').textContent = e.target.files[0]?.name || ''; });

  const cameraModal = $('#cameraModal');
  const cameraPreview = $('#cameraPreview');
  function closeCamera() {
    activeCameraStream?.getTracks().forEach(track => track.stop());
    activeCameraStream = null;
    if (cameraPreview) cameraPreview.srcObject = null;
    cameraModal.hidden = true;
  }
  $('#cameraButton')?.addEventListener('click', async () => {
    cameraModal.hidden = false;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not allow camera access here. Use localhost or HTTPS.');
      activeCameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
      cameraPreview.srcObject = activeCameraStream;
    } catch (error) { closeCamera(); showError('Camera unavailable: ' + error.message); }
  });
  $('#cameraClose')?.addEventListener('click', closeCamera);
  cameraModal?.addEventListener('click', e => { if (e.target === cameraModal) closeCamera(); });
  $('#cameraCapture')?.addEventListener('click', async () => {
    const canvas = $('#cameraCanvas');
    canvas.width = cameraPreview.videoWidth || 640;
    canvas.height = cameraPreview.videoHeight || 480;
    canvas.getContext('2d').drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .92));
    if (blob) chooseFile(new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }));
    closeCamera();
  });
  window.addEventListener('beforeunload', closeCamera);

  function showError(message) {
    $('#emptyResults').hidden = true;
    $('#loadingResults').hidden = true;
    $('#renderedResults').hidden = true;
    const el = $('#errorResults');
    el.hidden = false;
    el.textContent = '✿ ' + message;
    $('#resultsStatus').textContent = 'PLEASE CHECK INPUT';
  }
  function showLoading() {
    $('#emptyResults').hidden = true;
    $('#errorResults').hidden = true;
    $('#renderedResults').hidden = true;
    $('#loadingResults').hidden = false;
    $('#resultsStatus').textContent = 'PROCESSING...';
    runButton.disabled = true;
    runButton.querySelector('.run-arrow').textContent = '◌';
  }
  function makeElement(tag, cls, text) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function showResults(data) {
    const stats = $('#resultStats');
    const grid = $('#outputGrid');
    stats.replaceChildren(); grid.replaceChildren();
    (data.stats || []).forEach(s => {
      const tile = makeElement('div','result-stat');
      tile.append(makeElement('b','',s.value),makeElement('span','',s.label));
      stats.append(tile);
    });
    (data.outputs || []).forEach((o, i) => {
      const card = makeElement('article','output-card');
      const header = makeElement('div','output-card-header');
      const title = makeElement('h3','',o.label);
      const download = makeElement('a','download-result','↓ Save');
      download.href = o.src;
      download.download = `visionlab-${labId}-${String(i+1).padStart(2,'0')}.png`;
      download.setAttribute('aria-label', 'Download ' + o.label);
      header.append(title,download);
      const image = makeElement('img');
      image.src = o.src; image.loading = 'lazy'; image.alt = o.label + ' visualization';
      card.append(header,image);
      if (o.note) card.append(makeElement('p','output-note',o.note));
      grid.append(card);
    });
    if (data.video) {
      const card = makeElement('div','result-video');
      const video = makeElement('video');
      video.src = data.video; video.controls = true; video.playsInline = true;
      const download = makeElement('a','', '↓ Download processed MP4');
      download.href = data.video; download.download = 'visionlab-yolo-result.mp4';
      card.append(video,download); grid.append(card);
    }
    $('#resultNote').textContent = data.note || 'Observe the differences between the input and the processed outputs.';
    $('#errorResults').hidden = true;
    $('#loadingResults').hidden = true;
    $('#renderedResults').hidden = false;
    $('#resultsStatus').textContent = '✦ EXPERIMENT COMPLETE';
  }
  form?.addEventListener('submit', async e => {
    e.preventDefault();
    const useLfw = labId === '5' && mode?.value === 'faces';
    const useVideo = labId === '6' && inputType?.value === 'video';
    const videoFile = $('#videoInput')?.files?.[0];
    if (!useLfw && !useVideo && !currentFile) return showError('Choose an image first, or click “Try sample flower”.');
    if (useVideo && !videoFile) return showError('Choose a short MP4 video first.');
    const body = new FormData(form);
    if (currentFile && !useVideo && !useLfw) body.set('image',currentFile,currentFile.name);
    if (useLfw || useVideo) body.delete('image');
    if (useVideo) body.set('video',videoFile,videoFile.name);
    showLoading();
    try {
      const response = await fetch('/api/process/' + labId, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Experiment could not be completed.');
      showResults(data);
    } catch (error) {
      showError(error instanceof TypeError ? 'Cannot connect to the Python server. Start it with “python app.py”, then reload the website.' : error.message);
    } finally {
      runButton.disabled = false;
      runButton.querySelector('.run-arrow').textContent = '↗';
    }
  });
})();
