/**
 * NeuroMotion - Diagnostic Controller
 * Minimal, clean, and professional API integration
 */

const state = {
  models: [],
  selectedModelId: 'catboost',
  children: [],
  selectedChildId: null,
  currentThreshold: 0.63,
  selectedFile: null,
};

// Subtle monochrome/clean icons
const ICONS = {
  check: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  alert: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
};

document.addEventListener('DOMContentLoaded', async () => {
  setupDropZone();
  await loadModels();
  await loadChildren();
  await loadBenchmark();

  // Run initial diagnostic assessment with first child
  if (state.selectedChildId) {
    runDiagnosis();
  }
});

// =====================================================================
// MODEL SELECTION
// =====================================================================
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    state.models = data.models || [];
    renderModelCards();
  } catch (err) {
    console.error('Failed to load models:', err);
  }
}

function renderModelCards() {
  const container = document.getElementById('modelGrid');
  if (!container) return;

  container.innerHTML = state.models.map(m => {
    const isSelected = m.id === state.selectedModelId;
    return `
      <div class="model-option-card ${isSelected ? 'selected' : ''}" 
           data-model-id="${m.id}" 
           onclick="selectModel('${m.id}')">
        <div>
          <div class="model-header-row">
            <span class="model-name">${m.name}</span>
            <span class="model-tag">${m.tag}</span>
          </div>
          <div class="model-type">${m.type}</div>
        </div>
        <div class="model-stats-row">
          <span class="stat-pill">Accuracy: <strong>${m.accuracy}</strong></span>
          <span class="stat-pill">AUC: <strong>${m.auc}</strong></span>
        </div>
      </div>
    `;
  }).join('');
}

function selectModel(modelId) {
  state.selectedModelId = modelId;
  renderModelCards();
  if (state.selectedChildId) {
    runDiagnosis();
  }
}

// =====================================================================
// COHORT & SUBJECT FILTERING
// =====================================================================
async function loadChildren() {
  try {
    const res = await fetch('/api/children');
    const data = await res.json();
    state.children = data.children || [];
    filterChildrenList();
  } catch (err) {
    console.error('Failed to load cohort:', err);
  }
}

function filterChildrenList() {
  const filterVal = document.getElementById('splitFilter').value;
  const selectElem = document.getElementById('childSelect');
  if (!selectElem) return;

  let filtered = state.children;
  if (filterVal === 'test') {
    filtered = state.children.filter(c => c.is_test);
  } else if (filterVal === 'asd') {
    filtered = state.children.filter(c => c.group === 'ASD');
  } else if (filterVal === 'td') {
    filtered = state.children.filter(c => c.group === 'TD');
  }

  selectElem.innerHTML = filtered.map(c => {
    const splitTag = c.is_test ? 'Held-Out Test' : 'Train Cohort';
    return `<option value="${c.id}">${c.id} (${c.group}, ${splitTag})</option>`;
  }).join('');

  if (filtered.length > 0) {
    const exists = filtered.find(c => c.id === state.selectedChildId);
    state.selectedChildId = exists ? exists.id : filtered[0].id;
    selectElem.value = state.selectedChildId;
  } else {
    state.selectedChildId = null;
  }
}

function onChildSelected() {
  const selectElem = document.getElementById('childSelect');
  state.selectedChildId = selectElem.value;
}

function selectRandomChild() {
  const selectElem = document.getElementById('childSelect');
  const options = selectElem.options;
  if (options.length === 0) return;

  const randomIndex = Math.floor(Math.random() * options.length);
  selectElem.selectedIndex = randomIndex;
  state.selectedChildId = options[randomIndex].value;
  runDiagnosis();
}

function updateThreshold(val) {
  const numVal = parseFloat(val);
  state.currentThreshold = numVal;

  const displayElem = document.getElementById('thresholdDisplay');
  if (displayElem) {
    displayElem.textContent = numVal.toFixed(2);
  }

  const marker = document.getElementById('gaugeMarker');
  if (marker) {
    marker.style.left = `${(numVal * 100).toFixed(1)}%`;
    const label = marker.querySelector('.pin-label');
    if (label) label.textContent = `θ ${numVal.toFixed(2)}`;
  }
}

// =====================================================================
// TAB NAVIGATION
// =====================================================================
function switchInputTab(tab) {
  const tabCohort = document.getElementById('tabCohort');
  const tabUpload = document.getElementById('tabUpload');
  const cohortContent = document.getElementById('cohortContent');
  const uploadContent = document.getElementById('uploadContent');

  if (tab === 'cohort') {
    tabCohort.classList.add('active');
    tabUpload.classList.remove('active');
    cohortContent.classList.remove('hidden');
    uploadContent.classList.add('hidden');
  } else {
    tabCohort.classList.remove('active');
    tabUpload.classList.add('active');
    cohortContent.classList.add('hidden');
    uploadContent.classList.remove('hidden');
  }
}

// =====================================================================
// FILE UPLOAD HANDLING
// =====================================================================
function setupDropZone() {
  const dropZone = document.getElementById('dropZone');
  if (!dropZone) return;

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--text-secondary)';
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--border-muted)';
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) processSelectedFile(files[0]);
  });
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) processSelectedFile(files[0]);
}

function processSelectedFile(file) {
  state.selectedFile = file;
  const infoElem = document.getElementById('fileSelectedInfo');
  if (infoElem) {
    infoElem.classList.remove('hidden');
    infoElem.innerHTML = `
      <span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
      <button class="btn btn-ghost" style="padding: 0.2rem 0.5rem; font-size: 0.72rem;" onclick="clearUploadedFile()">Remove</button>
    `;
  }
}

function clearUploadedFile() {
  state.selectedFile = null;
  const input = document.getElementById('fileInput');
  if (input) input.value = '';
  const infoElem = document.getElementById('fileSelectedInfo');
  if (infoElem) infoElem.classList.add('hidden');
}

// =====================================================================
// DIAGNOSTIC INFERENCE & RENDERING
// =====================================================================
async function runDiagnosis() {
  if (!state.selectedChildId) return;

  const runBtn = document.getElementById('runBtn');
  const origText = runBtn.textContent;
  runBtn.disabled = true;
  runBtn.textContent = 'Analyzing...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        child_id: state.selectedChildId,
        model_id: state.selectedModelId,
        threshold: state.currentThreshold
      })
    });

    const data = await res.json();
    if (!res.ok) {
      alert(`Diagnostic Error: ${data.error || 'Server error'}`);
      return;
    }

    renderDiagnosisResults(data);
  } catch (err) {
    console.error('Diagnosis request failed:', err);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = origText;
  }
}

function renderDiagnosisResults(data) {
  const isASD = data.predicted_label === 1;

  // 1. Subject Header
  document.getElementById('resChildId').textContent = data.child_id;
  const splitPill = document.getElementById('resSplitBadge');
  if (splitPill) {
    splitPill.textContent = data.is_test_set ? 'Held-Out Test Set (Unseen)' : 'Training / Validation Cohort';
  }

  // 2. Verdict Banner & Icon
  const verdictBanner = document.getElementById('verdictBadge');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictName = document.getElementById('resVerdictName');

  if (isASD) {
    verdictBanner.className = 'verdict-banner asd-positive';
    verdictIcon.innerHTML = ICONS.alert;
    verdictName.textContent = 'Autism Spectrum Disorder (ASD)';
  } else {
    verdictBanner.className = 'verdict-banner td-normal';
    verdictIcon.innerHTML = ICONS.check;
    verdictName.textContent = 'Typically Developing Control (TD)';
  }

  // 3. Confidence & Probability
  document.getElementById('resConfidence').textContent = `${data.confidence_pct}%`;
  document.getElementById('resProb').textContent = `${data.mean_probability.toFixed(3)}`;
  document.getElementById('resVolatility').textContent = `± ${data.volatility_std.toFixed(3)}`;

  // 4. Probability Track
  const gaugeFill = document.getElementById('gaugeFill');
  const pctWidth = Math.min(Math.max(data.mean_probability * 100, 2), 100);
  gaugeFill.style.width = `${pctWidth}%`;

  const gaugeMarker = document.getElementById('gaugeMarker');
  gaugeMarker.style.left = `${(data.threshold_used * 100).toFixed(1)}%`;
  const markerLabel = gaugeMarker.querySelector('.pin-label');
  if (markerLabel) markerLabel.textContent = `θ ${data.threshold_used.toFixed(2)}`;

  // 5. Ground Truth Check
  document.getElementById('resGroundTruth').textContent = `${data.true_diagnosis} (label=${data.true_label})`;
  const matchBadge = document.getElementById('resMatchBadge');
  if (data.is_correct) {
    matchBadge.className = 'match-status correct';
    matchBadge.textContent = 'Correct Diagnosis';
  } else {
    matchBadge.className = 'match-status mismatch';
    matchBadge.textContent = 'Discordant Diagnosis';
  }

  // 6. Multi-Model Consensus (Clean Table Rows)
  const consensusList = document.getElementById('consensusList');
  consensusList.innerHTML = (data.consensus || []).map(m => {
    const isSelected = (m.model_id === state.selectedModelId);
    const badgeClass = m.predicted_label === 1 ? 'asd' : 'td';
    return `
      <tr class="${isSelected ? 'active-model-row' : ''}">
        <td>${m.model_name} ${isSelected ? '<span style="color: #38bdf8; font-size: 0.7rem; font-weight: 500;">(Active)</span>' : ''}</td>
        <td class="mono">${m.probability.toFixed(3)}</td>
        <td><span class="badge ${badgeClass}">${m.diagnosis}</span></td>
      </tr>
    `;
  }).join('');

  // 7. 8-Trial Kinematic Consistency
  const trialsGrid = document.getElementById('trialsGrid');
  trialsGrid.innerHTML = (data.trials || []).map(t => {
    const trialIsASD = t.probability >= data.threshold_used;
    const heightPct = Math.min(Math.max(t.probability * 100, 6), 100);
    return `
      <div class="trial-item">
        <span class="trial-num">T${t.trial_num}</span>
        <div class="trial-bar-wrap" title="Trial ${t.trial_num}: P(ASD) = ${t.probability.toFixed(3)}">
          <div class="trial-bar-fill ${trialIsASD ? 'asd-fill' : 'td-fill'}" style="height: ${heightPct}%;"></div>
        </div>
        <span class="trial-val">${t.probability.toFixed(2)}</span>
      </div>
    `;
  }).join('');

  // 8. Top Biomechanical Feature Attributions
  const biomarkersBody = document.getElementById('biomarkersBody');
  biomarkersBody.innerHTML = (data.top_features || []).map(f => {
    return `
      <tr>
        <td class="mono">${f.feature}</td>
        <td class="mono">${f.value.toFixed(4)}</td>
        <td>${f.impact}</td>
        <td>${f.direction}</td>
      </tr>
    `;
  }).join('');
}

// Upload Diagnosis
async function runUploadDiagnosis() {
  if (!state.selectedFile) {
    alert('Please select a .xlsx or .csv movement data file first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', state.selectedFile);
  formData.append('model_id', state.selectedModelId);
  formData.append('threshold', state.currentThreshold);

  try {
    const res = await fetch('/api/upload_predict', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      alert(`File Evaluation Error: ${data.error || 'Server error'}`);
      return;
    }

    alert(`File Analysis Complete\n\nFile: ${data.filename} (${data.row_count} trials)\nModel: ${data.model_used}\nDiagnosis: ${data.predicted_diagnosis}\nMean P(ASD): ${data.mean_probability.toFixed(3)}\nConfidence: ${data.confidence_pct}%`);
  } catch (err) {
    console.error('Upload evaluation failed:', err);
    alert('Failed to evaluate uploaded file.');
  }
}

// =====================================================================
// BENCHMARK EVIDENCE TABLE
// =====================================================================
async function loadBenchmark() {
  try {
    const res = await fetch('/api/benchmark');
    const data = await res.json();
    const benchBody = document.getElementById('benchmarkBody');
    if (!benchBody) return;

    benchBody.innerHTML = (data.benchmarks || []).map(b => {
      const isTop = b.model.includes('CatBoost');
      return `
        <tr class="${isTop ? 'active-model-row' : ''}">
          <td>${b.model} ${isTop ? '<span style="color: #38bdf8; font-size: 0.7rem;">(Standard)</span>' : ''}</td>
          <td class="mono">${b.bacc}</td>
          <td class="mono">${b.sens}</td>
          <td class="mono">${b.spec}</td>
          <td class="mono">${b.f1}</td>
          <td class="mono">${b.auc}</td>
          <td class="mono">${b.ece}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}
