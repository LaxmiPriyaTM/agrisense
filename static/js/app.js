/**
 * AgriSense — Frontend JavaScript
 * Handles image upload, API calls, results display, and chatbot
 */

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
let chatHistory = [];
let currentFile = null;
const API_BASE = window.location.origin;

// Disease severity to emoji mapping
const SEVERITY_ICON = {
  NONE: '✅', LOW: '🟡', MEDIUM: '🟠', HIGH: '🔴', CRITICAL: '🚨'
};

// ─────────────────────────────────────────────
// ON LOAD
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initUploadZone();
  checkHealth();
  loadHistory();
  
  // Enter key for chat
  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
});

// ─────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    
    const dot = document.querySelector('.status-dot');
    const txt = document.getElementById('statusText');
    
    if (data.status === 'healthy') {
      dot.classList.add('online');
      txt.textContent = 'API Online';
    }
  } catch {
    document.getElementById('statusText').textContent = 'API Offline';
  }
}

// ─────────────────────────────────────────────
// UPLOAD ZONE
// ─────────────────────────────────────────────
function initUploadZone() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('imageInput');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const changeBtn = document.getElementById('changeImageBtn');

  // Click to upload
  zone.addEventListener('click', (e) => {
    if (e.target !== changeBtn) input.click();
  });

  // File selection
  input.addEventListener('change', () => {
    if (input.files[0]) handleFileSelect(input.files[0]);
  });

  // Drag and drop
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragging'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault(); zone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
  });

  // Change image button
  changeBtn.addEventListener('click', (e) => { e.stopPropagation(); input.click(); });

  // Analyze button
  analyzeBtn.addEventListener('click', runAnalysis);
}

function handleFileSelect(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Please upload an image file (JPG, PNG, WEBP)', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File size must be under 10MB', 'error');
    return;
  }

  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById('previewImage');
    const placeholder = document.getElementById('uploadPlaceholder');
    const overlay = document.getElementById('previewOverlay');

    preview.src = e.target.result;
    preview.classList.remove('hidden');
    placeholder.classList.add('hidden');
    overlay.classList.remove('hidden');
    document.getElementById('analyzeBtn').disabled = false;
  };
  reader.readAsDataURL(file);
  resetResults();
}

// ─────────────────────────────────────────────
// MAIN ANALYSIS
// ─────────────────────────────────────────────
async function runAnalysis() {
  if (!currentFile) return;

  const preference = document.querySelector('input[name="preference"]:checked')?.value || 'balanced';
  
  // Show loading
  showLoading();
  document.getElementById('analyzeBtn').disabled = true;

  // Animate loading steps
  const steps = ['step1', 'step2', 'step3'];
  for (let i = 0; i < steps.length; i++) {
    await delay(600);
    document.getElementById(steps[i]).classList.add('active');
    if (i > 0) document.getElementById(steps[i-1]).classList.remove('active');
    document.getElementById(steps[i-1 >= 0 ? steps[i-1] : steps[0]]).classList.add('done');
  }

  try {
    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('get_advisory', 'true');
    formData.append('farmer_preference', preference);

    const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!data.success) throw new Error(data.error || 'Analysis failed');
    
    displayResults(data);
    await loadHistory();

  } catch (err) {
    showError(err.message);
  } finally {
    document.getElementById('analyzeBtn').disabled = false;
  }
}

// ─────────────────────────────────────────────
// DISPLAY RESULTS
// ─────────────────────────────────────────────
function displayResults(data) {
  const pred = data.prediction;
  const primary = pred.primary;
  const isHealthy = primary.is_healthy;

  // Hide loading, show results
  document.getElementById('loadingState').classList.add('hidden');
  document.getElementById('resultsContent').classList.remove('hidden');

  // Disease Badge
  document.getElementById('badgeIcon').textContent = isHealthy ? '✅' : '🦠';
  document.getElementById('cropName').textContent = primary.crop;
  document.getElementById('diseaseName').textContent = primary.disease;

  // Severity chip
  const advisory = data.advisory;
  let severity = 'MEDIUM';
  if (isHealthy) severity = 'NONE';
  else if (advisory?.decisions?.alert_level) severity = advisory.decisions.alert_level;
  
  const chip = document.getElementById('severityChip');
  chip.textContent = `${SEVERITY_ICON[severity] || ''} ${severity}`;
  chip.className = `severity-chip severity-${severity}`;

  // Confidence bar
  const conf = primary.confidence;
  document.getElementById('confidenceValue').textContent = primary.confidence_formatted;
  document.getElementById('confidenceFill').style.width = `${conf}%`;
  document.getElementById('confidenceFill').style.background = 
    conf >= 90 ? 'linear-gradient(90deg,#16a34a,#4ade80)' :
    conf >= 70 ? 'linear-gradient(90deg,#b45309,#fbbf24)' :
                 'linear-gradient(90deg,#b91c1c,#f87171)';
  document.getElementById('confidenceMsg').textContent = pred.confidence_message;

  // Top 3 predictions
  const list = document.getElementById('predictionsList');
  list.innerHTML = pred.top_3.map((p, i) => `
    <div class="prediction-item">
      <div class="prediction-rank">${i + 1}</div>
      <div class="prediction-name">${p.crop} — ${p.disease}</div>
      <div class="prediction-bar">
        <div class="prediction-bar-fill" style="width:${p.confidence}%"></div>
      </div>
      <div class="prediction-pct">${p.confidence.toFixed(1)}%</div>
    </div>
  `).join('');

  // Advisory
  if (advisory?.advisory) {
    const advisoryText = document.getElementById('advisoryText');
    advisoryText.innerHTML = markdownToHtml(advisory.advisory);
    document.getElementById('advisorySection').classList.remove('hidden');
    // Auto-open advisory
    document.getElementById('advisoryContent').classList.remove('hidden');
    document.getElementById('advisoryToggle').classList.add('open');
  }

  // Auto-scroll to results
  document.getElementById('resultsContent').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─────────────────────────────────────────────
// UI STATE HELPERS
// ─────────────────────────────────────────────
function showLoading() {
  document.getElementById('resultsPlaceholder').classList.add('hidden');
  document.getElementById('resultsContent').classList.add('hidden');
  document.getElementById('loadingState').classList.remove('hidden');
  
  // Reset step states
  ['step1','step2','step3'].forEach(id => {
    document.getElementById(id).classList.remove('active','done');
  });
  document.getElementById('step1').classList.add('active');
}

function resetResults() {
  document.getElementById('resultsContent').classList.add('hidden');
  document.getElementById('loadingState').classList.add('hidden');
  document.getElementById('resultsPlaceholder').classList.remove('hidden');
}

function showError(msg) {
  document.getElementById('loadingState').classList.add('hidden');
  document.getElementById('resultsPlaceholder').classList.remove('hidden');
  document.getElementById('resultsPlaceholder').innerHTML = `
    <div class="placeholder-icon">❌</div>
    <h3 style="color:#f87171">Analysis Failed</h3>
    <p style="color:#6b7280">${msg}</p>
    <button class="btn btn-secondary" style="margin-top:1rem" onclick="resetResults()">Try Again</button>
  `;
}

function toggleAdvisory() {
  const content = document.getElementById('advisoryContent');
  const icon = document.getElementById('advisoryToggle');
  content.classList.toggle('hidden');
  icon.classList.toggle('open');
}

// ─────────────────────────────────────────────
// CHATBOT
// ─────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  input.value = '';
  input.disabled = true;
  document.getElementById('sendBtn').disabled = true;
  document.getElementById('quickReplies').classList.add('hidden');

  appendBubble(message, 'user');
  const typingEl = appendTyping();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: chatHistory })
    });
    const data = await res.json();

    typingEl.remove();
    if (data.success) {
      chatHistory = data.history;
      appendBubble(data.reply, 'bot');
    } else {
      appendBubble('Sorry, I encountered an error. Please try again.', 'bot');
    }
  } catch {
    typingEl.remove();
    appendBubble('Connection error. Please check if the server is running.', 'bot');
  } finally {
    input.disabled = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }
}

function sendQuick(msg) {
  document.getElementById('chatInput').value = msg;
  sendMessage();
}

function appendBubble(text, role) {
  const messages = document.getElementById('chatMessages');
  const isBot = role === 'bot';
  
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${isBot ? 'bot' : 'user'}-bubble`;
  bubble.innerHTML = `
    <div class="bubble-avatar">${isBot ? '🌾' : '👤'}</div>
    <div class="bubble-content">${isBot ? markdownToHtml(text) : escapeHtml(text)}</div>
  `;
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function appendTyping() {
  const messages = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'chat-bubble bot-bubble';
  el.innerHTML = `
    <div class="bubble-avatar">🌾</div>
    <div class="bubble-content bubble-typing">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>
  `;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

// ─────────────────────────────────────────────
// SCAN HISTORY
// ─────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`);
    const data = await res.json();
    renderHistory(data.history || []);
  } catch { /* silent fail */ }
}

function renderHistory(history) {
  const list = document.getElementById('historyList');
  if (!history.length) {
    list.innerHTML = '<p class="no-history">No scans yet. Upload a leaf to get started!</p>';
    return;
  }
  list.innerHTML = history.map(item => `
    <div class="history-item">
      <div class="history-icon">${item.is_healthy ? '✅' : '🦠'}</div>
      <div class="history-info">
        <div class="history-disease">${item.disease}</div>
        <div class="history-crop">${item.crop} • ${formatTime(item.timestamp)}</div>
      </div>
      <div class="history-conf">${item.confidence.toFixed(0)}%</div>
    </div>
  `).join('');
}

async function clearHistory() {
  try {
    await fetch(`${API_BASE}/clear-history`, { method: 'POST' });
    renderHistory([]);
  } catch { /* silent */ }
}

// ─────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────
function markdownToHtml(md) {
  if (!md) return '';
  return md
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/## (.*?)$/gm, '<h2>$1</h2>')
    .replace(/### (.*?)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^\d+\. (.*?)$/gm, '<li>$1</li>')
    .replace(/^- (.*?)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[hul])/gm, '')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function formatTime(isoStr) {
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function showToast(msg, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed;bottom:2rem;right:2rem;z-index:9999;
    padding:0.8rem 1.2rem;border-radius:10px;font-size:0.85rem;font-weight:500;
    background:${type==='error'?'#7f1d1d':'#14532d'};
    color:${type==='error'?'#fca5a5':'#86efac'};
    border:1px solid ${type==='error'?'rgba(248,113,113,0.3)':'rgba(74,222,128,0.3)'};
    animation:fadeIn 0.3s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}
