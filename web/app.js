const API = '/api';
const POLL_INTERVAL = 3000;

let modelsData = [];
let statusData = { backend: null, clients: {} };
let lastLogCount = 0;

// --- Polling ---

async function fetchJSON(path) {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
}

async function pollModels() {
    try {
        modelsData = await fetchJSON('/models');
        renderModels(modelsData);
        renderModelSelect(modelsData);
    } catch (e) { /* ignore poll errors */ }
}

async function pollStatus() {
    try {
        statusData = await fetchJSON('/status');
        renderSession(statusData);
    } catch (e) { /* ignore poll errors */ }
}

async function pollTools() {
    try {
        const tools = await fetchJSON('/tools');
        renderTools(tools);
    } catch (e) { /* ignore poll errors */ }
}

async function pollLogs() {
    try {
        const lines = await fetchJSON('/logs');
        renderLogs(lines);
    } catch (e) { /* ignore poll errors */ }
}

// --- Actions ---

async function startServe() {
    const model = document.getElementById('model-select').value;
    const backend = document.getElementById('backend-select').value;
    if (!model) return;

    await fetch(API + '/serve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, backend: backend || null }),
    });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function stopServe() {
    await fetch(API + '/serve', { method: 'DELETE' });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function downloadModel(name) {
    await fetch(API + `/models/${name}/download`, { method: 'POST' });
    setTimeout(pollModels, 500);
    setTimeout(pollLogs, 500);
}

async function removeModel(name) {
    await fetch(API + `/models/${name}`, { method: 'DELETE' });
    setTimeout(pollModels, 500);
}

async function setupTool(name) {
    await fetch(API + `/tools/${name}/setup`, { method: 'POST' });
    setTimeout(pollTools, 1000);
    setTimeout(pollLogs, 500);
}

async function startClient(name) {
    await fetch(API + `/clients/${name}`, { method: 'POST' });
    setTimeout(pollStatus, 500);
}

async function stopClient(name) {
    await fetch(API + `/clients/${name}`, { method: 'DELETE' });
    setTimeout(pollStatus, 500);
}

// --- Rendering ---

function renderModels(models) {
    const el = document.getElementById('models-content');
    if (!models.length) {
        el.innerHTML = '<span class="nothing-running">No models configured.</span>';
        return;
    }

    let html = '<table><tr><th>Model</th><th>Format</th><th>Status</th><th>Size</th><th></th></tr>';
    for (const m of models) {
        const statusClass = m.downloaded ? 'status-downloaded' : 'status-not-downloaded';
        const statusText = m.downloaded ? 'downloaded' : 'not downloaded';
        const actions = m.downloaded
            ? `<button class="btn btn-sm btn-danger" onclick="removeModel('${m.name}')">Remove</button>`
            : `<button class="btn btn-sm btn-primary" onclick="downloadModel('${m.name}')">Download</button>`;
        html += `<tr>
            <td>${m.name}</td>
            <td>${m.format}</td>
            <td class="${statusClass}">${statusText}</td>
            <td>${m.size_human}</td>
            <td>${actions}</td>
        </tr>`;
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderModelSelect(models) {
    const sel = document.getElementById('model-select');
    const current = sel.value;
    sel.innerHTML = '<option value="">-- select model --</option>';
    for (const m of models) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        sel.appendChild(opt);
    }
    if (current) sel.value = current;
    updateBackendSelect();
}

function updateBackendSelect() {
    const modelName = document.getElementById('model-select').value;
    const sel = document.getElementById('backend-select');
    sel.innerHTML = '<option value="">-- default --</option>';

    const model = modelsData.find(m => m.name === modelName);
    if (model) {
        for (const b of model.backends) {
            const opt = document.createElement('option');
            opt.value = b;
            opt.textContent = b + (b === model.default_backend ? ' (default)' : '');
            sel.appendChild(opt);
        }
    }
}

function renderSession(status) {
    const el = document.getElementById('session-content');

    if (!status.backend && Object.keys(status.clients).length === 0) {
        el.innerHTML = '<span class="nothing-running">Nothing running.</span>';
        return;
    }

    let html = '';
    if (status.backend) {
        const b = status.backend;
        const dotClass = b.health === 'healthy' ? 'dot-healthy' : 'dot-unhealthy';
        const healthClass = b.health === 'healthy' ? 'status-healthy' : 'status-unhealthy';
        html += `<div class="session-info">
            <div><span class="label">Backend:</span> ${b.name}</div>
            <div><span class="label">Model:</span> ${b.model}</div>
            <div><span class="label">URL:</span> <a href="http://localhost:${b.port}/v1" style="color:#5e7ce2">http://localhost:${b.port}/v1</a></div>
            <div><span class="label">Health:</span> <span class="dot ${dotClass}"></span><span class="${healthClass}">${b.health}</span></div>
            <div><span class="label">Uptime:</span> ${b.uptime}</div>
        </div>`;
    }

    const clientNames = Object.keys(status.clients);
    if (clientNames.length > 0) {
        html += '<h3>Clients</h3><table><tr><th>Name</th><th>Port</th><th>Uptime</th><th></th></tr>';
        for (const name of clientNames) {
            const c = status.clients[name];
            html += `<tr>
                <td>${name}</td>
                <td>${c.port || '-'}</td>
                <td>${c.uptime}</td>
                <td><button class="btn btn-sm btn-danger" onclick="stopClient('${name}')">Stop</button></td>
            </tr>`;
        }
        html += '</table>';
    }

    el.innerHTML = html;
}

function renderTools(tools) {
    const el = document.getElementById('tools-content');
    if (!tools.length) {
        el.innerHTML = '<span class="nothing-running">No tools configured.</span>';
        return;
    }

    let html = '<table><tr><th>Tool</th><th>Type</th><th>Status</th><th></th></tr>';
    for (const t of tools) {
        const statusClass = 'status-' + t.status.replace(' ', '-');
        const actions = (t.status === 'not installed')
            ? `<button class="btn btn-sm btn-primary" onclick="setupTool('${t.name}')">Setup</button>`
            : '';
        html += `<tr>
            <td>${t.name}</td>
            <td>${t.install_type}</td>
            <td class="${statusClass}">${t.status}</td>
            <td>${actions}</td>
        </tr>`;
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderLogs(lines) {
    const el = document.getElementById('log-output');
    const shouldScroll = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    el.textContent = lines.join('\n');
    if (shouldScroll) {
        el.scrollTop = el.scrollHeight;
    }
}

// --- Init ---

document.getElementById('model-select').addEventListener('change', updateBackendSelect);

pollModels();
pollStatus();
pollTools();
pollLogs();

setInterval(pollModels, POLL_INTERVAL);
setInterval(pollStatus, POLL_INTERVAL);
setInterval(pollTools, POLL_INTERVAL * 3);  // tools change less frequently
setInterval(pollLogs, POLL_INTERVAL);
