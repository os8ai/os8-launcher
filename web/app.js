const API = '/api';
const POLL_INTERVAL = 3000;

let modelsData = [];
let toolsData = [];
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
        toolsData = await fetchJSON('/tools');
        renderBackends(toolsData.filter(t => t.kind === 'backend'));
        renderClients(toolsData.filter(t => t.kind === 'client'));
        renderClientSelect(toolsData);
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
    const client = document.getElementById('client-select').value;
    if (!model) return;

    await fetch(API + '/serve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, backend: backend || null }),
    });

    // If a container client was selected, launch it after a delay (backend needs to start)
    if (client) {
        const tool = toolsData.find(t => t.name === client);
        if (tool && tool.install_type === 'container') {
            // Poll until backend is healthy, then start client
            setTimeout(() => launchClientWhenReady(client), 2000);
        }
    }

    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function launchClientWhenReady(name) {
    // Check if backend is running yet
    try {
        const status = await fetchJSON('/status');
        if (status.backend && status.backend.health === 'healthy') {
            await fetch(API + `/clients/${name}`, { method: 'POST' });
            setTimeout(pollStatus, 500);
        } else {
            // Not ready yet, try again
            setTimeout(() => launchClientWhenReady(name), 3000);
        }
    } catch (e) {
        setTimeout(() => launchClientWhenReady(name), 3000);
    }
}

async function stopServe() {
    await fetch(API + '/serve', { method: 'DELETE' });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function downloadModel(name) {
    await fetch(API + `/models/${name}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: 'vllm' }),
    });
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
    const current = sel.value;
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
    if (current) sel.value = current;
}

function renderClientSelect(tools) {
    const sel = document.getElementById('client-select');
    const current = sel.value;
    sel.innerHTML = '<option value="">-- none --</option>';

    const clients = tools.filter(t => t.kind === 'client');
    for (const c of clients) {
        const opt = document.createElement('option');
        opt.value = c.name;
        let label = c.name;
        if (c.install_type === 'pip') label += ' (terminal)';
        if (c.install_type === 'bridge') label += ' (bridge)';
        opt.textContent = label;
        sel.appendChild(opt);
    }
    if (current) sel.value = current;
    updateClientHint();
}

function updateClientHint() {
    const clientName = document.getElementById('client-select').value;
    const hint = document.getElementById('client-hint');

    if (!clientName) {
        hint.style.display = 'none';
        return;
    }

    const tool = toolsData.find(t => t.name === clientName);
    if (!tool) {
        hint.style.display = 'none';
        return;
    }

    if (tool.install_type === 'pip') {
        hint.style.display = 'block';
        hint.innerHTML = `<strong>${clientName}</strong> is a terminal app. After serving, run in your terminal:<br><code>./launcher client ${clientName}</code>`;
    } else if (tool.install_type === 'bridge') {
        hint.style.display = 'block';
        hint.innerHTML = `<strong>${clientName}</strong> connects directly to the backend API. No launch needed.`;
    } else if (tool.install_type === 'container') {
        hint.style.display = 'block';
        hint.innerHTML = `<strong>${clientName}</strong> will launch automatically after the backend is ready.`;
    } else {
        hint.style.display = 'none';
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

function renderBackends(backends) {
    const el = document.getElementById('backends-content');
    if (!backends.length) {
        el.innerHTML = '<span class="nothing-running">No backends configured.</span>';
        return;
    }

    let html = '<table><tr><th>Name</th><th>Type</th><th>Status</th><th></th></tr>';
    for (const t of backends) {
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

function renderClients(clients) {
    const el = document.getElementById('clients-content');
    if (!clients.length) {
        el.innerHTML = '<span class="nothing-running">No clients configured.</span>';
        return;
    }

    let html = '<table><tr><th>Name</th><th>Type</th><th>Status</th><th></th></tr>';
    for (const t of clients) {
        const statusClass = 'status-' + t.status.replace(' ', '-');
        const actions = (t.status === 'not installed' )
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

// --- Settings / Credentials ---

let credentialsData = { ngc: false, hf: false };

async function pollCredentials() {
    try {
        credentialsData = await fetchJSON('/credentials');
    } catch (e) { /* ignore */ }
}

function openSettings() {
    document.getElementById('settings-modal').style.display = 'flex';
    document.getElementById('ngc-key-input').value = '';
    document.getElementById('hf-token-input').value = '';
    updateCredentialStatus();
}

function closeSettings() {
    document.getElementById('settings-modal').style.display = 'none';
}

function updateCredentialStatus() {
    const ngcStatus = document.getElementById('ngc-status');
    const hfStatus = document.getElementById('hf-status');

    if (credentialsData.ngc) {
        ngcStatus.textContent = 'configured';
        ngcStatus.className = 'key-status configured';
    } else {
        ngcStatus.textContent = 'not set';
        ngcStatus.className = 'key-status missing';
    }

    if (credentialsData.hf) {
        hfStatus.textContent = 'configured';
        hfStatus.className = 'key-status configured';
    } else {
        hfStatus.textContent = 'not set';
        hfStatus.className = 'key-status missing';
    }
}

async function saveCredentials() {
    const ngcKey = document.getElementById('ngc-key-input').value.trim();
    const hfToken = document.getElementById('hf-token-input').value.trim();

    const body = {};
    if (ngcKey) body.ngc_api_key = ngcKey;
    if (hfToken) body.hf_token = hfToken;

    if (Object.keys(body).length === 0) {
        closeSettings();
        return;
    }

    const res = await fetch(API + '/credentials', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    credentialsData = await res.json();
    updateCredentialStatus();
    closeSettings();
}

// Close modal on backdrop click
document.getElementById('settings-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeSettings();
});

// --- Init ---

document.getElementById('model-select').addEventListener('change', updateBackendSelect);
document.getElementById('client-select').addEventListener('change', updateClientHint);

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab).classList.add('active');
    });
});

pollModels();
pollStatus();
pollTools();
pollLogs();
pollCredentials();

setInterval(pollModels, POLL_INTERVAL);
setInterval(pollStatus, POLL_INTERVAL);
setInterval(pollTools, POLL_INTERVAL * 3);  // tools change less frequently
setInterval(pollLogs, POLL_INTERVAL);
