const API = '/api';
const POLL_INTERVAL = 3000;

// Active project + remembered selections. The server auto-creates a
// "default" project if none exists, so this never has to redirect.
let pendingProjectDefaults = null;  // {model, backend, client}

(async function loadActiveProject() {
    try {
        const r = await fetch('/api/projects');
        if (!r.ok) return;
        const data = await r.json();
        const indicator = document.getElementById('active-project-indicator');
        if (indicator && data.active) indicator.textContent = '· project: ' + data.active;
        const p = data.active_project;
        if (p) {
            pendingProjectDefaults = {
                model: p.last_model || '',
                backend: p.last_backend || '',
                client: p.last_client || '',
            };
        }
    } catch (e) { /* ignore — dashboard will still load */ }
})();

function applyProjectDefaults() {
    // Called from renderModelSelect / renderClientSelect after each
    // populates its <select>. We apply each piece once and then null it
    // out so user choices are never overwritten on later polls.
    if (!pendingProjectDefaults) return;
    const d = pendingProjectDefaults;
    const modelSel = document.getElementById('model-select');
    const clientSel = document.getElementById('client-select');
    if (d.model && modelSel && [...modelSel.options].some(o => o.value === d.model)) {
        modelSel.value = d.model;
        modelSel.dispatchEvent(new Event('change'));
        d.model = '';
    }
    const backendSel = document.getElementById('backend-select');
    if (d.backend && backendSel && [...backendSel.options].some(o => o.value === d.backend)) {
        backendSel.value = d.backend;
        d.backend = '';
    }
    if (d.client && clientSel && [...clientSel.options].some(o => o.value === d.client)) {
        clientSel.value = d.client;
        clientSel.dispatchEvent(new Event('change'));
        d.client = '';
    }
    if (!d.model && !d.backend && !d.client) pendingProjectDefaults = null;
}

function _showOverlay(html) {
    let overlay = document.getElementById('dashboard-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'dashboard-overlay';
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(15,15,20,0.85);color:#fff;display:flex;align-items:center;justify-content:center;font-family:sans-serif;z-index:9999;backdrop-filter:blur(4px)';
        document.body.appendChild(overlay);
    }
    overlay.innerHTML = `<div style="background:#1f2230;padding:2rem 2.5rem;border-radius:12px;max-width:480px;text-align:center;box-shadow:0 12px 40px rgba(0,0,0,0.5)">${html}</div>`;
    return overlay;
}

function _hideOverlay() {
    const overlay = document.getElementById('dashboard-overlay');
    if (overlay) overlay.remove();
}

async function stopDashboard() {
    const overlay = _showOverlay(`
        <h2 style="margin:0 0 0.75rem;font-size:1.4rem">Stop the dashboard server?</h2>
        <p style="margin:0 0 1.25rem;opacity:0.85;line-height:1.4">The dashboard process will exit completely. You'll need to run <code style="background:#000;padding:2px 6px;border-radius:4px">os8-launcher</code> in a terminal to bring it back.</p>
        <div style="display:flex;gap:0.75rem;justify-content:center">
            <button id="overlay-confirm" style="background:#c0392b;color:#fff;border:none;padding:0.6rem 1.2rem;border-radius:6px;font-size:1rem;cursor:pointer">Stop server</button>
            <button id="overlay-cancel" style="background:#444;color:#fff;border:none;padding:0.6rem 1.2rem;border-radius:6px;font-size:1rem;cursor:pointer">Cancel</button>
        </div>
    `);
    document.getElementById('overlay-cancel').onclick = _hideOverlay;
    document.getElementById('overlay-confirm').onclick = async () => {
        _showOverlay(`<h2 style="margin:0 0 0.75rem;font-size:1.4rem">Stopping…</h2><p style="margin:0;opacity:0.85">Sending shutdown signal.</p>`);
        try {
            await fetch(API + '/server/stop', { method: 'POST' });
        } catch (e) { /* expected — server going down */ }
        // Poll /api/health until it stops responding, then show the final message.
        const start = Date.now();
        while (Date.now() - start < 10000) {
            await new Promise(r => setTimeout(r, 300));
            try {
                const res = await fetch(API + '/health', { cache: 'no-store' });
                if (!res.ok) break;
            } catch (e) { break; /* down */ }
        }
        _showOverlay(`
            <h2 style="margin:0 0 0.75rem;font-size:1.4rem;color:#e74c3c">Dashboard stopped</h2>
            <p style="margin:0 0 1rem;opacity:0.85;line-height:1.4">The server process has exited.</p>
            <p style="margin:0;opacity:0.85;line-height:1.4">Run <code style="background:#000;padding:2px 6px;border-radius:4px">os8-launcher</code> in a terminal to start it again.</p>
        `);
    };
}

async function restartDashboard() {
    _showOverlay(`<h2 style="margin:0 0 0.75rem;font-size:1.4rem">Restarting dashboard…</h2><p id="restart-status" style="margin:0;opacity:0.85">Sending restart signal.</p>`);
    // Capture the current server's identity BEFORE asking it to restart, so
    // we can later tell the new process apart from the old one. The bind/
    // rebind gap is too small to detect by polling for connection failure.
    let oldServerId = null;
    try {
        const h = await (await fetch(API + '/health', { cache: 'no-store' })).json();
        oldServerId = h.server_id || null;
    } catch (e) { /* ignore */ }
    try {
        await fetch(API + '/server/restart', { method: 'POST' });
    } catch (e) { /* expected during the restart window */ }
    const status = document.getElementById('restart-status');
    if (status) status.textContent = 'Waiting for new server…';
    const start = Date.now();
    while (Date.now() - start < 90000) {
        await new Promise(r => setTimeout(r, 300));
        try {
            const res = await fetch(API + '/health', { cache: 'no-store' });
            if (!res.ok) continue;
            const body = await res.json();
            // Only treat /api/health as "back" once we've confirmed we're
            // talking to a NEW process. The old process keeps answering
            // /api/health (and serving stale /api/status, /api/logs) the
            // whole time stop_all is draining containers.
            // - If body.server_id is missing, we're hitting a pre-server_id
            //   build — almost certainly the old process. Keep waiting.
            // - If it matches the captured oldServerId, same process. Wait.
            if (!body.server_id) continue;
            if (body.server_id === oldServerId) continue;
            {
                // Hard-reload the page so the browser pulls fresh JS/HTML
                // from the new server. Without this the user keeps running
                // whatever app.js was loaded before the restart, which is
                // why edits to the dashboard appear to "not take effect"
                // and why the in-memory log/status state can look stale.
                if (status) status.textContent = 'Reloading page…';
                location.reload();
                return;
            }
        } catch (e) { /* still down */ }
    }
    _showOverlay(`
        <h2 style="margin:0 0 0.75rem;font-size:1.4rem;color:#e74c3c">Dashboard did not come back</h2>
        <p style="margin:0 0 1rem;opacity:0.85;line-height:1.4">No response after 30 seconds. Check the terminal where you ran <code style="background:#000;padding:2px 6px;border-radius:4px">./launcher server</code>.</p>
        <button onclick="location.reload()" style="background:#444;color:#fff;border:none;padding:0.6rem 1.2rem;border-radius:6px;font-size:1rem;cursor:pointer">Try reloading anyway</button>
    `);
}

let modelsData = [];
let toolsData = [];
let statusData = { backend: null, clients: {} };
let lastLogCount = 0;
// Set when the user clicks Serve, cleared once the backend appears in /status
// (or after a safety timeout, or when an Error: line shows up in /logs).
// While set, the Serve button and the model/backend/client selects are
// disabled so the user can't fire a second serve mid-startup.
let serveInFlight = false;
let serveInFlightTimer = null;
const SERVE_INFLIGHT_TIMEOUT_MS = 120000;

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
        // Startup is "done" once the backend is up AND, if a client was
        // requested, the client is up too. We don't track which client was
        // requested, so any client appearing is good enough — same heuristic
        // as before for the normal serve path.
        const _clientsUp = Object.keys(statusData.clients || {}).length > 0;
        if (statusData.backend && (!serveInFlight || _clientsUp || !document.getElementById('client-select')?.value)) {
            _clearServeInFlight();
        }
        updateLaunchControls();
    } catch (e) { /* ignore poll errors */ }
}

function _setServeInFlight() {
    serveInFlight = true;
    if (serveInFlightTimer) clearTimeout(serveInFlightTimer);
    serveInFlightTimer = setTimeout(() => {
        // Safety net: if startup never produced a backend in /status (e.g.
        // silent failure), don't strand the UI forever.
        _clearServeInFlight();
        updateLaunchControls();
    }, SERVE_INFLIGHT_TIMEOUT_MS);
    updateLaunchControls();
}

function _clearServeInFlight() {
    serveInFlight = false;
    if (serveInFlightTimer) {
        clearTimeout(serveInFlightTimer);
        serveInFlightTimer = null;
    }
}

function updateLaunchControls() {
    const backendUp = !!statusData.backend;
    const clientsUp = Object.keys(statusData.clients || {}).length > 0;
    const running = backendUp || clientsUp;
    const busy = running || serveInFlight;
    // Special case: backend is up but no client is running. The user can
    // pick a client and hit Serve to attach it to the existing backend.
    const canAttachClient = backendUp && !clientsUp && !serveInFlight;
    const serveBtn = document.getElementById('btn-serve');
    const stopBtn = document.getElementById('btn-stop');
    const modelSel = document.getElementById('model-select');
    const backendSel = document.getElementById('backend-select');
    const clientSel = document.getElementById('client-select');
    if (serveBtn) {
        serveBtn.disabled = busy && !canAttachClient;
        if (serveInFlight && !running) serveBtn.textContent = 'Starting…';
        else if (canAttachClient) serveBtn.textContent = 'Start client';
        else serveBtn.textContent = 'Serve';
    }
    if (stopBtn) stopBtn.disabled = !busy;
    // Lock model/backend to whatever is already running; client stays
    // editable in the attach-client case.
    if (modelSel) modelSel.disabled = busy;
    if (backendSel) backendSel.disabled = busy;
    if (clientSel) clientSel.disabled = busy && !canAttachClient;

    // When the backend is up, force the model/backend selects to reflect it
    // so the user isn't looking at a stale or empty selection.
    if (backendUp) {
        if (modelSel && statusData.backend.model) modelSel.value = statusData.backend.model;
        if (backendSel && statusData.backend.name) backendSel.value = statusData.backend.name;
    }
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
        // If a startup failed silently (backend never appeared in /status),
        // an Error: line in the logs is our signal to release the in-flight
        // latch so the user can change selections and try again.
        if (serveInFlight && !statusData.backend) {
            for (let i = lines.length - 1; i >= 0 && i >= lines.length - 10; i--) {
                if (lines[i].startsWith('Error:')) {
                    _clearServeInFlight();
                    updateLaunchControls();
                    break;
                }
            }
        }
    } catch (e) { /* ignore poll errors */ }
}

// --- Actions ---

async function startServe() {
    const client = document.getElementById('client-select').value;
    const backendUp = !!statusData.backend;
    const clientsUp = Object.keys(statusData.clients || {}).length > 0;

    // Attach-client path: backend is already running and the user picked a
    // client. Skip /api/serve and just start the client against the existing
    // backend. Source the model/backend from statusData (not the disabled
    // dropdowns) so a stale dropdown value can't silently no-op the click.
    if (backendUp && !clientsUp && client) {
        const tool = toolsData.find(t => t.name === client);
        if (tool && tool.install_type === 'container') {
            _setServeInFlight();
            forceLogScrollNext();
            await fetch(API + `/clients/${client}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: statusData.backend.model,
                    backend: statusData.backend.name,
                }),
            });
            setTimeout(pollStatus, 500);
            setTimeout(pollLogs, 500);
        }
        return;
    }

    // Normal serve path: nothing running yet. Read model/backend from the
    // dropdowns (the user is choosing them now).
    const model = document.getElementById('model-select').value;
    const backend = document.getElementById('backend-select').value;
    if (!model) return;

    // Server-side orchestration: the launcher starts the backend, waits for
    // it to become healthy, then starts the client. This guarantees strict
    // log ordering and prevents the client from racing ahead of the model.
    // Only container clients are launched automatically; CLI clients (aider,
    // etc.) are intentionally launched by the user from a terminal.
    let clientForServer = null;
    if (client) {
        const tool = toolsData.find(t => t.name === client);
        if (tool && tool.install_type === 'container') {
            clientForServer = client;
        }
    }

    _setServeInFlight();
    forceLogScrollNext();
    await fetch(API + '/serve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, backend: backend || null, client: clientForServer }),
    });

    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function stopServe() {
    _clearServeInFlight();
    forceLogScrollNext();
    await fetch(API + '/serve', { method: 'DELETE' });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function downloadModel(name) {
    await fetch(API + `/models/${name}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
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
    // Pass the currently-selected model+backend so the launcher can auto-start
    // a backend if none is running.
    const model = document.getElementById('model-select')?.value || null;
    const backend = document.getElementById('backend-select')?.value || null;
    await fetch(API + `/clients/${name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, backend }),
    });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function stopBackendOnly() {
    forceLogScrollNext();
    await fetch(API + '/backend', { method: 'DELETE' });
    setTimeout(pollStatus, 500);
    setTimeout(pollLogs, 500);
}

async function stopClient(name) {
    forceLogScrollNext();
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

    let html = '<table><tr><th>Model</th><th>Format</th><th>Status</th><th>Size</th><th>Verified</th><th></th></tr>';
    for (const m of models) {
        let statusClass, statusText, sizeCell, actions;
        if (m.state === 'downloading') {
            statusClass = 'status-downloading';
            const pct = m.expected_bytes ? Math.min(100, (m.size_bytes / m.expected_bytes) * 100) : null;
            statusText = pct !== null ? `downloading ${pct.toFixed(0)}%` : 'downloading';
            sizeCell = m.expected_human
                ? `${m.size_human} / ${m.expected_human}`
                : m.size_human;
            actions = `<span class="muted">in progress…</span>`;
        } else if (m.state === 'interrupted') {
            // Partial download with a leftover marker — dashboard died
            // mid-transfer, or the last attempt failed.  HF resumes skip
            // already-complete files, so Resume just works.
            statusClass = 'status-interrupted';
            const errSuffix = m.last_error ? ` — ${escapeAttr(m.last_error).slice(0, 60)}` : '';
            statusText = `interrupted${errSuffix}`;
            sizeCell = m.expected_human
                ? `${m.size_human} / ${m.expected_human} (partial)`
                : `${m.size_human} (partial)`;
            const titleAttr = m.last_error
                ? ` title="Last error: ${escapeAttr(m.last_error)}"`
                : '';
            actions = `
                <button class="btn btn-sm btn-primary" onclick="downloadModel('${m.name}')"${titleAttr}>Resume</button>
                <button class="btn btn-sm btn-danger" onclick="removeModel('${m.name}')" title="Delete partial download">Discard</button>
            `;
        } else if (m.state === 'downloaded') {
            statusClass = 'status-downloaded';
            statusText = 'downloaded';
            sizeCell = m.size_human;
            actions = `<button class="btn btn-sm btn-danger" onclick="removeModel('${m.name}')">Remove</button>`;
        } else {
            statusClass = 'status-not-downloaded';
            statusText = 'not downloaded';
            sizeCell = m.expected_human ? `~${m.expected_human}` : '—';
            actions = `<button class="btn btn-sm btn-primary" onclick="downloadModel('${m.name}')">Download</button>`;
        }
        const verifiedCell = renderVerificationCell(m);
        html += `<tr>
            <td>${m.name}</td>
            <td>${m.format}</td>
            <td class="${statusClass}">${statusText}</td>
            <td>${sizeCell}</td>
            <td>${verifiedCell}</td>
            <td>${actions}</td>
        </tr>`;
    }
    html += '</table>';
    el.innerHTML = html;
}

function renderVerificationCell(m) {
    const v = m.verification || {};
    const parts = [];
    for (const backend of m.backends) {
        const entry = v[backend];
        let badge;
        if (!entry) {
            badge = `<span class="badge badge-unknown" title="Never attempted on this machine">? ${backend}</span>`;
        } else if (entry.last_failure) {
            const tip = `Last failure: ${entry.last_failure}\nAt: ${entry.last_failure_on}` +
                (entry.verified ? `\nPreviously verified: ${entry.verified_on}` : '');
            badge = `<span class="badge badge-failed" title="${escapeAttr(tip)}">✗ ${backend}</span>`;
        } else if (entry.verified) {
            const tip = `Verified ${entry.verified_on}\nRuntime: ${entry.verified_runtime || 'unknown'}`;
            badge = `<span class="badge badge-verified" title="${escapeAttr(tip)}">✓ ${backend}</span>`;
        } else {
            badge = `<span class="badge badge-unknown" title="Never attempted on this machine">? ${backend}</span>`;
        }
        parts.push(badge);
    }
    return parts.join(' ');
}

function escapeAttr(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '&#10;');
}

function renderModelSelect(models) {
    const sel = document.getElementById('model-select');
    const current = sel.value;
    sel.innerHTML = '<option value="">-- select model --</option>';
    const ready = models.filter(m => m.state === 'downloaded');
    for (const m of ready) {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = m.name;
        sel.appendChild(opt);
    }
    if (ready.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.disabled = true;
        opt.textContent = '(no models downloaded)';
        sel.appendChild(opt);
    }
    if (current && ready.some(m => m.name === current)) {
        sel.value = current;
    }
    updateBackendSelect();
    applyProjectDefaults();
}

function updateBackendSelect() {
    const modelName = document.getElementById('model-select').value;
    const sel = document.getElementById('backend-select');
    const current = sel.value;
    sel.innerHTML = '<option value="">-- default --</option>';

    const model = modelsData.find(m => m.name === modelName);
    if (model) {
        const v = model.verification || {};
        for (const b of model.backends) {
            const opt = document.createElement('option');
            opt.value = b;
            const entry = v[b];
            let mark;
            if (entry && entry.last_failure) mark = ' ✗';
            else if (entry && entry.verified) mark = ' ✓';
            else mark = ' ?';
            opt.textContent = b + (b === model.default_backend ? ' (default)' : '') + mark;
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
        if (c.install_type === 'pip' || c.install_type === 'binary') label += ' (terminal)';
        if (c.install_type === 'bridge') label += ' (bridge)';
        opt.textContent = label;
        sel.appendChild(opt);
    }
    if (current) sel.value = current;
    updateClientHint();
    applyProjectDefaults();
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

    if (tool.install_type === 'pip' || tool.install_type === 'binary') {
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
        const baseUrl = `http://localhost:${b.port}`;
        const apiBase = `${baseUrl}/v1`;
        const endpoints = [
            `${apiBase}/models`,
            `${apiBase}/chat/completions`,
            `${apiBase}/completions`,
            `${apiBase}/embeddings`,
        ];
        const endpointsHtml = endpoints
            .map(e => `<div style="font-family:monospace;font-size:0.85em;opacity:0.8;margin-left:1rem">${e}</div>`)
            .join('');
        html += `<h3>Backend</h3>
        <div class="session-info">
            <div><span class="label">Backend:</span> ${b.name}</div>
            <div><span class="label">Model:</span> ${b.model}</div>
            <div><span class="label">URL:</span> <a href="${baseUrl}" target="_blank" style="color:#5e7ce2">${baseUrl}</a></div>
            <div><span class="label">API base:</span> <span style="font-family:monospace">${apiBase}</span></div>
            <div style="margin-top:0.25rem"><span class="label">Endpoints:</span>${endpointsHtml}</div>
            <div><span class="label">Health:</span> <span class="dot ${dotClass}"></span><span class="${healthClass}">${b.health}</span></div>
            <div><span class="label">Uptime:</span> ${b.uptime}</div>
            <div style="margin-top:0.5rem"><button class="btn btn-sm btn-danger" onclick="stopBackendOnly()">Stop backend</button></div>
        </div>`;
    }

    for (const name of Object.keys(status.clients)) {
        const c = status.clients[name];
        let urlLine;
        if (c.port) {
            urlLine = c.ready
                ? `<a href="http://localhost:${c.port}" target="_blank" style="color:#5e7ce2">http://localhost:${c.port}</a>`
                : `<span style="opacity:0.6">starting…</span>`;
        } else {
            urlLine = '-';
        }
        html += `<h3>Client</h3>
        <div class="session-info">
            <div><span class="label">Client:</span> ${name}</div>
            <div><span class="label">URL:</span> ${urlLine}</div>
            <div><span class="label">Uptime:</span> ${c.uptime}</div>
            <div style="margin-top:0.5rem"><button class="btn btn-sm btn-danger" onclick="stopClient('${name}')">Stop client</button></div>
        </div>`;
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
        let actions = '';
        if (t.status === 'not installed') actions = `<button class="btn btn-sm btn-primary" onclick="setupTool('${t.name}')">Setup</button>`;
        else if (t.status === 'installing') actions = `<span class="muted">installing…</span>`;
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
        let actions = '';
        if (t.status === 'not installed') actions = `<button class="btn btn-sm btn-primary" onclick="setupTool('${t.name}')">Setup</button>`;
        else if (t.status === 'installing') actions = `<span class="muted">installing…</span>`;
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

// Set by action handlers (startServe, stopServe, stopBackendOnly, stopClient)
// to force the log pane to scroll to the bottom on the next renderLogs, so
// the lines produced by the user's click aren't hidden if they had scrolled
// up to read older output.
let _forceLogScroll = false;
function forceLogScrollNext() { _forceLogScroll = true; }

// Escape HTML so log content can never inject markup, then turn bare http(s)
// URLs into clickable links that open in a new tab. Trailing punctuation
// (., ,, ), ], etc.) is excluded from the link so "...port 3000)." works.
function _escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}
function _linkifyEscaped(text) {
    return _escapeHtml(text).replace(
        /(https?:\/\/[^\s<>"']+[^\s<>"'.,;:!?)\]}])/g,
        url => `<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`,
    );
}

function renderLogs(lines) {
    const el = document.getElementById('log-output');
    const nextText = lines.join('\n');
    // 1. Don't touch the DOM if nothing changed — preserves any active selection.
    if (el.dataset.rawText === nextText) return;
    // 2. Don't clobber an active selection inside the log pane — wait for the
    //    user to copy / click away before re-rendering.
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && el.contains(sel.anchorNode)) return;
    const shouldScroll = _forceLogScroll || el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
    _forceLogScroll = false;
    el.innerHTML = _linkifyEscaped(nextText);
    el.dataset.rawText = nextText;
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

// --- Resizable dividers ---

(function initDividers() {
    const main = document.querySelector('main');
    const divV = document.getElementById('divider-v');
    const divH = document.getElementById('divider-h');
    const STORAGE_KEY = 'os8-layout';

    function loadLayout() {
        try {
            return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
        } catch (e) { return {}; }
    }

    function saveLayout(colPct, rowPct) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ col: colPct, row: rowPct }));
    }

    function applyLayout(colPct, rowPct) {
        main.style.gridTemplateColumns = `${colPct}fr ${100 - colPct}fr`;
        main.style.gridTemplateRows = `${rowPct}fr ${100 - rowPct}fr`;
        positionHandles(colPct, rowPct);
    }

    function positionHandles(colPct, rowPct) {
        divV.style.left = `${colPct}%`;
        divH.style.top = `${rowPct}%`;
    }

    function clamp(val, min, max) { return Math.min(max, Math.max(min, val)); }

    const saved = loadLayout();
    let colPct = saved.col || 50;
    let rowPct = saved.row || 50;
    applyLayout(colPct, rowPct);

    // Vertical divider drag
    divV.addEventListener('mousedown', (e) => {
        e.preventDefault();
        divV.classList.add('active');
        document.body.classList.add('resizing');
        const rect = main.getBoundingClientRect();
        function onMove(e) {
            colPct = clamp(((e.clientX - rect.left) / rect.width) * 100, 15, 85);
            applyLayout(colPct, rowPct);
        }
        function onUp() {
            divV.classList.remove('active');
            document.body.classList.remove('resizing');
            saveLayout(colPct, rowPct);
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // Horizontal divider drag
    divH.addEventListener('mousedown', (e) => {
        e.preventDefault();
        divH.classList.add('active');
        document.body.classList.add('resizing');
        const rect = main.getBoundingClientRect();
        function onMove(e) {
            rowPct = clamp(((e.clientY - rect.top) / rect.height) * 100, 15, 85);
            applyLayout(colPct, rowPct);
        }
        function onUp() {
            divH.classList.remove('active');
            document.body.classList.remove('resizing');
            saveLayout(colPct, rowPct);
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    // Double-click to reset
    divV.addEventListener('dblclick', () => { colPct = 50; applyLayout(colPct, rowPct); saveLayout(colPct, rowPct); });
    divH.addEventListener('dblclick', () => { rowPct = 50; applyLayout(colPct, rowPct); saveLayout(colPct, rowPct); });
})();
