// Dashboard-level actions shared by every page that wants the
// header's Restart / Stop server controls. The host page must define
// `API` (e.g. const API = '/api') before loading this script.

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
    _showOverlay(`
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
