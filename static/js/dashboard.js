// Equity Trading Terminal - Dashboard JS

// ============ AUTH FUNCTIONS ============
async function generateAuthUrl() {
    const btn = document.getElementById('auth-url-btn');
    const msg = document.getElementById('auth-url-msg');
    if (!btn) return;

    btn.disabled = true;
    btn.textContent = 'Generating...';
    try {
        const resp = await fetch('/auth');
        const data = await resp.json();
        if (data.url) {
            window.open(data.url, '_blank');
            btn.textContent = 'Auth Page Opened';
            btn.className = 'btn success';
            msg.innerHTML = '<span style="color:var(--accent-green)">Fyers login opened in new tab. Complete login, then paste the auth_code below.</span>';
        } else {
            msg.innerHTML = '<span style="color:var(--accent-red)">' + (data.error || 'Failed to generate URL') + '</span>';
            btn.textContent = 'Generate Auth URL';
            btn.disabled = false;
        }
    } catch(err) {
        msg.innerHTML = '<span style="color:var(--accent-red)">Error: ' + err.message + '</span>';
        btn.textContent = 'Generate Auth URL';
        btn.disabled = false;
    }
}

async function saveCredentials() {
    const appId = document.getElementById('input-app-id')?.value?.trim();
    const secret = document.getElementById('input-secret')?.value?.trim();
    const msg = document.getElementById('creds-msg');
    if (!appId || !secret) {
        msg.innerHTML = '<span style="color:var(--accent-red)">Both fields required</span>';
        return;
    }
    try {
        const resp = await fetch('/api/credentials', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ app_id: appId, secret: secret }),
        });
        const data = await resp.json();
        if (data.success) {
            msg.innerHTML = '<span style="color:var(--accent-green)">' + data.message + '</span>';
            const display = document.getElementById('current-app-id');
            if (display) display.textContent = appId;
        } else {
            msg.innerHTML = '<span style="color:var(--accent-red)">' + data.message + '</span>';
        }
    } catch(err) {
        msg.innerHTML = '<span style="color:var(--accent-red)">Error: ' + err.message + '</span>';
    }
}

async function logoutFyers() {
    if (!confirm('Logout from Fyers?')) return;
    await fetch('/auth/logout', { method: 'POST' });
    window.location.reload();
}

// Auth form handler
document.addEventListener('DOMContentLoaded', () => {
    const authForm = document.getElementById('authForm');
    if (authForm) {
        authForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            let code = document.getElementById('authCodeInput').value.trim();
            const msg = document.getElementById('auth-msg');
            if (!code) {
                msg.innerHTML = '<span style="color:var(--accent-red)">Please enter auth code</span>';
                return;
            }

            // Extract auth_code from full URL if user pasted the whole redirect URL
            try {
                if (code.includes('auth_code=')) {
                    const url = new URL(code);
                    code = url.searchParams.get('auth_code') || code;
                }
            } catch(urlErr) {
                const match = code.match(/auth_code=([^&\s#]+)/);
                if (match) code = match[1];
            }

            msg.innerHTML = '<span style="color:var(--accent-blue)">Authenticating...</span>';
            try {
                const resp = await fetch('/auth/callback', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: 'auth_code=' + encodeURIComponent(code),
                });
                const data = await resp.json();
                if (data.success) {
                    msg.innerHTML = '<span style="color:var(--accent-green)">Connected! Reloading...</span>';
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    msg.innerHTML = '<span style="color:var(--accent-red)">' + (data.message || 'Authentication failed') + '</span>';
                }
            } catch(err) {
                msg.innerHTML = '<span style="color:var(--accent-red)">Error: ' + err.message + '</span>';
            }
        });
    }
});

// ============ DASHBOARD ============
const socket = io();

// State
let state = {
    running: false,
    autoExecute: false,
    autoAdapt: true,
    regime: 'CONSOLIDATION',
    strategy: 'FUSION_FULL',
    dailyPnl: 0,
    activePositions: [],
    signals: [],
};

// Socket Events
socket.on('connect', () => {
    console.log('Connected to server');
    document.getElementById('conn-status').textContent = 'Connected';
    document.getElementById('conn-status').className = 'status-badge running';
    fetchStatus();
});

socket.on('disconnect', () => {
    document.getElementById('conn-status').textContent = 'Disconnected';
    document.getElementById('conn-status').className = 'status-badge stopped';
});

socket.on('update', (data) => {
    updateDashboard(data);
});

socket.on('training_complete', (data) => {
    addSignalLog(`Model training complete for ${data.symbol}`, 'info');
});

// API Calls
async function fetchStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        state.running = data.running;
        state.autoExecute = data.auto_execute;
        state.autoAdapt = data.auto_adapt;
        state.regime = data.regime;
        state.strategy = data.strategy;
        state.dailyPnl = data.daily_pnl;
        updateStatusUI();
    } catch (e) {
        console.error('Status fetch failed:', e);
    }
}

async function startPoller() {
    await fetch('/api/start', { method: 'POST' });
    state.running = true;
    updateStatusUI();
}

async function stopPoller() {
    await fetch('/api/stop', { method: 'POST' });
    state.running = false;
    updateStatusUI();
}

async function toggleAutoExecute() {
    const resp = await fetch('/api/toggle_auto_execute', { method: 'POST' });
    const data = await resp.json();
    state.autoExecute = data.auto_execute;
    updateStatusUI();
}

async function toggleAutoAdapt() {
    const resp = await fetch('/api/toggle_auto_adapt', { method: 'POST' });
    const data = await resp.json();
    state.autoAdapt = data.auto_adapt;
    updateStatusUI();
}

async function setUniverse() {
    const sel = document.getElementById('universe-select');
    await fetch('/api/set_universe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ universe: sel.value }),
    });
    fetchStatus();
}

async function setStrategy() {
    const sel = document.getElementById('strategy-select');
    await fetch('/api/set_strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy: sel.value }),
    });
    state.strategy = sel.value;
    updateStatusUI();
}

async function confirmTrade(posId) {
    await fetch(`/api/confirm_trade/${posId}`, { method: 'POST' });
    fetchPositions();
}

async function cancelTrade(posId) {
    await fetch(`/api/cancel_trade/${posId}`, { method: 'POST' });
    fetchPositions();
}

async function fetchPositions() {
    try {
        const resp = await fetch('/api/positions');
        const data = await resp.json();
        state.activePositions = data.active;
        state.dailyPnl = data.daily_pnl;
        renderPositions();
        renderPnl();
    } catch (e) {
        console.error('Positions fetch failed:', e);
    }
}

// UI Updates
function updateStatusUI() {
    // Running status
    const runBtn = document.getElementById('run-btn');
    const runStatus = document.getElementById('run-status');
    if (state.running) {
        runBtn.textContent = 'Stop';
        runBtn.className = 'btn danger';
        runBtn.onclick = stopPoller;
        runStatus.textContent = 'Running';
        runStatus.className = 'status-badge running';
    } else {
        runBtn.textContent = 'Start';
        runBtn.className = 'btn success';
        runBtn.onclick = startPoller;
        runStatus.textContent = 'Stopped';
        runStatus.className = 'status-badge stopped';
    }

    // Auto-execute
    const aeBtn = document.getElementById('ae-btn');
    aeBtn.textContent = state.autoExecute ? 'Auto-Execute: ON' : 'Auto-Execute: OFF';
    aeBtn.className = state.autoExecute ? 'btn success sm' : 'btn sm';

    // Auto-adapt
    const aaBtn = document.getElementById('aa-btn');
    aaBtn.textContent = state.autoAdapt ? 'Auto-Adapt: ON' : 'Auto-Adapt: OFF';
    aaBtn.className = state.autoAdapt ? 'btn primary sm' : 'btn sm';

    // Regime
    const regimeEl = document.getElementById('regime-display');
    regimeEl.textContent = state.regime;
    regimeEl.className = `regime-badge regime-${state.regime}`;

    // Strategy
    document.getElementById('strategy-display').textContent = state.strategy;

    renderPnl();
}

function updateDashboard(data) {
    state.activePositions = data.active_positions || [];
    state.dailyPnl = data.daily_pnl || 0;
    state.regime = data.regime || state.regime;
    state.strategy = data.strategy || state.strategy;

    renderPositions();
    renderPnl();
    updateStatusUI();

    // Add new signals to log
    if (data.signals && data.signals.length > 0) {
        data.signals.forEach(s => {
            const ticker = s.symbol.replace('NSE:', '').replace('-EQ', '');
            addSignalLog(
                `${s.direction} ${ticker} @ ${s.entry_price.toFixed(2)} ` +
                `(${s.confidence.toFixed(1)}% ${s.strength}) ` +
                `Pattern: ${s.pattern_name || 'ML'} R:R=${s.risk_reward}`,
                s.direction.toLowerCase()
            );
        });
    }

    // Pending signals
    if (data.pending_positions && data.pending_positions.length > 0) {
        renderPendingSignals(data.pending_positions);
    }
}

function renderPositions() {
    const tbody = document.getElementById('positions-body');
    if (!tbody) return;

    if (!state.activePositions || state.activePositions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text-muted)">No active positions</td></tr>';
        return;
    }

    tbody.innerHTML = state.activePositions.map(p => {
        const ticker = (p.symbol || '').replace('NSE:', '').replace('-EQ', '');
        const pnlClass = (p.unrealized_pnl || 0) >= 0 ? 'green' : 'red';
        return `<tr>
            <td class="dir-${(p.direction || '').toLowerCase()}">${p.direction}</td>
            <td><strong>${ticker}</strong></td>
            <td>${(p.entry_price || 0).toFixed(2)}</td>
            <td>${(p.current_price || 0).toFixed(2)}</td>
            <td>${(p.stop_loss || 0).toFixed(2)}</td>
            <td>${(p.target || 0).toFixed(2)}</td>
            <td class="${pnlClass}">${(p.unrealized_pnl || 0).toFixed(2)}</td>
            <td>${p.pattern_name || '-'}</td>
        </tr>`;
    }).join('');
}

function renderPendingSignals(pending) {
    const container = document.getElementById('pending-signals');
    if (!container) return;

    container.innerHTML = pending.map(p => {
        const ticker = (p.symbol || '').replace('NSE:', '').replace('-EQ', '');
        return `<div class="signal-entry ${(p.direction || '').toLowerCase()}">
            <span class="signal-symbol">${p.direction} ${ticker}</span>
            @ ${(p.entry_price || 0).toFixed(2)} | Conf: ${(p.signal_confidence || 0).toFixed(1)}%
            <br>
            <button class="btn success sm" onclick="confirmTrade('${p.id}')">Confirm</button>
            <button class="btn danger sm" onclick="cancelTrade('${p.id}')">Cancel</button>
        </div>`;
    }).join('');
}

function renderPnl() {
    const el = document.getElementById('daily-pnl');
    if (!el) return;
    const val = state.dailyPnl || 0;
    el.textContent = `${val >= 0 ? '+' : ''}${val.toFixed(2)}`;
    el.className = `metric-value ${val >= 0 ? 'green' : 'red'}`;
}

function addSignalLog(msg, type) {
    const log = document.getElementById('signal-log');
    if (!log) return;

    const entry = document.createElement('div');
    entry.className = `signal-entry ${type}`;
    const now = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="signal-time">${now}</span> ${msg}`;
    log.prepend(entry);

    // Keep max 50 entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// Initial load
document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    fetchPositions();
    setInterval(fetchPositions, 10000);
});
