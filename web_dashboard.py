#!/usr/bin/env python3
"""
PRNG Cluster Dashboard - HiveOS-Inspired Scientific Interface
==============================================================
Multi-route version with functional tabs

Routes:
  /           - Overview (main dashboard)
  /workers    - Detailed GPU stats per worker
  /stats      - Historical statistics
  /plots      - Seaborn/Matplotlib visualizations
  /settings   - Dashboard configuration
"""

import json
import os
import time
import base64
import io
from datetime import datetime, timedelta
from pathlib import Path

# Suppress Flask request logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from flask import Flask, render_template_string, jsonify, request

# GPU monitoring
try:
    from gpu_monitor import get_cluster_gpu_stats
    GPU_MONITOR_AVAILABLE = True
except ImportError:
    GPU_MONITOR_AVAILABLE = False
    def get_cluster_gpu_stats():
        return {}

app = Flask(__name__)

PROGRESS_FILE = "/tmp/cluster_progress.json"
SETTINGS_FILE = "/tmp/dashboard_settings.json"

DEFAULT_SETTINGS = {
    "refresh_interval": 2,
    "theme": "dark",
    "show_offline_workers": True,
    "plot_height": 380,
    "max_history_entries": 100
}

def load_settings():
    """Load settings from file, return defaults if not found"""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            saved = json.load(f)
            return {**DEFAULT_SETTINGS, **saved}
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to file"""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)
    return True

HISTORY_FILE_PATH = "/tmp/cluster_run_history.json"

def load_run_history():
    """Load run history from file"""
    try:
        with open(HISTORY_FILE_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_run_history(history):
    """Save run history to file"""
    settings = load_settings()
    max_entries = settings.get('max_history_entries', 100)
    # Trim to max entries
    history = history[:max_entries]
    with open(HISTORY_FILE_PATH, 'w') as f:
        json.dump(history, f, indent=2)

def add_run_to_history(run_data):
    """Add a completed run to history"""
    history = load_run_history()
    history.insert(0, {
        'timestamp': datetime.now().isoformat(),
        'step_name': run_data.get('step_name', 'Unknown'),
        'total_seeds': run_data.get('total_seeds', 0),
        'duration_seconds': run_data.get('elapsed_seconds', 0),
        'final_sps': run_data.get('final_sps', 0)
    })
    save_run_history(history)
    return history




PLOTS_DIR = "/tmp/prng_plots"
HISTORY_FILE = "/tmp/cluster_history.json"

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# Shared CSS and Base Template
# ============================================================================

BASE_CSS = """
:root {
    --bg-primary: #1a1d21;
    --bg-secondary: #22262a;
    --bg-card: #2a2e33;
    --bg-hover: #32373d;
    --text-primary: #e8e8e8;
    --text-secondary: #8a9099;
    --text-muted: #5a6068;
    --accent-green: #02e079;
    --accent-blue: #3b82f6;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --border-color: #3a3f45;
}

/* Light theme overrides */
body.light-theme {
    --bg-primary: #f5f5f7;
    --bg-secondary: #ffffff;
    --bg-card: #ffffff;
    --bg-hover: #e8e8ed;
    --text-primary: #1d1d1f;
    --text-secondary: #6e6e73;
    --text-muted: #86868b;
    --accent-green: #00a854;
    --accent-blue: #0071e3;
    --accent-orange: #f56300;
    --accent-red: #ff3b30;
    --accent-purple: #af52de;
    --border-color: #d2d2d7;
}

body.light-theme .card {
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

body.light-theme .summary-bar,
body.light-theme .summary-item {
    background: #ffffff;
    border-color: #d2d2d7;
}

body.light-theme .nav-tab {
    color: #6e6e73;
}

body.light-theme .nav-tab:hover,
body.light-theme .nav-tab.active {
    color: #1d1d1f;
}

body.light-theme .form-select,
body.light-theme .form-input {
    background: #f5f5f7;
    color: #1d1d1f;
}


:root {
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.5;
}

/* Top Summary Bar */
.summary-bar {
    display: flex;
    gap: 1px;
    background: var(--border-color);
    border-bottom: 1px solid var(--border-color);
}

.summary-item {
    flex: 1;
    background: var(--bg-secondary);
    padding: 10px 14px;
    text-align: center;
    min-width: 100px;
}

.summary-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
}

.summary-value.green { color: var(--accent-green); }
.summary-value.blue { color: var(--accent-blue); }
.summary-value.orange { color: var(--accent-orange); }
.summary-value.red { color: var(--accent-red); }

.summary-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Navigation Tabs */
.nav-tabs {
    display: flex;
    gap: 0;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 0 16px;
}

.nav-tab {
    padding: 12px 18px;
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 13px;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

.nav-tab:hover { color: var(--text-primary); background: var(--bg-hover); }
.nav-tab.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent-blue);
}

/* Main Content */
.main-content {
    padding: 16px;
    max-width: 1600px;
    margin: 0 auto;
}

/* Cards */
.card {
    background: var(--bg-card);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid var(--border-color);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--accent-blue);
}

/* Progress Bar */
.progress-container {
    background: var(--bg-primary);
    border-radius: 3px;
    height: 14px;
    overflow: hidden;
    margin-bottom: 10px;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
    border-radius: 3px;
    transition: width 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 600;
    min-width: 40px;
}

.progress-bar.complete { background: var(--accent-green); }

/* Stats Grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
}

.stat-item { text-align: center; }

.stat-value {
    font-size: 15px;
    font-weight: 600;
    color: var(--accent-green);
}

.stat-label {
    font-size: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
}

/* Worker Table */
.worker-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.worker-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-color);
    background: var(--bg-secondary);
}

.worker-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-color);
}

.worker-table tr:hover { background: var(--bg-hover); }

.worker-name { font-weight: 600; }
.worker-type { font-size: 11px; color: var(--text-secondary); }

/* GPU Segments */
.gpu-segments { display: flex; gap: 2px; }

.gpu-segment {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    background: var(--bg-primary);
}

.gpu-segment.active { background: var(--accent-green); }
.gpu-segment.idle { background: var(--text-muted); }
.gpu-segment.offline { background: var(--accent-red); }

/* Throughput */
.throughput { font-family: 'Monaco', 'Consolas', monospace; }
.throughput-value { color: var(--accent-green); font-weight: 600; }
.throughput-unit { color: var(--text-secondary); font-size: 11px; }

/* Mini Chart */
.mini-chart {
    width: 140px;
    height: 24px;
    background: var(--bg-primary);
    border-radius: 2px;
    display: flex;
    align-items: flex-end;
    gap: 1px;
    padding: 2px;
}

.mini-bar {
    flex: 1;
    background: var(--accent-green);
    min-height: 2px;
    border-radius: 1px;
    opacity: 0.7;
}

/* Charts */
.charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

.chart-wrapper {
    height: 450px;
    overflow: hidden;
}

.chart-wrapper .js-plotly-plot, .chart-wrapper .plotly {
    width: 100% !important;
    height: 100% !important;
}

.chart-placeholder {
    height: 180px;
    background: var(--bg-primary);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    font-size: 12px;
}

.chart-img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
}

/* Status indicators */
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.status-dot.online { background: var(--accent-green); }
.status-dot.offline { background: var(--accent-red); }
.status-dot.idle { background: var(--text-muted); }

/* Settings Form */
.form-group {
    margin-bottom: 16px;
}

.form-label {
    display: block;
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

.form-input, .form-select {
    width: 100%;
    max-width: 300px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
}

.form-input:focus, .form-select:focus {
    outline: none;
    border-color: var(--accent-blue);
}

.btn {
    padding: 8px 16px;
    background: var(--accent-blue);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
}

.btn:hover { opacity: 0.9; }

/* Waiting State */
.waiting-state {
    text-align: center;
    padding: 40px 16px;
}

.waiting-icon { font-size: 36px; margin-bottom: 12px; }
.waiting-title { font-size: 16px; font-weight: 600; margin-bottom: 6px; color: var(--text-secondary); }
.waiting-text { color: var(--text-muted); font-size: 12px; }

.command-box {
    background: var(--bg-primary);
    padding: 10px 16px;
    border-radius: 4px;
    font-family: monospace;
    margin-top: 16px;
    display: inline-block;
    font-size: 11px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 16px;
    color: var(--text-muted);
    font-size: 11px;
    border-top: 1px solid var(--border-color);
    margin-top: 30px;
}

/* GPU Detail Card */
.gpu-detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}

.gpu-card {
    background: var(--bg-secondary);
    border-radius: 4px;
    padding: 12px;
    border: 1px solid var(--border-color);
}

.gpu-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.gpu-id {
    font-weight: 600;
    color: var(--accent-blue);
}

.gpu-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    font-size: 11px;
}

.gpu-stat-value {
    font-weight: 600;
    color: var(--accent-green);
}

.gpu-stat-label {
    color: var(--text-muted);
    font-size: 9px;
    text-transform: uppercase;
}

/* History Table */
.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
}

.history-table th, .history-table td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.history-table th {
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-size: 10px;
    text-transform: uppercase;
}
"""

# ============================================================================
# Page Templates
# ============================================================================

def base_template(content, active_tab="overview", auto_refresh=True):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRNG Cluster Dashboard</title>
    {'<meta http-equiv="refresh" content="2">' if auto_refresh else ''}
    <style>{BASE_CSS}</style>
</head>
<body>
    <script>
    (function() {{
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/settings', false);
        try {{
            xhr.send();
            if (xhr.status === 200) {{
                var settings = JSON.parse(xhr.responseText);
                if (settings.theme === 'light') {{
                    document.body.classList.add('light-theme');
                }}
            }}
        }} catch(e) {{}}
    }})();
    </script>
    {{{{ summary_bar | safe }}}}

    <div class="nav-tabs">
        <a href="/" class="nav-tab {'active' if active_tab == 'overview' else ''}">Overview</a>
        <a href="/workers" class="nav-tab {'active' if active_tab == 'workers' else ''}">Workers</a>
        <a href="/stats" class="nav-tab {'active' if active_tab == 'stats' else ''}">Stats</a>
        <a href="/plots" class="nav-tab {'active' if active_tab == 'plots' else ''}">Plots</a>
        <a href="/settings" class="nav-tab {'active' if active_tab == 'settings' else ''}">Settings</a>
    </div>

    <div class="main-content">
        {content}
    </div>

    <div class="footer">
        PRNG Cluster Dashboard • 26 GPUs • ~285 TFLOPS • Auto-refresh: 2s
    </div>
</body>
</html>
"""

OVERVIEW_CONTENT = """
{% if state %}
<div class="card">
    <div class="card-header">
        <div class="card-title">{{ state.step_name }}</div>
        <div style="font-size: 11px;">
            <span class="status-dot {% if state.finished %}online{% else %}online{% endif %}"
                  style="animation: {% if not state.finished %}pulse 2s infinite{% endif %};"></span>
            {% if state.finished %}Complete{% else %}Running{% endif %}
        </div>
    </div>

    <div class="progress-container">
        <div class="progress-bar {% if state.finished %}complete{% endif %}"
             style="width: {{ progress_pct }}%">
            {{ "%.1f"|format(progress_pct) }}%
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-value">{{ "{:,}".format(state.seeds_completed) }}</div>
            <div class="stat-label">Seeds</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ state.jobs_completed }}/{{ state.total_jobs }}</div>
            <div class="stat-label">Jobs</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ "{:,.0f}".format(total_sps) }}</div>
            <div class="stat-label">Seeds/sec</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ active_gpus }}/{{ total_gpus }}</div>
            <div class="stat-label">GPUs</div>
        </div>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Cluster Workers</div>
        <span style="font-size: 11px; color: var(--accent-green);">{{ nodes|length }} online</span>
    </div>

    <table class="worker-table">
        <thead>
            <tr>
                <th>Worker</th>
                <th>Avg Clock</th>
                <th>Status</th>
                <th>Throughput</th>
                <th>Jobs</th>
                <th>Activity</th>
            </tr>
        </thead>
        <tbody>
            {% for hostname, node in nodes.items() %}
            <tr>
                <td>
                    <div class="worker-name">{{ hostname }}</div>
                    <div class="worker-type">{{ node.gpu_type }}</div>
                </td>
                <td>
                    {% set gpu_data = gpu_stats.get(hostname, []) %}
                    {% set avg_clock = 0 %}
                    {% if gpu_data %}
                        {% set clock_sum = namespace(val=0) %}
                        {% for g in gpu_data %}
                            {% set clock_sum.val = clock_sum.val + g.get('clock', 0) %}
                        {% endfor %}
                        {% set avg_clock = (clock_sum.val / gpu_data|length)|int %}
                    {% endif %}
                    <span style="color: {% if avg_clock > 1000 %}var(--accent-green){% elif avg_clock > 100 %}var(--accent-orange){% else %}var(--text-muted){% endif %}; font-weight: 600;">
                        {{ avg_clock }} MHz
                    </span>
                </td>
                <td>
                    {% if node.current_seeds_per_sec > 0 %}
                    <span style="color: var(--accent-green);">● Active</span>
                    {% else %}
                    <span style="color: var(--text-muted);">○ Idle</span>
                    {% endif %}
                </td>
                <td>
                    <span class="throughput-value">{{ "{:,.0f}".format(node.current_seeds_per_sec) }}</span>
                    <span class="throughput-unit">s/s</span>
                </td>
                <td>{{ node.jobs_completed }}</td>
                <td>
                    <div class="mini-chart">
                        {% set gpu_data = gpu_stats.get(hostname, []) %}
                        {% for i in range(node.total_gpus) %}
                            {% if gpu_data and i < gpu_data|length %}
                                {% set clock_pct = (gpu_data[i].get('clock', 0) / 2000 * 100)|int %}
                                <div class="mini-bar" style="height: {{ clock_pct if clock_pct > 5 else 5 }}%; background: {% if clock_pct > 50 %}var(--accent-green){% elif clock_pct > 10 %}var(--accent-orange){% else %}var(--text-muted){% endif %};"></div>
                            {% else %}
                                <div class="mini-bar" style="height: 5%;"></div>
                            {% endif %}
                        {% endfor %}
                    </div>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
<!-- Trial Stats Card -->
{% if state.trial_stats and state.trial_stats.trial_num %}
<div class="card">
    <div class="card-header">
        <div class="card-title">Live Trial Data</div>
        <span style="font-size: 11px; color: var(--accent-green);">Trial #{{ state.trial_stats.trial_num }}</span>
    </div>
    <div style="font-size: 12px; color: var(--text-secondary); margin-bottom: 12px;">
        {{ state.trial_stats.config_desc }}
    </div>
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-value" style="color: var(--accent-blue);">{{ "{:,}".format(state.trial_stats.forward_survivors) }}</div>
            <div class="stat-label">Forward Survivors</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color: var(--accent-purple);">{{ "{:,}".format(state.trial_stats.reverse_survivors) }}</div>
            <div class="stat-label">Reverse Survivors</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color: var(--accent-green);">{{ "{:,}".format(state.trial_stats.bidirectional) }}</div>
            <div class="stat-label">Bidirectional</div>
        </div>
        <div class="stat-item">
            <div class="stat-value" style="color: var(--accent-orange);">{{ "{:,}".format(state.trial_stats.best_bidirectional) }}</div>
            <div class="stat-label">Best So Far</div>
        </div>
    </div>
    <!-- Accumulated Totals -->
    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">
        <div style="font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">📊 Accumulated Across All Trials</div>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value" style="color: var(--accent-blue); font-size: 16px;">{{ "{:,}".format(state.trial_stats.accumulated_forward|default(0)) }}</div>
                <div class="stat-label">Total Forward</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: var(--accent-purple); font-size: 16px;">{{ "{:,}".format(state.trial_stats.accumulated_reverse|default(0)) }}</div>
                <div class="stat-label">Total Reverse</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" style="color: var(--accent-green); font-size: 16px;">{{ "{:,}".format(state.trial_stats.accumulated_bidirectional|default(0)) }}</div>
                <div class="stat-label">Total Bidirectional</div>
            </div>
        </div>
    </div>
</div>
{% endif %}

{% else %}
<div class="card">
    <div class="waiting-state">
        <div class="waiting-icon">🔍</div>
        <div class="waiting-title">Waiting for cluster activity...</div>
        <div class="waiting-text">Start a pipeline job to see real-time progress</div>
        <div class="command-box">python3 window_optimizer.py --strategy bayesian --trials 20 --lottery-file synthetic_lottery.json</div>
    </div>
</div>
{% endif %}
"""

WORKERS_CONTENT = """
<div class="card">
    <div class="card-header">
        <div class="card-title">Worker Details</div>
    </div>

    {% if nodes %}
    {% for hostname, node in nodes.items() %}
    <div style="margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border-color);">
            <div>
                <span style="font-size: 14px; font-weight: 600;">{{ hostname }}</span>
                <span style="color: var(--text-secondary); margin-left: 8px;">{{ node.total_gpus }}× {{ node.gpu_type }}</span>
            </div>
            <div>
                {% if node.current_seeds_per_sec > 0 %}
                <span class="status-dot online"></span>Active
                {% else %}
                <span class="status-dot idle"></span>Idle
                {% endif %}
            </div>
        </div>

        <div class="gpu-detail-grid">
            {% for i in range(node.total_gpus) %}
            <div class="gpu-card">
                <div class="gpu-card-header">
                    <span class="gpu-id">GPU {{ i }}</span>
                    <span style="font-size: 10px; color: var(--text-secondary);">{{ node.gpu_type }}</span>
                </div>
                <div class="gpu-stats">
                    <div>
                        <div class="gpu-stat-value">{% if node.current_seeds_per_sec > 0 %}{{ "{:,.0f}".format(node.current_seeds_per_sec / node.total_gpus) }}{% else %}0{% endif %}</div>
                        <div class="gpu-stat-label">Seeds/s</div>
                    </div>
                    <div>
                        {% set gpu_list = gpu_stats.get(hostname, []) %}
                        {% set gpu_info = gpu_list[i] if i < gpu_list|length else {} %}
                        {% set gpu_clock = gpu_info.get('clock', 0) if gpu_info else 0 %}
                        <div class="gpu-stat-value" style="color: {% if gpu_clock > 1000 %}var(--accent-green){% elif gpu_clock > 0 %}var(--accent-orange){% else %}var(--text-secondary){% endif %};">{{ gpu_clock }} MHz</div>
                        <div class="gpu-stat-label">Clock</div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endfor %}
    {% else %}
    <div class="waiting-state">
        <div class="waiting-icon">🖥️</div>
        <div class="waiting-title">No worker data available</div>
        <div class="waiting-text">Worker information will appear when a job is running</div>
    </div>
    {% endif %}
</div>
"""

STATS_CONTENT = """
<div class="card">
    <div class="card-header">
        <div class="card-title">Cluster Statistics</div>
    </div>

    <div class="stats-grid" style="margin-bottom: 20px;">
        <div class="stat-item">
            <div class="stat-value">{{ total_gpus }}</div>
            <div class="stat-label">Total GPUs</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">~285</div>
            <div class="stat-label">TFLOPS</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Nodes</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ "{:,.0f}".format(total_sps) }}</div>
            <div class="stat-label">Current Seeds/s</div>
        </div>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Hardware Summary</div>
    </div>

    <table class="history-table">
        <thead>
            <tr>
                <th>Node</th>
                <th>GPUs</th>
                <th>Type</th>
                <th>VRAM</th>
                <th>Compute</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>zeus (localhost)</td>
                <td>2</td>
                <td>RTX 3080 Ti</td>
                <td>12GB</td>
                <td>~60 TFLOPS</td>
            </tr>
            <tr>
                <td>rig-6600 (192.168.3.120)</td>
                <td>12</td>
                <td>RX 6600</td>
                <td>8GB</td>
                <td>~113 TFLOPS</td>
            </tr>
            <tr>
                <td>rig-6600b (192.168.3.154)</td>
                <td>12</td>
                <td>RX 6600</td>
                <td>8GB</td>
                <td>~113 TFLOPS</td>
            </tr>
        </tbody>
    </table>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Recent Runs</div>
    </div>

    <table class="history-table">
        <thead>
            <tr>
                <th>Time</th>
                <th>Step</th>
                <th>Seeds</th>
                <th>Duration</th>
                <th>Throughput</th>
            </tr>
        </thead>
        <tbody>
            {% if state %}
            <tr>
                <td>{{ elapsed_str }} ago</td>
                <td>{{ state.step_name }}</td>
                <td>{{ "{:,}".format(state.seeds_completed) }}</td>
                <td>{{ elapsed_str }}</td>
                <td>{{ "{:,.0f}".format(total_sps) }} s/s</td>
            </tr>
            {% else %}
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-muted);">No recent runs</td>
            </tr>
            {% endif %}
        </tbody>
    </table>
</div>
"""

PLOTS_CONTENT = """
<div class="card">
    <div class="card-header">
        <div class="card-title">Optuna Study Visualization</div>
    </div>

    <form method="GET" action="/plots" style="margin-bottom: 16px;">
        <div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end;">
            <div class="form-group" style="margin-bottom: 0;">
                <label class="form-label">Study</label>
                <select name="study" class="form-select" style="min-width: 280px;">
                    {% for s in studies %}
                    <option value="{{ s.name }}" {% if s.name == selected_study %}selected{% endif %}>
                        {{ s.name }} ({{ s.completed }}/{{ s.trials }} trials)
                    </option>
                    {% endfor %}
                    {% if not studies %}
                    <option value="">No studies found</option>
                    {% endif %}
                </select>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
                <label class="form-label">X-Axis Parameter</label>
                <select name="param_x" class="form-select">
                    {% for p in available_params %}
                    <option value="{{ p }}" {% if p == param_x %}selected{% endif %}>{{ p }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group" style="margin-bottom: 0;">
                <label class="form-label">Y-Axis Parameter</label>
                <select name="param_y" class="form-select">
                    {% for p in available_params %}
                    <option value="{{ p }}" {% if p == param_y %}selected{% endif %}>{{ p }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit" class="btn">Generate Plot</button>
        </div>
    </form>

    {% if heatmap_error %}
    <div style="color: var(--accent-orange); font-size: 12px; margin-bottom: 12px;">{{ heatmap_error }}</div>
    {% endif %}
</div>

<div class="charts-grid">
    <div class="card">
        <div class="card-header">
            <div class="card-title">Parameter Optimization Scatter</div>
            {% if heatmap_chart %}<a href="/plot/heatmap?study={{ selected_study }}&param_x={{ param_x }}&param_y={{ param_y }}" target="_blank" style="font-size: 11px; color: var(--accent-blue);">↗ Full Screen</a>{% endif %}
        </div>
        {% if heatmap_chart %}
        <div class="chart-wrapper">{{ heatmap_chart | safe }}</div>
        {% else %}
        <div class="chart-placeholder">Select a study and parameters above</div>
        {% endif %}
    </div>

    <div class="card">
        <div class="card-header">
            <div class="card-title">Trial Convergence</div>
            {% if convergence_chart %}<a href="/plot/convergence?study={{ selected_study }}" target="_blank" style="font-size: 11px; color: var(--accent-blue);">↗ Full Screen</a>{% endif %}
        </div>
        {% if convergence_chart %}
        <div class="chart-wrapper">{{ convergence_chart | safe }}</div>
        {% else %}
        <div class="chart-placeholder">Trial convergence over time</div>
        {% endif %}
    </div>

    <div class="card">
        <div class="card-header">
            <div class="card-title">Parameter Importance</div>
            {% if importance_chart %}<a href="/plot/importance?study={{ selected_study }}" target="_blank" style="font-size: 11px; color: var(--accent-blue);">↗ Full Screen</a>{% endif %}
        </div>
        {% if importance_chart %}
        <div class="chart-wrapper">{{ importance_chart | safe }}</div>
        {% else %}
        <div class="chart-placeholder">Need 4+ completed trials for importance analysis</div>
        {% endif %}
    </div>

    <div class="card">
        <div class="card-header">
            <div class="card-title">Score Distribution</div>
            {% if distribution_chart %}<a href="/plot/distribution?study={{ selected_study }}" target="_blank" style="font-size: 11px; color: var(--accent-blue);">↗ Full Screen</a>{% endif %}
        </div>
        {% if distribution_chart %}
        <div class="chart-wrapper">{{ distribution_chart | safe }}</div>
        {% else %}
        <div class="chart-placeholder">Score distribution histogram</div>
        {% endif %}
    </div>
</div>
"""

SETTINGS_CONTENT = """
<div class="card">
    <div class="card-header">
        <div class="card-title">Dashboard Settings</div>
    </div>

    <form id="settings-form">
        <div class="form-group">
            <label class="form-label">Auto-refresh Interval</label>
            <select class="form-select" name="refresh_interval" id="refresh_interval">
                <option value="1" {{ 'selected' if settings.refresh_interval == 1 else '' }}>1 second</option>
                <option value="2" {{ 'selected' if settings.refresh_interval == 2 else '' }}>2 seconds</option>
                <option value="5" {{ 'selected' if settings.refresh_interval == 5 else '' }}>5 seconds</option>
                <option value="10" {{ 'selected' if settings.refresh_interval == 10 else '' }}>10 seconds</option>
                <option value="30" {{ 'selected' if settings.refresh_interval == 30 else '' }}>30 seconds</option>
                <option value="0" {{ 'selected' if settings.refresh_interval == 0 else '' }}>Disabled</option>
            </select>
            <small style="color: var(--text-muted);">How often the dashboard auto-refreshes</small>
        </div>

        <div class="form-group">
            <label class="form-label">Theme</label>
            <select class="form-select" name="theme" id="theme">
                <option value="dark" {{ 'selected' if settings.theme == 'dark' else '' }}>Dark (HiveOS)</option>
                <option value="light" {{ 'selected' if settings.theme == 'light' else '' }}>Light</option>
            </select>
        </div>

        <div class="form-group">
            <label class="form-label">Show Offline Workers</label>
            <select class="form-select" name="show_offline_workers" id="show_offline_workers">
                <option value="true" {{ 'selected' if settings.show_offline_workers else '' }}>Yes</option>
                <option value="false" {{ 'selected' if not settings.show_offline_workers else '' }}>No</option>
            </select>
        </div>

        <div class="form-group">
            <label class="form-label">Plot Height (pixels)</label>
            <input type="number" class="form-input" name="plot_height" id="plot_height"
                   value="{{ settings.plot_height }}" min="200" max="800" step="10">
        </div>

        <div class="form-group">
            <label class="form-label">Max History Entries</label>
            <input type="number" class="form-input" name="max_history_entries" id="max_history_entries"
                   value="{{ settings.max_history_entries }}" min="10" max="1000" step="10">
            <small style="color: var(--text-muted);">Number of completed runs to keep in history</small>
        </div>

        <div style="margin-top: 20px; display: flex; gap: 10px;">
            <button type="submit" class="btn btn-primary" id="save-btn">
                💾 Save Settings
            </button>
            <button type="button" class="btn btn-secondary" id="reset-btn">
                🔄 Reset to Defaults
            </button>
        </div>

        <div id="save-status" style="margin-top: 10px; display: none;"></div>
    </form>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Cluster Configuration</div>
    </div>

    <div class="form-group">
        <label class="form-label">Progress File Path</label>
        <input type="text" class="form-input" value="/tmp/cluster_progress.json" readonly>
    </div>

    <div class="form-group">
        <label class="form-label">Optuna Studies Directory</label>
        <input type="text" class="form-input" value="{{ studies_dir }}" readonly>
    </div>

    <div class="form-group">
        <label class="form-label">Settings File</label>
        <input type="text" class="form-input" value="/tmp/dashboard_settings.json" readonly>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">Data Management</div>
    </div>

    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
        <button type="button" class="btn btn-secondary" onclick="clearProgress()">
            🗑️ Clear Progress Data
        </button>
        <button type="button" class="btn btn-secondary" onclick="exportHistory()">
            📤 Export Run History
        </button>
    </div>

    <div id="data-status" style="margin-top: 10px; display: none;"></div>
</div>

<div class="card">
    <div class="card-header">
        <div class="card-title">About</div>
    </div>
    <p style="color: var(--text-secondary); font-size: 12px;">
        PRNG Cluster Dashboard v2.1<br>
        HiveOS-inspired interface for distributed PRNG analysis<br>
        26 GPUs • 3 Nodes • ~285 TFLOPS<br><br>
        <span style="color: var(--text-muted);">
            Session 6: Functional settings with persistence</span><br><br>
        <button type="button" class="btn btn-secondary" onclick="shutdownServer()" style="background: var(--accent-red);">
            ⏻ Shutdown Dashboard</button>
        <script>
        function shutdownServer() {
            if (confirm('Shutdown the dashboard server?')) {
                fetch('/shutdown', {method: 'POST'})
                .then(() => { document.body.innerHTML = '<h1 style="color:#e8e8e8;text-align:center;margin-top:100px;">Dashboard stopped.</h1>'; });
            }
        }
        </script>
        <span style="color: var(--text-muted);">
        </span>
    </p>
</div>

<script>
document.getElementById('settings-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = {
        refresh_interval: parseInt(document.getElementById('refresh_interval').value),
        theme: document.getElementById('theme').value,
        show_offline_workers: document.getElementById('show_offline_workers').value === 'true',
        plot_height: parseInt(document.getElementById('plot_height').value),
        max_history_entries: parseInt(document.getElementById('max_history_entries').value)
    };

    fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        const status = document.getElementById('save-status');
        if (data.success) {
            status.innerHTML = '<span style="color: var(--accent-green);">✓ Settings saved!</span>';
            // Apply theme immediately
            if (formData.theme === 'light') {
                document.body.classList.add('light-theme');
            } else {
                document.body.classList.remove('light-theme');
            }
            status.style.display = 'block';
        } else {
            status.innerHTML = '<span style="color: var(--accent-red);">✗ Error: ' + data.error + '</span>';
            status.style.display = 'block';
        }
        setTimeout(() => { status.style.display = 'none'; }, 3000);
    });
});

document.getElementById('reset-btn').addEventListener('click', function() {
    if (confirm('Reset all settings to defaults?')) {
        fetch('/api/settings/reset', {method: 'POST'})
        .then(r => r.json())
        .then(d => { if (d.success) location.reload(); });
    }
});

function clearProgress() {
    if (confirm('Clear all progress data?')) {
        fetch('/api/clear-progress', {method: 'POST'})
        .then(r => r.json())
        .then(d => {
            const status = document.getElementById('data-status');
            status.innerHTML = d.success ?
                '<span style="color: var(--accent-green);">✓ Cleared</span>' :
                '<span style="color: var(--accent-red);">✗ Error</span>';
            status.style.display = 'block';
            setTimeout(() => { status.style.display = 'none'; }, 3000);
        });
    }
}

function exportHistory() {
    window.location.href = '/api/export-history';
}
</script>
"""

# ============================================================================
# Helper Functions
# ============================================================================

def read_progress():
    """Read progress from JSON file"""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_common_context():
    """Get common template context"""
    # Get real GPU stats
    gpu_stats = get_cluster_gpu_stats() if GPU_MONITOR_AVAILABLE else {}

    state = read_progress()

    if state:
        nodes = state.get("nodes", {})
        total_seeds = state.get("total_seeds", 1)
        seeds_completed = state.get("seeds_completed", 0)
        progress_pct = min(100, (seeds_completed / total_seeds * 100)) if total_seeds > 0 else 0
        total_sps = sum(n.get("current_seeds_per_sec", 0) for n in nodes.values())
        total_gpus = sum(n.get("total_gpus", 0) for n in nodes.values())
        active_gpus = sum(n.get("total_gpus", 0) for n in nodes.values() if n.get("current_seeds_per_sec", 0) > 0)
        elapsed = state.get("elapsed_seconds", 0)

        if state.get("finished"):
            eta_str = "Done"
        elif total_sps > 0:
            remaining = total_seeds - seeds_completed
            eta_seconds = remaining / total_sps
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "--:--"

        elapsed_str = str(timedelta(seconds=int(elapsed)))
    else:
        nodes = {}
        progress_pct = 0
        total_seeds = 0
        total_sps = 0
        total_gpus = 26
        active_gpus = 0
        eta_str = "--:--"
        elapsed_str = "0:00:00"

    # Filter offline workers if setting disabled
    settings_data = load_settings()
    if not settings_data.get('show_offline_workers', True) and nodes:
        nodes = {k: v for k, v in nodes.items() if v.get('current_seeds_per_sec', 0) > 0}

    # Summary bar HTML
    summary_bar = f"""
    <div class="summary-bar">
        <div class="summary-item">
            <div class="summary-value">{len(nodes) if nodes else 0}</div>
            <div class="summary-label">Workers</div>
        </div>
        <div class="summary-item">
            <div class="summary-value green">{total_gpus}</div>
            <div class="summary-label">GPUs</div>
        </div>
        <div class="summary-item">
            <div class="summary-value blue">{total_sps:,.0f}</div>
            <div class="summary-label">Seeds/sec</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{total_seeds:,.0f}</div>
            <div class="summary-label">Total Seeds</div>
        </div>
        <div class="summary-item">
            <div class="summary-value orange">{progress_pct:.0f}%</div>
            <div class="summary-label">Progress</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{eta_str}</div>
            <div class="summary-label">ETA</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{elapsed_str}</div>
            <div class="summary-label">Elapsed</div>
        </div>
    </div>
    """

    return {
        'state': state,
        'nodes': nodes,
        'progress_pct': progress_pct,
        'total_seeds': total_seeds,
        'total_sps': total_sps,
        'total_gpus': total_gpus,
        'active_gpus': active_gpus,
        'eta_str': eta_str,
        'elapsed_str': elapsed_str,
        'summary_bar': summary_bar,
        'gpu_stats': gpu_stats if GPU_MONITOR_AVAILABLE else {},
        'settings': load_settings(),
        'studies_dir': '/home/michael/distributed_prng_analysis/optuna_studies/'
    }


def list_optuna_studies():
    """List all available Optuna studies"""
    studies = []
    study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

    if not os.path.exists(study_path):
        return studies

    for f in os.listdir(study_path):
        if f.endswith('.db'):
            study_name = f.replace('.db', '')
            storage = f"sqlite:///{os.path.join(study_path, f)}"
            try:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.load_study(study_name=study_name, storage=storage)
                completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                studies.append({
                    'name': study_name,
                    'file': f,
                    'trials': len(study.trials),
                    'completed': completed,
                    'params': list(study.trials[0].params.keys()) if study.trials else []
                })
            except Exception as e:
                pass

    # Sort by name (most recent first based on timestamp in name)
    studies.sort(key=lambda x: x['name'], reverse=True)
    return studies


def generate_heatmap(study_name=None, param_x=None, param_y=None):
    """Generate parameter optimization scatter plot from Optuna data"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

        if not study_name:
            # Get most recent study
            studies = [f for f in os.listdir(study_path) if f.endswith('.db')]
            if not studies:
                return None, "No studies found"
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace('.db', '')

        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return None, f"Need at least 2 completed trials (have {len(completed)})"

        # Get available parameters
        params = list(completed[0].params.keys())

        # Default to first two params if not specified
        if not param_x or param_x not in params:
            param_x = params[0] if params else None
        if not param_y or param_y not in params:
            param_y = params[1] if len(params) > 1 else params[0]

        if not param_x or not param_y:
            return None, "No parameters available"

        # Extract data
        x_values = []
        y_values = []
        scores = []

        for trial in completed:
            x_val = trial.params.get(param_x)
            y_val = trial.params.get(param_y)
            if x_val is not None and y_val is not None:
                x_values.append(float(x_val) if isinstance(x_val, (int, float)) else hash(str(x_val)) % 100)
                y_values.append(float(y_val) if isinstance(y_val, (int, float)) else hash(str(y_val)) % 100)
                scores.append(trial.value if trial.value else 0)

        if len(x_values) < 2:
            return None, "Not enough data points"

        # Create figure with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1d21')
        ax.set_facecolor('#2a2e33')

        # Scatter plot with color mapping
        scatter = ax.scatter(x_values, y_values, c=scores, cmap='viridis', s=100, alpha=0.8, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Score', color='#e8e8e8')
        cbar.ax.yaxis.set_tick_params(color='#e8e8e8')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e8e8e8')

        ax.set_xlabel(param_x, color='#e8e8e8', fontsize=11)
        ax.set_ylabel(param_y, color='#e8e8e8', fontsize=11)
        ax.set_title(f'Optuna Study: {study_name}', color='#e8e8e8', fontsize=12)
        ax.tick_params(colors='#8a9099')

        for spine in ax.spines.values():
            spine.set_color('#3a3f45')

        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1d21', edgecolor='none', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode('utf-8'), None

    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}"


def generate_convergence_chart(study_name=None):
    """Generate trial convergence chart"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith('.db')]
            if not studies:
                return None
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace('.db', '')

        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return None

        trial_nums = [t.number for t in completed]
        scores = [t.value if t.value else 0 for t in completed]

        # Calculate running best
        running_best = []
        best_so_far = float('-inf')
        for score in scores:
            best_so_far = max(best_so_far, score)
            running_best.append(best_so_far)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1d21')
        ax.set_facecolor('#2a2e33')

        ax.plot(trial_nums, scores, 'o-', color='#3b82f6', alpha=0.6, label='Trial Score')
        ax.plot(trial_nums, running_best, '-', color='#02e079', linewidth=2, label='Best So Far')

        ax.set_xlabel('Trial Number', color='#e8e8e8')
        ax.set_ylabel('Score', color='#e8e8e8')
        ax.set_title('Trial Convergence', color='#e8e8e8')
        ax.tick_params(colors='#8a9099')
        ax.legend(facecolor='#2a2e33', edgecolor='#3a3f45')

        for spine in ax.spines.values():
            spine.set_color('#3a3f45')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1d21', edgecolor='none', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.read()).decode('utf-8')
    except:
        return None


def generate_distribution_chart(study_name=None):
    """Generate score distribution histogram"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith('.db')]
            if not studies:
                return None
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace('.db', '')

        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return None

        scores = [t.value if t.value else 0 for t in completed]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#1a1d21')
        ax.set_facecolor('#2a2e33')

        sns.histplot(scores, kde=True, color='#3b82f6', ax=ax, alpha=0.7)

        ax.set_xlabel('Score', color='#e8e8e8')
        ax.set_ylabel('Count', color='#e8e8e8')
        ax.set_title('Score Distribution', color='#e8e8e8')
        ax.tick_params(colors='#8a9099')

        for spine in ax.spines.values():
            spine.set_color('#3a3f45')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="#1a1d21", edgecolor="none", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.read()).decode("utf-8")
    except:
        return None


# ============================================================================
# Plotly Interactive Charts
# ============================================================================

def generate_heatmap_plotly(study_name=None, param_x=None, param_y=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)
    """Generate interactive parameter optimization scatter plot using Plotly"""
    try:
        import plotly.graph_objects as go
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith(".db")]
            if not studies:
                return None, "No studies found"
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace(".db", "")

        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return None, f"Need at least 2 completed trials (have {len(completed)})"

        params = list(completed[0].params.keys())

        if not param_x or param_x not in params:
            param_x = params[0] if params else None
        if not param_y or param_y not in params:
            param_y = params[1] if len(params) > 1 else params[0]

        if not param_x or not param_y:
            return None, "No parameters available"

        x_values, y_values, scores, hover_texts = [], [], [], []

        for trial in completed:
            x_val = trial.params.get(param_x)
            y_val = trial.params.get(param_y)
            if x_val is not None and y_val is not None:
                x_values.append(float(x_val) if isinstance(x_val, (int, float)) else hash(str(x_val)) % 100)
                y_values.append(float(y_val) if isinstance(y_val, (int, float)) else hash(str(y_val)) % 100)
                score = trial.value if trial.value else 0
                scores.append(score)
                hover_text = f"<b>Trial {trial.number}</b><br>{param_x}: {x_val}<br>{param_y}: {y_val}<br>Score: {score:.4f}"
                hover_texts.append(hover_text)

        if len(x_values) < 2:
            return None, "Not enough data points"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_values, y=y_values, mode="markers",
            marker=dict(size=14, color=scores, colorscale="Viridis", showscale=True,
                       colorbar=dict(title="Score", tickfont=dict(color="#e8e8e8")),
                       line=dict(width=1, color="white")),
            text=hover_texts, hoverinfo="text", hovertemplate="%{text}<extra></extra>"
        ))
        # Calculate axis ranges with padding (start from 0, add 10% margin on top)
        x_max = max(x_values) * 1.1 if x_values else 100
        y_max = max(y_values) * 1.1 if y_values else 100

        fig.update_layout(autosize=True, height=plot_height,
            title=dict(text=f"Optuna Study: {study_name}", font=dict(color="#e8e8e8")),
            xaxis=dict(title=param_x, color="#8a9099", gridcolor="#3a3f45", range=[0, x_max]),
            yaxis=dict(title=param_y, color="#8a9099", gridcolor="#3a3f45", range=[0, y_max]),
            plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21", font=dict(color="#e8e8e8"),
            hovermode="closest", margin=dict(l=60, r=40, t=60, b=60)
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True}), None
    except Exception as e:
        return None, f"Error: {str(e)}"


def generate_convergence_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)
    """Generate interactive trial convergence chart using Plotly"""
    try:
        import plotly.graph_objects as go
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"
        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith(".db")]
            if not studies: return None
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace(".db", "")
        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2: return None
        trial_nums = [t.number for t in completed]
        scores = [t.value if t.value else 0 for t in completed]
        running_best = []
        best_so_far = float("-inf")
        for score in scores:
            best_so_far = max(best_so_far, score)
            running_best.append(best_so_far)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trial_nums, y=scores, mode="lines+markers", name="Trial Score",
            line=dict(color="#3b82f6"), marker=dict(size=8), hovertemplate="Trial %{x}<br>Score: %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=trial_nums, y=running_best, mode="lines", name="Best So Far",
            line=dict(color="#02e079", width=3), hovertemplate="Trial %{x}<br>Best: %{y:.4f}<extra></extra>"))
        fig.update_layout(autosize=True, height=plot_height, title=dict(text="Trial Convergence", font=dict(color="#e8e8e8")),
            xaxis=dict(title="Trial Number", color="#8a9099", gridcolor="#3a3f45"),
            yaxis=dict(title="Score", color="#8a9099", gridcolor="#3a3f45"),
            plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21", font=dict(color="#e8e8e8"),
            legend=dict(font=dict(color="#e8e8e8")), hovermode="x unified", margin=dict(l=50, r=30, t=40, b=40))
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    except: return None


def generate_distribution_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)
    """Generate interactive score distribution histogram using Plotly"""
    try:
        import plotly.graph_objects as go
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"
        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith(".db")]
            if not studies: return None
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace(".db", "")
        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2: return None
        scores = [t.value if t.value else 0 for t in completed]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=scores, nbinsx=20, marker=dict(color="#3b82f6", line=dict(color="#1a1d21", width=1)),
            hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>"))
        fig.update_layout(autosize=True, height=plot_height, title=dict(text="Score Distribution", font=dict(color="#e8e8e8")),
            xaxis=dict(title="Score", color="#8a9099", gridcolor="#3a3f45"),
            yaxis=dict(title="Count", color="#8a9099", gridcolor="#3a3f45"),
            plot_bgcolor="#2a2e33", paper_bgcolor="#1a1d21", font=dict(color="#e8e8e8"),
            bargap=0.1, margin=dict(l=50, r=30, t=40, b=40))
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    except:
        return None

def generate_importance_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)
    """Generate parameter importance bar chart using Plotly"""
    try:
        import plotly.graph_objects as go
        import optuna
        from optuna.importance import get_param_importances
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_path = "/home/michael/distributed_prng_analysis/optuna_studies/"

        if not study_name:
            studies = [f for f in os.listdir(study_path) if f.endswith(".db")]
            if not studies:
                return None
            study_name = max(studies, key=lambda x: os.path.getmtime(os.path.join(study_path, x))).replace(".db", "")

        storage = f"sqlite:///{os.path.join(study_path, study_name)}.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 4:
            return None

        try:
            importances = get_param_importances(study)
        except:
            return None

        if not importances:
            return None

        # Sort by importance descending
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        params = [item[0] for item in sorted_items]
        values = [item[1] * 100 for item in sorted_items]

        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=params,
            x=values,
            orientation='h',
            marker=dict(color='#3b82f6'),
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            autosize=True,
            title=dict(text="Parameter Importance", font=dict(color="#e8e8e8")),
            xaxis=dict(title="Importance %", color="#8a9099", gridcolor="#3a3f45", range=[0, 100]),
            yaxis=dict(color="#8a9099", autorange="reversed"),
            plot_bgcolor="#2a2e33",
            paper_bgcolor="#1a1d21",
            font=dict(color="#e8e8e8"),
            margin=dict(l=120, r=30, t=40, b=40)
        )

        return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
    except:
        return None


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def overview():
    ctx = get_common_context()
    html = base_template(OVERVIEW_CONTENT, "overview")
    ctx['auto_refresh'] = True
    return render_template_string(html, **ctx)


@app.route('/workers')
def workers():
    ctx = get_common_context()
    html = base_template(WORKERS_CONTENT, "workers")
    return render_template_string(html, **ctx)


@app.route('/stats')
def stats():
    ctx = get_common_context()
    html = base_template(STATS_CONTENT, "stats")
    return render_template_string(html, **ctx)


@app.route('/plots')
def plots():
    ctx = get_common_context()

    # Get available studies
    studies = list_optuna_studies()
    ctx['studies'] = studies

    # Get selected study from query params
    selected_study = request.args.get('study')
    if not selected_study and studies:
        selected_study = studies[0]['name']  # Default to most recent
    ctx['selected_study'] = selected_study

    # Get available params for selected study
    available_params = []
    if selected_study:
        for s in studies:
            if s['name'] == selected_study:
                available_params = s['params']
                break
    ctx['available_params'] = available_params

    # Get selected params from query
    param_x = request.args.get('param_x')
    param_y = request.args.get('param_y')
    if not param_x and available_params:
        param_x = available_params[0]
    if not param_y and len(available_params) > 1:
        param_y = available_params[1]
    elif not param_y and available_params:
        param_y = available_params[0]
    ctx['param_x'] = param_x
    ctx['param_y'] = param_y

    # Generate charts
    heatmap_chart, heatmap_error = generate_heatmap_plotly(selected_study, param_x, param_y)
    ctx['heatmap_chart'] = heatmap_chart
    ctx['heatmap_error'] = heatmap_error
    ctx['convergence_chart'] = generate_convergence_plotly(selected_study)
    ctx['distribution_chart'] = generate_distribution_plotly(selected_study)
    ctx['importance_chart'] = generate_importance_plotly(selected_study)  # TODO: Add parameter importance
    ctx['auto_refresh'] = False

    html = base_template(PLOTS_CONTENT, "plots", auto_refresh=False)
    return render_template_string(html, **ctx)


@app.route('/settings')
def settings():
    ctx = get_common_context()
    html = base_template(SETTINGS_CONTENT, "settings", auto_refresh=False)
    return render_template_string(html, **ctx)


@app.route('/api/progress')
def api_progress():
    state = read_progress()
    if state:
        return jsonify(state)
    return jsonify({"status": "waiting"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})



@app.route('/plot/<plot_type>')
def full_plot(plot_type):
    """Full page plot view with interactive Plotly charts"""
    study_name = request.args.get('study')
    param_x = request.args.get('param_x')
    param_y = request.args.get('param_y')

    chart_html = None
    title = ""

    if plot_type == 'heatmap':
        chart_html, _ = generate_heatmap_plotly(study_name, param_x, param_y)
        title = f"Parameter Optimization - {study_name}"
    elif plot_type == 'convergence':
        chart_html = generate_convergence_plotly(study_name)
        title = f"Trial Convergence - {study_name}"
    elif plot_type == 'distribution':
        chart_html = generate_distribution_plotly(study_name)
        title = f"Score Distribution - {study_name}"
    elif plot_type == 'importance':
        chart_html = generate_importance_plotly(study_name)
        title = f"Parameter Importance - {study_name}"

    if not chart_html:
        return "No data available", 404

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #1a1d21; min-height: 100vh; }}
        h1 {{ color: #e8e8e8; font-family: sans-serif; font-size: 18px; text-align: center; }}
        .chart-container {{ width: 100%; height: calc(100vh - 140px); min-height: 600px; }}
        .actions {{ text-align: center; margin-top: 10px; }}
        .btn {{ padding: 8px 16px; background: #3b82f6; color: white; border-radius: 4px; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="chart-container">{chart_html}</div>
    <div class="actions"><a href="/plots?study={study_name or ''}&param_x={param_x or ''}&param_y={param_y or ''}" class="btn">Back to Plots</a></div>
</body>
</html>"""




# ============================================================================
# Settings API Routes (Added Session 6)
# ============================================================================

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """Get or update settings"""
    if request.method == 'GET':
        return jsonify(load_settings())
    elif request.method == 'POST':
        try:
            new_settings = request.get_json()
            current = load_settings()
            if 'refresh_interval' in new_settings:
                current['refresh_interval'] = max(0, min(60, int(new_settings['refresh_interval'])))
            if 'theme' in new_settings:
                current['theme'] = new_settings['theme'] if new_settings['theme'] in ['dark', 'light'] else 'dark'
            if 'show_offline_workers' in new_settings:
                current['show_offline_workers'] = bool(new_settings['show_offline_workers'])
            if 'plot_height' in new_settings:
                current['plot_height'] = max(200, min(800, int(new_settings['plot_height'])))
            if 'max_history_entries' in new_settings:
                current['max_history_entries'] = max(10, min(1000, int(new_settings['max_history_entries'])))
            save_settings(current)
            return jsonify({"success": True, "settings": current})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})


@app.route('/api/settings/reset', methods=['POST'])
def api_settings_reset():
    """Reset settings to defaults"""
    try:
        save_settings(DEFAULT_SETTINGS.copy())
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/clear-progress', methods=['POST'])
def api_clear_progress():
    """Clear the progress file"""
    try:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/export-history')
def api_export_history():
    """Export run history as JSON download"""
    try:
        history_file = "/tmp/cluster_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                data = f.read()
            response = app.response_class(response=data, status=200, mimetype='application/json')
            response.headers['Content-Disposition'] = 'attachment; filename=cluster_history.json'
            return response
        return jsonify({"error": "No history file found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the dashboard server"""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # For newer Flask/Werkzeug, use os._exit
        import os
        os._exit(0)
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    # Kill any existing process on port 5000
    import subprocess
    subprocess.run(['fuser', '-k', '5000/tcp'], capture_output=True)
    import time
    time.sleep(1)

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  PRNG Cluster Dashboard v2.0                               ║")
    print("║                                                            ║")
    print("║  Local:   http://localhost:5000                            ║")
    print("║  Network: http://192.168.3.127:5000                        ║")
    print("║                                                            ║")
    print("║  Routes:                                                   ║")
    print("║    /         - Overview                                    ║")
    print("║    /workers  - Worker details                              ║")
    print("║    /stats    - Statistics                                  ║")
    print("║    /plots    - Visualizations                              ║")
    print("║    /settings - Configuration                               ║")
    print("║                                                            ║")
    print("║  Press Ctrl+C to stop                                      ║")
    print("╚════════════════════════════════════════════════════════════╝")

    app.run(host='0.0.0.0', port=5000, debug=False)
