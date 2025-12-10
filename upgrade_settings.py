#!/usr/bin/env python3
"""
Settings Page Upgrade for web_dashboard.py
==========================================
Run on Zeus: python3 upgrade_settings.py

This script:
1. Backs up the current dashboard
2. Adds settings persistence functions
3. Replaces SETTINGS_CONTENT with functional form
4. Adds API routes for settings
5. Updates context to include settings
"""

import re
import os
import shutil
from datetime import datetime

DASHBOARD_FILE = "/home/michael/distributed_prng_analysis/web_dashboard.py"

# =============================================================================
# NEW CODE TO INSERT
# =============================================================================

SETTINGS_FUNCTIONS = '''
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

'''

BUTTON_CSS = '''
.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
}

.btn-primary {
    background: var(--accent-green);
    color: #000;
}

.btn-primary:hover {
    background: #00c868;
}

.btn-secondary {
    background: var(--bg-hover);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: var(--border-color);
}

.form-input {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 13px;
}

.form-input:focus {
    outline: none;
    border-color: var(--accent-blue);
}

.form-input[readonly] {
    color: var(--text-muted);
    cursor: not-allowed;
}
'''

NEW_SETTINGS_CONTENT = '''SETTINGS_CONTENT = """
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
            Session 6: Functional settings with persistence
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
"""'''

API_ROUTES = '''

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

'''


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Settings Page Upgrade for web_dashboard.py                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    if not os.path.exists(DASHBOARD_FILE):
        print(f"✗ Error: {DASHBOARD_FILE} not found")
        return False
    
    # Backup
    backup_file = f"{DASHBOARD_FILE}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(DASHBOARD_FILE, backup_file)
    print(f"✓ Backup created: {backup_file}")
    
    # Read content
    with open(DASHBOARD_FILE, 'r') as f:
        content = f.read()
    
    changes = 0
    
    # 1. Add settings functions after PROGRESS_FILE line
    if 'SETTINGS_FILE = ' not in content:
        content = content.replace(
            'PROGRESS_FILE = "/tmp/cluster_progress.json"',
            'PROGRESS_FILE = "/tmp/cluster_progress.json"' + SETTINGS_FUNCTIONS
        )
        print("✓ Added settings functions")
        changes += 1
    else:
        print("• Settings functions already exist")
    
    # 2. Add button CSS - find last } before closing """ in BASE_CSS
    if '.btn {' not in content:
        # Find BASE_CSS and add button styles before the closing
        base_css_match = re.search(r'(BASE_CSS = """.*?)(""")', content, re.DOTALL)
        if base_css_match:
            content = content.replace(
                base_css_match.group(0),
                base_css_match.group(1) + BUTTON_CSS + '\n"""'
            )
            print("✓ Added button CSS")
            changes += 1
    else:
        print("• Button CSS already exists")
    
    # 3. Replace SETTINGS_CONTENT
    pattern = r'SETTINGS_CONTENT = """.*?"""'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, NEW_SETTINGS_CONTENT, content, flags=re.DOTALL)
        print("✓ Replaced SETTINGS_CONTENT with functional form")
        changes += 1
    
    # 4. Add settings to context - both active and waiting states
    if '"settings": load_settings()' not in content:
        # Active state context
        if '"elapsed": elapsed_str,' in content:
            content = content.replace(
                '"elapsed": elapsed_str,',
                '"elapsed": elapsed_str,\n            "settings": load_settings(),\n            "studies_dir": OPTUNA_DIR,'
            )
            print("✓ Added settings to active context")
            changes += 1
        
        # Waiting state context
        if '"elapsed": "--:--",' in content:
            content = content.replace(
                '"elapsed": "--:--",',
                '"elapsed": "--:--",\n            "settings": load_settings(),\n            "studies_dir": OPTUNA_DIR,'
            )
            print("✓ Added settings to waiting context")
            changes += 1
    else:
        print("• Settings already in context")
    
    # 5. Add API routes - find the if __name__ block and add before it
    if '@app.route(\'/api/settings\'' not in content:
        if 'if __name__ ==' in content:
            content = content.replace(
                "if __name__ ==",
                API_ROUTES + "\nif __name__ =="
            )
            print("✓ Added API routes")
            changes += 1
    else:
        print("• API routes already exist")
    
    # Write back
    with open(DASHBOARD_FILE, 'w') as f:
        f.write(content)
    
    print()
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Upgrade complete! {changes} changes applied                        ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()
    print("Restart the dashboard:")
    print("  pkill -f web_dashboard.py")
    print("  sleep 1")
    print("  python3 ~/distributed_prng_analysis/web_dashboard.py &")
    print()
    print("Then visit: http://192.168.3.127:5000/settings")
    
    return True


if __name__ == "__main__":
    main()
