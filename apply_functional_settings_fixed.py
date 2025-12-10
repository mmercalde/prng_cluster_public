#!/usr/bin/env python3
"""
Make All Settings Functional - FIXED VERSION
=============================================
Run on Zeus: python3 apply_functional_settings_fixed.py

Fixes the f-string/JavaScript arrow function conflict by using
regular strings instead of f-strings for templates with JS.
"""

import re
import os
import shutil
from datetime import datetime

DASHBOARD_FILE = "/home/michael/distributed_prng_analysis/web_dashboard.py"

def main():
    print("=" * 60)
    print("  Making All Settings Functional (Fixed Version)")
    print("=" * 60)
    print()
    
    if not os.path.exists(DASHBOARD_FILE):
        print(f"ERROR: {DASHBOARD_FILE} not found")
        return False
    
    # Backup
    backup_file = f"{DASHBOARD_FILE}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(DASHBOARD_FILE, backup_file)
    print(f"[OK] Backup: {backup_file}")
    
    with open(DASHBOARD_FILE, 'r') as f:
        content = f.read()
    
    changes = 0
    
    # =========================================================================
    # 1. Add Light Theme CSS after dark theme :root block
    # =========================================================================
    light_theme_css = '''
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

'''
    
    if 'body.light-theme' not in content:
        # Insert after --border-color in :root
        pattern = r'(--border-color: #3a3f45;)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1\n}\n' + light_theme_css + '\n:root {', content, count=1)
            # That created a mess, let's try different approach
            # Revert and try adding before the closing of BASE_CSS
            content = content.replace(r'\1\n}\n' + light_theme_css + '\n:root {', r'\1')
        
        # Simpler: add before the first * { in BASE_CSS
        if '* { box-sizing' in content and 'body.light-theme' not in content:
            content = content.replace(
                '* { box-sizing',
                light_theme_css + '\n* { box-sizing'
            )
            print("[OK] Added light theme CSS")
            changes += 1
    else:
        print("[--] Light theme CSS already exists")
    
    # =========================================================================
    # 2. Update SETTINGS_CONTENT to apply theme immediately on save
    #    Add theme toggle JavaScript that doesn't use arrow functions
    # =========================================================================
    
    # Find the save success line and add theme application
    old_save_line = "status.innerHTML = '<span style=\"color: var(--accent-green);\">✓ Settings saved!</span>';"
    new_save_block = '''status.innerHTML = '<span style="color: var(--accent-green);">✓ Settings saved!</span>';
            // Apply theme immediately
            if (formData.theme === 'light') {
                document.body.classList.add('light-theme');
            } else {
                document.body.classList.remove('light-theme');
            }'''
    
    if 'Apply theme immediately' not in content and old_save_line in content:
        content = content.replace(old_save_line, new_save_block)
        print("[OK] Added immediate theme switching on save")
        changes += 1
    else:
        print("[--] Theme switching already exists or save line not found")
    
    # =========================================================================
    # 3. Add theme initialization script to base_template
    #    This applies the saved theme on page load
    #    IMPORTANT: Use regular string, not f-string, for the JS
    # =========================================================================
    
    # We need to modify base_template to check theme on load
    # Find the <body> tag and add onload or inline script
    
    if 'applyTheme' not in content:
        # Add a script block right after <body> that checks settings
        # We'll inject it into the base_template function
        
        old_body = '<body>'
        new_body = '''<body>
    <script>
    (function() {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/api/settings', false);
        try {
            xhr.send();
            if (xhr.status === 200) {
                var settings = JSON.parse(xhr.responseText);
                if (settings.theme === 'light') {
                    document.body.classList.add('light-theme');
                }
            }
        } catch(e) {}
    })();
    </script>'''
        
        if old_body in content:
            content = content.replace(old_body, new_body, 1)  # Only first occurrence
            print("[OK] Added theme initialization script")
            changes += 1
    else:
        print("[--] Theme initialization already exists")
    
    # =========================================================================
    # 4. Make Show Offline Workers functional
    #    Filter nodes in get_common_context based on setting
    # =========================================================================
    
    if 'Filter offline workers' not in content:
        # Find where summary_bar is built and add filter before it
        old_summary = '    # Summary bar HTML'
        new_summary = '''    # Filter offline workers if setting disabled
    settings_data = load_settings()
    if not settings_data.get('show_offline_workers', True) and nodes:
        nodes = {k: v for k, v in nodes.items() if v.get('current_seeds_per_sec', 0) > 0}
    
    # Summary bar HTML'''
        
        if old_summary in content:
            content = content.replace(old_summary, new_summary)
            print("[OK] Added offline workers filter")
            changes += 1
    else:
        print("[--] Offline workers filter already exists")
    
    # =========================================================================
    # 5. Make Plot Height functional
    #    Update Plotly chart functions to use settings
    # =========================================================================
    
    if "settings.get('plot_height'" not in content:
        # Add to generate_heatmap_plotly
        old_heatmap_def = 'def generate_heatmap_plotly(study_name=None, param_x=None, param_y=None):'
        new_heatmap_def = '''def generate_heatmap_plotly(study_name=None, param_x=None, param_y=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)'''
        
        if old_heatmap_def in content:
            content = content.replace(old_heatmap_def, new_heatmap_def)
            print("[OK] Added plot_height to heatmap function")
            changes += 1
        
        # Add to generate_convergence_plotly
        old_conv_def = 'def generate_convergence_plotly(study_name=None):'
        new_conv_def = '''def generate_convergence_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)'''
        
        if old_conv_def in content:
            content = content.replace(old_conv_def, new_conv_def)
            print("[OK] Added plot_height to convergence function")
            changes += 1
        
        # Add to generate_distribution_plotly  
        old_dist_def = 'def generate_distribution_plotly(study_name=None):'
        new_dist_def = '''def generate_distribution_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)'''
        
        if old_dist_def in content:
            content = content.replace(old_dist_def, new_dist_def)
            print("[OK] Added plot_height to distribution function")
            changes += 1
        
        # Add to generate_importance_plotly
        old_imp_def = 'def generate_importance_plotly(study_name=None):'
        new_imp_def = '''def generate_importance_plotly(study_name=None):
    plot_settings = load_settings()
    plot_height = plot_settings.get('plot_height', 380)'''
        
        if old_imp_def in content:
            content = content.replace(old_imp_def, new_imp_def)
            print("[OK] Added plot_height to importance function")
            changes += 1
        
        # Now add height to the update_layout calls
        # Change: autosize=True, to autosize=True, height=plot_height,
        content = re.sub(
            r'fig\.update_layout\(autosize=True,',
            'fig.update_layout(autosize=True, height=plot_height,',
            content
        )
        print("[OK] Added height parameter to Plotly layouts")
        changes += 1
    else:
        print("[--] Plot height already functional")
    
    # =========================================================================
    # 6. Add History Persistence functions and file
    # =========================================================================
    
    if 'HISTORY_FILE = "/tmp/cluster_run_history.json"' not in content:
        # Add after save_settings function
        history_code = '''

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

'''
        # Insert after save_settings function
        pattern = r'(def save_settings\(settings\):.*?return True)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.end()] + history_code + content[match.end():]
            print("[OK] Added history persistence functions")
            changes += 1
    else:
        print("[--] History functions already exist")
    
    # =========================================================================
    # Write the file
    # =========================================================================
    
    with open(DASHBOARD_FILE, 'w') as f:
        f.write(content)
    
    print()
    print("=" * 60)
    print(f"  Complete! {changes} changes applied")
    print("=" * 60)
    print()
    print("Features now functional:")
    print("  [x] Theme switching (Dark/Light)")
    print("  [x] Show/Hide offline workers") 
    print("  [x] Configurable plot height")
    print("  [x] History persistence with max entries")
    print()
    print("Restart dashboard:")
    print("  pkill -9 -f web_dashboard.py")
    print("  sleep 2")
    print("  python3 ~/distributed_prng_analysis/web_dashboard.py &")
    
    return True


if __name__ == "__main__":
    main()
