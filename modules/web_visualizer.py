#!/usr/bin/env python3
"""
Web-based Results Visualizer for LAN Access
Hosts analysis results via HTTP for remote viewing
"""
import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socket

class ResultsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_dashboard()
        elif path == '/results':
            self.serve_results_list()
        elif path.startswith('/view/'):
            result_file = path[6:]  # Remove '/view/'
            self.serve_result_detail(result_file)
        elif path == '/api/results':
            self.serve_api_results()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>PRNG Analysis Results</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .result-card { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #4CAF50; }
        .result-card.concern { border-left-color: #f44336; }
        .result-card.caution { border-left-color: #ff9800; }
        .meta { color: #888; font-size: 0.9em; }
        .confidence { font-weight: bold; }
        .normal { color: #4CAF50; }
        .concern { color: #f44336; }
        .caution { color: #ff9800; }
        a { color: #2196F3; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .refresh { float: right; }
    </style>
    <script>
        function refreshResults() {
            location.reload();
        }
        setInterval(refreshResults, 30000); // Auto-refresh every 30 seconds
    </script>
</head>
<body>
    <div class="header">
        <h1>PRNG Analysis Dashboard</h1>
        <button onclick="refreshResults()" class="refresh">Refresh Results</button>
        <div class="meta">Auto-refresh every 30 seconds | <a href="/results">All Results</a></div>
    </div>
    <div id="results">Loading results...</div>
    
    <script>
        fetch('/api/results')
            .then(r => r.json())
            .then(data => {
                let html = '';
                data.forEach(result => {
                    let cssClass = 'result-card';
                    let statusClass = 'normal';
                    let status = 'NORMAL';
                    
                    if (result.likely_mt && result.confidence > 0.8) {
                        cssClass += ' concern';
                        statusClass = 'concern';
                        status = 'CONCERN';
                    } else if (result.likely_mt && result.confidence > 0.5) {
                        cssClass += ' caution';
                        statusClass = 'caution';
                        status = 'CAUTION';
                    }
                    
                    html += `
                        <div class="${cssClass}">
                            <h3><a href="/view/${result.filename}">${result.analysis_type || 'Analysis'}</a></h3>
                            <div class="confidence ${statusClass}">Status: ${status} | Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                            <div class="meta">
                                Data Points: ${result.data_points || 'N/A'} | 
                                Method: ${result.conversion_method || result.method || 'N/A'} | 
                                Time: ${new Date(result.timestamp * 1000).toLocaleString()}
                            </div>
                        </div>
                    `;
                });
                document.getElementById('results').innerHTML = html || '<p>No results found</p>';
            })
            .catch(e => {
                document.getElementById('results').innerHTML = '<p>Error loading results</p>';
            });
    </script>
</body>
</html>'''
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_api_results(self):
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        results = []
        
        if os.path.exists(results_dir):
            for filename in sorted(os.listdir(results_dir), reverse=True)[:20]:  # Latest 20
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(results_dir, filename), 'r') as f:
                            data = json.load(f)
                        
                        # Extract key info for dashboard
                        result_info = {
                            'filename': filename,
                            'timestamp': data.get('timestamp', 0),
                            'analysis_type': data.get('analysis_type', 'unknown'),
                            'data_points': data.get('data_points', 0),
                            'conversion_method': data.get('conversion_method', ''),
                            'method': data.get('method', ''),
                            'confidence': data.get('confidence', 0),
                            'likely_mt': False
                        }
                        
                        # Check if MT patterns detected (various result formats)
                        if 'results' in data and isinstance(data['results'], dict):
                            # Multiple method results
                            for method_result in data['results'].values():
                                if method_result.get('likely_mt', False):
                                    result_info['likely_mt'] = True
                                    break
                        elif data.get('likely_mt', False):
                            result_info['likely_mt'] = True
                        
                        results.append(result_info)
                    except:
                        continue
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(results).encode())
    
    def serve_result_detail(self, filename):
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        filepath = os.path.join(results_dir, filename)
        
        if not os.path.exists(filepath) or not filename.endswith('.json'):
            self.send_error(404)
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Result: {filename}</title>
    <style>
        body {{ font-family: monospace; margin: 20px; background: #1a1a1a; color: #fff; }}
        .header {{ background: #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        pre {{ background: #2a2a2a; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        a {{ color: #2196F3; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{filename}</h2>
        <a href="/">← Back to Dashboard</a>
    </div>
    <pre>{json.dumps(data, indent=2)}</pre>
</body>
</html>'''
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        except:
            self.send_error(500)

class WebVisualizer:
    def __init__(self, core):
        self.core = core
        self.module_name = "WebVisualizer"
        self.server = None
        self.server_thread = None
        self.port = 8080
        
    def start_server(self, port=8080):
        """Start the web server"""
        self.port = port
        try:
            self.server = HTTPServer(('0.0.0.0', port), ResultsHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            # Get local IP
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            print(f"✓ Web visualizer started successfully")
            print(f"  Local access: http://localhost:{port}")
            print(f"  LAN access: http://{local_ip}:{port}")
            print(f"  Server running on all interfaces (0.0.0.0:{port})")
            return True
        except Exception as e:
            print(f"✗ Failed to start web server: {e}")
            return False
    
    def stop_server(self):
        """Stop the web server"""
        if self.server:
            self.server.shutdown()
            self.server = None
            print("✓ Web server stopped")
    
    def menu(self):
        """Web visualizer menu"""
        while True:
            if hasattr(self.core, "clear_screen"):
                self.core.clear_screen()
            if hasattr(self.core, "print_header"):
                self.core.print_header()
            
            print(f"\n{self.module_name.upper()}")
            print("-" * 35)
            print("WEB VISUALIZER OPTIONS:")
            
            if self.server:
                print("  Status: ✓ RUNNING")
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                print(f"  LAN URL: http://{local_ip}:{self.port}")
                print()
                print("  1. View Dashboard (show URL)")
                print("  2. Stop Web Server")
                print("  3. Restart Server (different port)")
            else:
                print("  Status: ✗ STOPPED")
                print()
                print("  1. Start Web Server")
                print("  2. Start with Custom Port")
            
            print("  4. Back to Main Menu")
            print("-" * 35)
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                if self.server:
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                    print(f"\n🌐 Web Dashboard URLs:")
                    print(f"  Local: http://localhost:{self.port}")
                    print(f"  LAN: http://{local_ip}:{self.port}")
                    print(f"\nAccess from any device on your network!")
                    input("\nPress Enter to continue...")
                else:
                    if self.start_server():
                        input("\nPress Enter to continue...")
            elif choice == '2':
                if self.server:
                    self.stop_server()
                else:
                    try:
                        port = int(input("Enter port number (default 8080): ") or "8080")
                        self.start_server(port)
                    except ValueError:
                        print("Invalid port number")
                input("Press Enter to continue...")
            elif choice == '3' and self.server:
                self.stop_server()
                try:
                    port = int(input("Enter new port number: "))
                    self.start_server(port)
                except ValueError:
                    print("Invalid port number")
                input("Press Enter to continue...")
            elif choice == '4':
                break
            else:
                print("Invalid choice")
                input("Press Enter to continue...")
    
    def shutdown(self):
        """Module shutdown"""
        if self.server:
            self.stop_server()
        print(f"Shutting down {self.module_name}...")
