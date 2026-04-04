#!/usr/bin/env python3
"""
Netconsole UDP listener for Zeus.
Receives kernel messages from all rigs and logs them.
Run as: python3 netconsole_listener.py
Or install as systemd service using install_netconsole_zeus.sh
"""
import socket
import datetime
import os
import signal
import sys

LOG_DIR = "/home/michael/distributed_prng_analysis/logs"
LOG_FILE = os.path.join(LOG_DIR, "netconsole_all_rigs.log")
LISTEN_PORT = 6667

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', LISTEN_PORT))

    def shutdown(sig, frame):
        print("Shutting down listener", flush=True)
        s.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"Netconsole listener started on port {LISTEN_PORT}", flush=True)

    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.datetime.now()} LISTENER STARTED\n")
        f.flush()
        while True:
            try:
                data, addr = s.recvfrom(65535)
                msg = f"{datetime.datetime.now()} {addr[0]}: {data.decode(errors='?')}"
                f.write(msg)
                f.flush()
                print(msg, end='', flush=True)
            except Exception as e:
                f.write(f"{datetime.datetime.now()} ERROR: {e}\n")
                f.flush()

if __name__ == "__main__":
    main()
