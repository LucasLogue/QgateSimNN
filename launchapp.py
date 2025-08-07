import webview
import threading
import subprocess
import time
import os
import sys
import platform
import socket
import re

PORT = 8600  # Change this to anything you want

def wait_for_port(host: str, port: int, timeout: int = 20):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.5)

def start_streamlit():
    print(f"[+] Launching Streamlit server on port {PORT}...")

    subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.headless", "true", f"--server.port={PORT}"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
    )

    if not wait_for_port("localhost", PORT, timeout=20):
        print(f"❌ Streamlit server failed to start on port {PORT} within timeout.")
        sys.exit(1)
    print(f"✅ Streamlit server is up on http://localhost:{PORT}")

if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()

    # Small grace delay before launching GUI
    time.sleep(1.5)

    # Create the pop-out window
    webview.create_window("Quantum Dot Gate Optimizer", f"http://localhost:{PORT}",
                          width=1280, height=800, resizable=True)

    # Start the GUI using Edge backend
    webview.start(gui='edgechromium')#, debug=True)
