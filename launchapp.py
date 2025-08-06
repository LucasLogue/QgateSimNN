import webview
import threading
import subprocess
import time
import os
import signal
import sys

def start_streamlit():
    # Optional: kill any previous streamlit instances on port 8501
    subprocess.call("fuser -k 8501/tcp", shell=True)

    # Run Streamlit in the background, headless (no auto-open browser)
    subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if os.name != 'nt' else None  # for clean shutdown on Linux
    )

    # Wait a few seconds for Streamlit server to boot
    time.sleep(3)

if __name__ == "__main__":
    threading.Thread(target=start_streamlit, daemon=True).start()

    # Make the pop-out window with pywebview
    try:
        webview.create_window("Quantum Dot Gate Optimizer", "http://localhost:8501",
                              width=1280, height=800, resizable=True)
    except KeyboardInterrupt:
        print("\n[!] Closing window... bye bye")
        sys.exit(0)