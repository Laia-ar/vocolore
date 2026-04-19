"""
Launch both UIs together (wifi_debug_ui.py and wifi_user_ui.py).
When either UI exits, the other is terminated.
"""

import os
import subprocess
import sys
import time
import signal
import atexit


def main():
    procs = []
    
    def cleanup_all():
        """Terminate all child processes."""
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
    
    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        print(f"\nReceived signal {signum}, cleaning up...")
        cleanup_all()
        sys.exit(0)
    
    # Register cleanup handlers
    atexit.register(cleanup_all)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        env_debug = os.environ.copy()
        env_debug.setdefault("DEBUG", "1")
        env_debug.setdefault("AUTO_START", "1")
        env_user = os.environ.copy()
        env_user.setdefault("ENABLE_FREEPIK", "1")
        env_user.setdefault("OPEN_IMAGE", "1")
        env_user.setdefault("LAUNCH_TRANSCRIBE", "0")  # let debug UI start it
        
        debug_proc = subprocess.Popen([sys.executable, "wifi_debug_ui.py"], env=env_debug)
        user_proc = subprocess.Popen([sys.executable, "wifi_user_ui.py"], env=env_user)
        procs.extend([debug_proc, user_proc])

        # Wait for either UI to exit
        while True:
            if debug_proc.poll() is not None or user_proc.poll() is not None:
                break
            time.sleep(0.5)
    finally:
        cleanup_all()


if __name__ == "__main__":
    main()
