"""
Launch both UIs together (wifi_debug_ui.py and wifi_user_ui.py).
When either UI exits, the other is terminated.
"""

import os
import subprocess
import sys
import time


def main():
    procs = []
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
        for p in procs:
            if p.poll() is None:
                p.kill()


if __name__ == "__main__":
    main()
