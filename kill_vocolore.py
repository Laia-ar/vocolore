#!/usr/bin/env python3
"""
Kill all Vocolore-related processes (emergency cleanup).
Use this if the UIs leave orphaned subprocesses.
"""

import subprocess
import sys
import os


def kill_vocolore_processes():
    """Kill all vocolore-related Python processes."""
    processes_to_kill = [
        "wifi_transcribe.py",
        "wifi_debug_ui.py", 
        "wifi_user_ui.py",
        "run_wifi_and_ui.py"
    ]
    
    killed = []
    
    try:
        # Get list of Python processes
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.splitlines():
            for proc_name in processes_to_kill:
                if proc_name in line and "kill_vocolore" not in line:
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            # Don't kill ourselves
                            if pid != os.getpid():
                                subprocess.run(["kill", "-9", str(pid)], capture_output=True)
                                killed.append(f"{proc_name} (PID: {pid})")
                        except (ValueError, IndexError):
                            pass
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    if killed:
        print("Killed processes:")
        for k in killed:
            print(f"  - {k}")
    else:
        print("No Vocolore processes found.")
    
    return True


if __name__ == "__main__":
    print("Vocolore Process Cleanup")
    print("=" * 40)
    kill_vocolore_processes()
