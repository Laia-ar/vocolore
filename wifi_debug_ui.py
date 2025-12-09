"""
Debug-oriented Tkinter UI for WiFi transcription.
- Start/stop wifi_transcribe.py
- Live toggles and numeric settings written to the runtime config file
- Full log output viewer
"""

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

DEBUG = os.getenv("DEBUG", "0") == "1"


class DebugUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WiFi Transcribe (Debug)")

        self.runtime_config = os.getenv("RUNTIME_CONFIG_FILE", ".runtime_config.json")
        self.proc = None
        self.reader_thread = None
        self.msg_queue: "queue.Queue[str]" = queue.Queue()
        self.running = False

        cfg = self._load_runtime_config()

        # Controls
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", padx=8, pady=4)
        self.start_btn = tk.Button(btn_frame, text="Start", width=10, command=self.start_proc)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(btn_frame, text="Stop", width=10, command=self.stop_proc, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(btn_frame, textvariable=self.status_var, anchor="w").pack(side="left", padx=8)

        # Toggles and numeric settings
        opt_frame = tk.Frame(root)
        opt_frame.pack(fill="x", padx=8, pady=4)
        self.var_freepik = tk.BooleanVar(value=cfg.get("ENABLE_FREEPIK", False))
        self.var_open = tk.BooleanVar(value=cfg.get("OPEN_IMAGE", False))
        self.var_print = tk.BooleanVar(value=cfg.get("PRINT_IMAGE", False))
        self.var_debug = tk.BooleanVar(value=cfg.get("DEBUG_TIMING", False))
        self.chk_freepik = tk.Checkbutton(opt_frame, text="Freepik", variable=self.var_freepik, command=self.apply_config)
        self.chk_freepik.pack(side="left", padx=4)
        self.chk_open = tk.Checkbutton(opt_frame, text="Open image", variable=self.var_open, command=self.apply_config)
        self.chk_open.pack(side="left", padx=4)
        self.chk_print = tk.Checkbutton(opt_frame, text="Print image", variable=self.var_print, command=self.apply_config)
        self.chk_print.pack(side="left", padx=4)
        tk.Checkbutton(opt_frame, text="Debug timing", variable=self.var_debug, command=self.apply_config).pack(side="left", padx=4)

        num_frame = tk.Frame(root)
        num_frame.pack(fill="x", padx=8, pady=4)
        tk.Label(num_frame, text="MIN_AUDIO_SEC").pack(side="left")
        self.entry_min = tk.Entry(num_frame, width=8)
        self.entry_min.pack(side="left", padx=4)
        self.entry_min.insert(0, str(cfg.get("MIN_AUDIO_SEC", 4.0)))
        tk.Label(num_frame, text="MAX_BUFFER_SEC").pack(side="left")
        self.entry_max = tk.Entry(num_frame, width=8)
        self.entry_max.pack(side="left", padx=4)
        self.entry_max.insert(0, str(cfg.get("MAX_BUFFER_SEC", 16.0)))
        tk.Button(num_frame, text="Apply", command=self.apply_config).pack(side="left", padx=6)

        # Log
        self.log = scrolledtext.ScrolledText(root, height=28, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

        self.root.after(100, self.drain_messages)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.apply_config()
        if os.getenv("AUTO_START", "1") == "1":
            self.start_proc()

    def _load_runtime_config(self):
        if os.path.isfile(self.runtime_config):
            try:
                with open(self.runtime_config, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def _write_runtime_config(self):
        cfg = {
            "ENABLE_FREEPIK": self.var_freepik.get(),
            "OPEN_IMAGE": self.var_open.get() and self.var_freepik.get(),
            "PRINT_IMAGE": self.var_print.get() and self.var_freepik.get(),
            "DEBUG_TIMING": self.var_debug.get(),
            "RUNNING": self.running,
        }
        try:
            cfg["MIN_AUDIO_SEC"] = float(self.entry_min.get())
        except ValueError:
            cfg["MIN_AUDIO_SEC"] = 0.3
        try:
            cfg["MAX_BUFFER_SEC"] = float(self.entry_max.get())
        except ValueError:
            cfg["MAX_BUFFER_SEC"] = 16.0
        try:
            with open(self.runtime_config, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
        except Exception as exc:
            self.append_log(f"Failed to write runtime config: {exc}\n")
        # enforce UI state for open/print based on freepik toggle
        state = tk.NORMAL if self.var_freepik.get() else tk.DISABLED
        self.chk_open.config(state=state)
        self.chk_print.config(state=state)
        if not self.var_freepik.get():
            self.var_open.set(False)
            self.var_print.set(False)

    def apply_config(self):
        self._write_runtime_config()
        self.append_log("Config applied.\n")

    def append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")
        if DEBUG:
            print(text, end="", flush=True)

    def start_proc(self):
        if self.running:
            return
        self._write_runtime_config()
        cmd = [sys.executable, "wifi_transcribe.py"]
        env = os.environ.copy()
        env["ENABLE_FREEPIK"] = "1" if self.var_freepik.get() else "0"
        env["OPEN_IMAGE"] = "1" if self.var_open.get() else "0"
        env["PRINT_IMAGE"] = "1" if self.var_print.get() else "0"
        env["DEBUG_TIMING"] = "1" if self.var_debug.get() else "0"
        env["MIN_AUDIO_SEC"] = str(self.entry_min.get())
        env["MAX_BUFFER_SEC"] = str(self.entry_max.get())
        env["RUNTIME_CONFIG_FILE"] = self.runtime_config
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self.append_log(f"Failed to start: {exc}\n")
            self.proc = None
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("Running")
        self._write_runtime_config()
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()

    def stop_proc(self):
        if not self.running or not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped")
        self._write_runtime_config()

    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.msg_queue.put(line)
        rc = self.proc.wait()
        self.msg_queue.put(f"\n[process exited with code {rc}]\n")
        self.running = False

    def drain_messages(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.append_log(msg)
        except queue.Empty:
            pass

        if not self.running and self.start_btn["state"] == "disabled":
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if self.proc:
                self.status_var.set("Stopped")

        self.root.after(100, self.drain_messages)

    def on_close(self):
        self.stop_proc()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DebugUI(root)
    root.mainloop()
