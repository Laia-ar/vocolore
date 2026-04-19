"""
Debug-oriented Tkinter UI for WiFi transcription.
- Start/stop wifi_transcribe.py
- Live toggles and numeric settings written to the runtime config file
- Full log output viewer
"""

import atexit
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

DEBUG = os.getenv("DEBUG", "0") == "1"


def cleanup_process(proc):
    """Ensure process is terminated."""
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                proc.kill()
                proc.wait(timeout=1)
            except ProcessLookupError:
                pass


class DebugUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WiFi Transcribe (Debug)")

        self.runtime_config = os.getenv("RUNTIME_CONFIG_FILE", ".runtime_config.json")
        self.proc = None
        self.reader_thread = None
        self.msg_queue: "queue.Queue[str]" = queue.Queue()
        self.running = False
        self._cleanup_registered = False
        
        # Register cleanup on exit
        atexit.register(self._ensure_cleanup)
        
        # Handle signals
        def signal_handler(signum, frame):
            self.on_close()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

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
        
        # Battery indicator
        self.battery_var = tk.StringVar(value="Bat: --")
        self.battery_lbl = tk.Label(btn_frame, textvariable=self.battery_var, anchor="w", fg="#666666")
        self.battery_lbl.pack(side="right", padx=8)

        # Toggles and numeric settings
        opt_frame = tk.Frame(root)
        opt_frame.pack(fill="x", padx=8, pady=4)
        self.var_freepik = tk.BooleanVar(value=cfg.get("ENABLE_FREEPIK", False))
        self.var_open = tk.BooleanVar(value=cfg.get("OPEN_IMAGE", False))
        self.var_print = tk.BooleanVar(value=cfg.get("PRINT_IMAGE", False))
        self.var_pdf = tk.BooleanVar(value=cfg.get("PRINT_TO_PDF", False))
        self.var_debug = tk.BooleanVar(value=cfg.get("DEBUG_TIMING", False))
        self.page_size_var = tk.StringVar(value=cfg.get("PRINT_PAGE_SIZE", "A4"))
        self.provider_var = tk.StringVar(value=cfg.get("IMAGE_PROVIDER", "freepik"))
        self.model_var = tk.StringVar(value=cfg.get("FREEPIK_MODEL", "gemini-2-5-flash-image-preview"))
        self.gemini_model_var = tk.StringVar(value=cfg.get("GEMINI_MODEL", "gemini-3.1-flash-image-preview"))
        
        self.chk_freepik = tk.Checkbutton(opt_frame, text="Gen Img", variable=self.var_freepik, command=self.apply_config)
        self.chk_freepik.pack(side="left", padx=4)
        self.chk_open = tk.Checkbutton(opt_frame, text="Open image", variable=self.var_open, command=self.apply_config)
        self.chk_open.pack(side="left", padx=4)
        self.chk_print = tk.Checkbutton(opt_frame, text="Print image", variable=self.var_print, command=self.apply_config)
        self.chk_print.pack(side="left", padx=4)
        self.chk_pdf = tk.Checkbutton(opt_frame, text="PDF copy", variable=self.var_pdf, command=self.apply_config)
        self.chk_pdf.pack(side="left", padx=4)
        tk.Checkbutton(opt_frame, text="Debug timing", variable=self.var_debug, command=self.apply_config).pack(side="left", padx=4)
        
        # Page size selector
        tk.Label(opt_frame, text="Page:").pack(side="left", padx=(12, 4))
        self.page_size_combo = tk.OptionMenu(opt_frame, self.page_size_var, "A4", "A5", "A6", command=lambda _: self.apply_config())
        self.page_size_combo.pack(side="left", padx=4)
        
        # Provider selector
        tk.Label(opt_frame, text="Provider:").pack(side="left", padx=(12, 4))
        self.provider_combo = tk.OptionMenu(opt_frame, self.provider_var, "freepik", "gemini", command=lambda _: self.on_provider_change())
        self.provider_combo.pack(side="left", padx=4)
        
        # Model selector frame (to swap based on provider)
        self.model_frame = tk.Frame(opt_frame)
        self.model_frame.pack(side="left", padx=4)
        tk.Label(self.model_frame, text="Model:").pack(side="left")
        
        # Freepik models
        freepik_models = [
            "gemini-2-5-flash-image-preview",
            "mystic",
            "flux-kontext-pro",
            "flux-2-pro",
            "flux-2-turbo",
            "flux-2-klein",
            "seedream-v4-5",
            "seedream-v4",
            "z-image",
        ]
        self.freepik_model_combo = tk.OptionMenu(self.model_frame, self.model_var, *freepik_models, command=lambda _: self.apply_config())
        self.freepik_model_combo.pack(side="left", padx=4)
        
        # Gemini models (using actual Google API model names)
        gemini_models = [
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image",
            "gemini-2.0-flash-preview",
        ]
        self.gemini_model_combo = tk.OptionMenu(self.model_frame, self.gemini_model_var, *gemini_models, command=lambda _: self.apply_config())
        # Initially hidden, shown when gemini selected

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
        self.root.after(1000, self._update_battery)  # Update battery every second
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.on_provider_change()  # Initialize correct model selector
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

    def _update_battery(self):
        """Update battery level indicator from runtime config."""
        try:
            cfg = self._load_runtime_config()
            level = cfg.get("BATTERY_LEVEL")
            if level is not None:
                try:
                    pct = int(level)
                    if pct <= 20:
                        self.battery_var.set(f"Bat: {pct}%")
                        self.battery_lbl.config(fg="#d9534f")  # Red (low)
                    elif pct <= 50:
                        self.battery_var.set(f"Bat: {pct}%")
                        self.battery_lbl.config(fg="#f0ad4e")  # Orange (medium)
                    else:
                        self.battery_var.set(f"Bat: {pct}%")
                        self.battery_lbl.config(fg="#5cb85c")  # Green (good)
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass
        # Schedule next update
        self.root.after(1000, self._update_battery)

    def _write_runtime_config(self):
        cfg = self._load_runtime_config()
        if not isinstance(cfg, dict):
            cfg = {}
        # Determine which model to use based on provider
        provider = self.provider_var.get()
        model = self.gemini_model_var.get() if provider == "gemini" else self.model_var.get()
        
        cfg.update(
            {
                "ENABLE_FREEPIK": self.var_freepik.get(),
                "OPEN_IMAGE": self.var_open.get() and self.var_freepik.get(),
                "PRINT_IMAGE": self.var_print.get() and self.var_freepik.get(),
                "PRINT_TO_PDF": self.var_pdf.get() and self.var_freepik.get(),
                "DEBUG_TIMING": self.var_debug.get(),
                "PRINT_PAGE_SIZE": self.page_size_var.get(),
                "IMAGE_PROVIDER": provider,
                "FREEPIK_MODEL": self.model_var.get(),
                "GEMINI_MODEL": self.gemini_model_var.get(),
                "RUNNING": self.running,
            }
        )
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
        self.chk_pdf.config(state=state)
        if not self.var_freepik.get():
            self.var_open.set(False)
            self.var_print.set(False)
            self.var_pdf.set(False)

    def on_provider_change(self):
        """Switch between Freepik and Gemini model selectors."""
        provider = self.provider_var.get()
        if provider == "gemini":
            self.freepik_model_combo.pack_forget()
            self.gemini_model_combo.pack(side="left", padx=4)
        else:
            self.gemini_model_combo.pack_forget()
            self.freepik_model_combo.pack(side="left", padx=4)
        self.apply_config()

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
        env["PRINT_TO_PDF"] = "1" if self.var_pdf.get() else "0"
        env["DEBUG_TIMING"] = "1" if self.var_debug.get() else "0"
        env["PRINT_PAGE_SIZE"] = self.page_size_var.get()
        env["IMAGE_PROVIDER"] = self.provider_var.get()
        env["FREEPIK_MODEL"] = self.model_var.get()
        env["GEMINI_MODEL"] = self.gemini_model_var.get()
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
        """Stop the subprocess with guaranteed cleanup."""
        if not self.running and not self.proc:
            return
        
        self.running = False
        
        if self.proc:
            cleanup_process(self.proc)
            self.proc = None
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
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

    def _ensure_cleanup(self):
        """Guaranteed cleanup called on exit."""
        if not self._cleanup_registered:
            self._cleanup_registered = True
            self.stop_proc()
    
    def on_close(self):
        """Handle window close event."""
        self._ensure_cleanup()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = DebugUI(root)
    root.mainloop()
