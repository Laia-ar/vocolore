"""
User-facing Tkinter UI for WiFi transcription.
- Big Start/Stop buttons.
- Status indicators for ESP button press, recording/transcribing, and Freepik image generation.
- Shows latest transcription text and last image filename.

It runs wifi_transcribe.py as a subprocess and parses its stdout to drive the UI.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:
    Image = None
    ImageTk = None

DEBUG = os.getenv("DEBUG", "0") == "1"


def dlog(msg: str):
    if DEBUG:
        print(f"[USER_UI] {msg}", flush=True)

class StatusLabel(tk.Label):
    def set_state(self, text, color):
        self.config(text=text, bg=color)


class UserUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("WiFi Transcribe")
        root.geometry("700x600")
        self.freepik_enabled = os.getenv("ENABLE_FREEPIK", "0") == "1"
        self.runtime_config = os.getenv("RUNTIME_CONFIG_FILE", ".runtime_config.json")
        self.last_cfg_mtime = None
        self.open_enabled = self.freepik_enabled and (os.getenv("OPEN_IMAGE", "0") == "1")
        self.print_enabled = self.freepik_enabled and (os.getenv("PRINT_IMAGE", "0") == "1")
        self.launch_transcribe = os.getenv("LAUNCH_TRANSCRIBE", "1") == "1"
        self.running_flag = False
        self.last_image_path = None

        self.proc = None
        self.reader_thread = None
        self.msg_queue: "queue.Queue[str]" = queue.Queue()
        self.running = False

        # Top controls
        # Status indicators
        status_frame = tk.Frame(root)
        status_frame.pack(fill="x", padx=12, pady=8)
        self.lbl_ready = StatusLabel(status_frame, text="Ready: no", width=14, bg="#f0ad4e", font=("Helvetica", 12))
        self.lbl_ready.pack(side="left", padx=6)
        self.lbl_btn = StatusLabel(status_frame, text="Button: idle", width=20, bg="#cccccc", font=("Helvetica", 12))
        self.lbl_btn.pack(side="left", padx=6)
        self.lbl_transcribe = StatusLabel(status_frame, text="Transcribe: idle", width=20, bg="#cccccc", font=("Helvetica", 12))
        self.lbl_transcribe.pack(side="left", padx=6)
        self.lbl_freepik = StatusLabel(status_frame, text="Freepik: off", width=20, bg="#cccccc", font=("Helvetica", 12))
        self.lbl_freepik.pack(side="left", padx=6)
        self.lbl_open = StatusLabel(status_frame, text="Open img: off", width=14, bg="#cccccc", font=("Helvetica", 12))
        self.lbl_open.pack(side="left", padx=6)
        self.lbl_print = StatusLabel(status_frame, text="Print img: off", width=14, bg="#cccccc", font=("Helvetica", 12))
        self.lbl_print.pack(side="left", padx=6)

        # Latest transcription
        tk.Label(root, text="Latest transcription:", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.transcription_var = tk.StringVar(value="")
        self.transcription_lbl = tk.Label(root, textvariable=self.transcription_var, wraplength=660, justify="left", font=("Helvetica", 12))
        self.transcription_lbl.pack(fill="x", padx=12)

        # Last image
        tk.Label(root, text="Last image saved:", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.image_var = tk.StringVar(value="(none)")
        self.image_lbl = tk.Label(root, textvariable=self.image_var, wraplength=660, justify="left", font=("Helvetica", 12))
        self.image_lbl.pack(fill="x", padx=12)
        # Image preview
        self.image_preview = tk.Label(root)
        self.image_preview.pack(padx=12, pady=(4, 8))
        self._photo_ref = None

        # Schedule message drain
        self.root.after(100, self.drain_messages)
        # Schedule config watcher
        self.root.after(500, self.check_config)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # auto-start if enabled
        if self.launch_transcribe:
            self.start_proc()

    def append_log(self, text: str):
        # Forward logs to debug UI via runtime config file (if needed)
        # No-op here to keep user UI clean
        return

    def start_proc(self):
        if self.running:
            return
        cmd = [sys.executable, "wifi_transcribe.py"]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
        except Exception as exc:
            self.append_log(f"Failed to start: {exc}\n")
            self.proc = None
            return
        self.running = True
        self.lbl_transcribe.set_state("Transcribe: idle", "#cccccc")
        # Freepik status reflects current env default
        if self.freepik_enabled:
            self.lbl_freepik.set_state("Freepik: on", "#a8e6cf")
            self.lbl_open.set_state(f"Open img: {'on' if self.open_enabled else 'off'}", "#a8e6cf" if self.open_enabled else "#cccccc")
            self.lbl_print.set_state(f"Print img: {'on' if self.print_enabled else 'off'}", "#a8e6cf" if self.print_enabled else "#cccccc")
        else:
            self.lbl_freepik.set_state("Freepik: off", "#cccccc")
            self.lbl_open.set_state("Open img: off", "#cccccc")
            self.lbl_print.set_state("Print img: off", "#cccccc")
        # Ready remains no until connection message arrives
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
        self.lbl_btn.set_state("Button: idle", "#cccccc")
        self.lbl_transcribe.set_state("Transcribe: idle", "#cccccc")
        self.lbl_freepik.set_state("Freepik: off", "#cccccc")
        self.lbl_ready.set_state("Ready: no", "#f0ad4e")

    def _reader(self):
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            # echo to console so button UP/DOWN are visible
            print(line, end="", flush=True)
            self.msg_queue.put(line)
        rc = self.proc.wait()
        self.msg_queue.put(f"\n[process exited with code {rc}]\n")
        self.running = False

    def _handle_line(self, line: str):
        # Log
        self.append_log(line)
        lower = line.lower()
        if "connected to wifi audio source" in lower:
            self.lbl_ready.set_state("Ready: yes", "#5cb85c")
        # Button state
        if "[ctrl]" in lower and "down" in lower:
            self.lbl_btn.set_state("Button: DOWN", "#ffcc66")
        elif "[ctrl]" in lower and "up" in lower:
            self.lbl_btn.set_state("Button: UP", "#cccccc")
        # Transcription status
        if "transcribing" in lower:
            self.lbl_transcribe.set_state("Transcribing...", "#87cefa")
        if "transcription:" in lower:
            self.lbl_transcribe.set_state("Transcribe: done", "#a8e6cf")
            # capture text after colon
            parts = line.split("Transcription:", 1)
            if len(parts) == 2:
                self.transcription_var.set(parts[1].strip())
        # Freepik status
        if "requesting freepik" in lower or "freepik status" in lower:
            self.lbl_freepik.set_state("Freepik: loading", "#ffd54f")
            dlog("Freepik loading...")
        if "image saved to" in lower:
            self.lbl_freepik.set_state("Freepik: on", "#a8e6cf")
            self.lbl_open.set_state(f"Open img: {'on' if self.open_enabled else 'off'}", "#a8e6cf" if self.open_enabled else "#cccccc")
            self.lbl_print.set_state(f"Print img: {'on' if self.print_enabled else 'off'}", "#a8e6cf" if self.print_enabled else "#cccccc")
            # capture filename
            parts = line.split("Image saved to", 1)
            if len(parts) == 2:
                path = parts[1].strip()
                self.image_var.set(path)
                self.last_image_path = path
                dlog(f"Image saved path received: {path}")
                self._show_image(path)
        if "freepik generation failed" in lower or "freepik request error" in lower:
            self.lbl_freepik.set_state("Freepik: error", "#f8bcbc")
            self.lbl_open.set_state("Open img: off", "#cccccc")
            self.lbl_print.set_state("Print img: off", "#cccccc")
            dlog("Freepik error encountered.")

    def check_config(self):
        try:
            if os.path.isfile(self.runtime_config):
                mtime = os.path.getmtime(self.runtime_config)
                if self.last_cfg_mtime is None or mtime != self.last_cfg_mtime:
                    self.last_cfg_mtime = mtime
                    with open(self.runtime_config, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    freepik = bool(data.get("ENABLE_FREEPIK", self.freepik_enabled))
                    open_img = bool(data.get("OPEN_IMAGE", self.open_enabled))
                    print_img = bool(data.get("PRINT_IMAGE", self.print_enabled))
                    running = bool(data.get("RUNNING", self.running_flag))
                    last_image = data.get("LAST_IMAGE", self.last_image_path)
                    changed = (freepik != self.freepik_enabled) or (open_img != self.open_enabled) or (print_img != self.print_enabled)
                    self.freepik_enabled = freepik
                    self.open_enabled = open_img and freepik
                    self.print_enabled = print_img and freepik
                    self.running_flag = running
                    if changed:
                        if self.freepik_enabled:
                            self.lbl_freepik.set_state("Freepik: on", "#a8e6cf")
                            self.lbl_open.set_state(f"Open img: {'on' if self.open_enabled else 'off'}", "#a8e6cf" if self.open_enabled else "#cccccc")
                            self.lbl_print.set_state(f"Print img: {'on' if self.print_enabled else 'off'}", "#a8e6cf" if self.print_enabled else "#cccccc")
                        else:
                            self.lbl_freepik.set_state("Freepik: off", "#cccccc")
                            self.lbl_open.set_state("Open img: off", "#cccccc")
                            self.lbl_print.set_state("Print img: off", "#cccccc")
                    # Ready flag from debug start/stop
                    if self.running_flag:
                        self.lbl_ready.set_state("Ready: waiting", "#f0ad4e")
                    else:
                        self.lbl_ready.set_state("Ready: no", "#f0ad4e")
                    # Update image preview if path changed
                    if last_image and last_image != self.last_image_path:
                        self.last_image_path = last_image
                        self.image_var.set(last_image)
                        self._show_image(last_image)
        except Exception:
            pass
        self.root.after(500, self.check_config)

    def _show_image(self, path: str):
        p = Path(path)
        if not p.exists():
            dlog(f"Image file not found: {path}")
            return
        max_w, max_h = 400, 400
        img_obj = None
        if Image is not None and ImageTk is not None:
            try:
                img_obj = Image.open(p)
                img_obj.thumbnail((max_w, max_h))
                photo = ImageTk.PhotoImage(img_obj)
                dlog(f"Loaded image via Pillow: {path} size={img_obj.size}")
            except Exception as exc:
                photo = None
                dlog(f"Pillow load failed: {exc}")
        else:
            try:
                photo = tk.PhotoImage(file=str(p))
                w, h = photo.width(), photo.height()
                factor = max(1, w // max_w, h // max_h)
                if factor > 1:
                    photo = photo.subsample(factor, factor)
                dlog(f"Loaded image via Tk PhotoImage: {path} size=({w},{h})")
            except Exception as exc:
                photo = None
                dlog(f"Tk PhotoImage load failed: {exc}")
        if photo:
            self._photo_ref = photo
            self.image_preview.config(image=photo)
        else:
            dlog("No photo produced; preview not updated.")

    def drain_messages(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_line(msg)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_messages)

    def on_close(self):
        self.stop_proc()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = UserUI(root)
    root.mainloop()
