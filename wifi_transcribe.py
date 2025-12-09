import json
import os
import queue
import socket
import subprocess
import threading
import time
import wave
import gc
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from rich.console import Console

try:
    from PIL import Image  # type: ignore
except ImportError:
    Image = None

# Load environment early so config below picks up .env/config.env values
load_dotenv()
load_dotenv("config.env")

# Configuration (override via environment variables if needed)
HOST = os.getenv("WIFI_HOST", "192.168.4.1")
PORT = int(os.getenv("WIFI_PORT", "5005"))
SAMPLE_RATE = int(os.getenv("WIFI_SAMPLE_RATE", "16000"))  # actual WiFi source rate
TARGET_RATE = 16000  # Whisper target sample rate
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "es")
MIN_AUDIO_SEC = float(os.getenv("MIN_AUDIO_SEC", "4.0"))       # ignore very short clips (skip noise)
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "16.0"))    # force flush if buffer grows too long
SAVE_WIFI_WAV = os.getenv("SAVE_WIFI_WAV", "0") == "1"         # set to 1 to persist raw audio
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
ENABLE_FREEPIK = os.getenv("ENABLE_FREEPIK", "0") == "1"
FREEPIK_WEBHOOK_URL = os.getenv("FREEPIK_WEBHOOK_URL", "https://www.example.com/webhook")
PRINT_IMAGE = os.getenv("PRINT_IMAGE", "0") == "1"
PRINT_COMMAND = os.getenv("PRINT_COMMAND", "/usr/bin/lp")
PRINT_TO_PDF = os.getenv("PRINT_TO_PDF", "0") == "1"
PRINT_PDF_DIR = os.getenv("PRINT_PDF_DIR", "printouts")
SAVE_CLIP_WAV = os.getenv("SAVE_CLIP_WAV", "1") == "1"
CLIP_WAV_DIR = os.getenv("CLIP_WAV_DIR", "clips")
DEBUG_TIMING = os.getenv("DEBUG_TIMING", "0") == "1"
OPEN_IMAGE = os.getenv("OPEN_IMAGE", "0") == "1"
RUNTIME_CONFIG_FILE = os.getenv("RUNTIME_CONFIG_FILE", ".runtime_config.json")
PRE_ROLL_SEC = float(os.getenv("PRE_ROLL_SEC", "0.3"))  # prepend this much from previous clip

console = Console()
stop_event = threading.Event()
buffer_lock = threading.Lock()
audio_buffer = bytearray()
last_audio_time = None
transcribe_queue: "queue.Queue[bytes]" = queue.Queue()
buffer_packet_count = 0
clip_counter = 0
runtime_lock = threading.Lock()
runtime_flags = {
    "ENABLE_FREEPIK": ENABLE_FREEPIK,
    "PRINT_IMAGE": PRINT_IMAGE,
    "OPEN_IMAGE": OPEN_IMAGE,
    "PRINT_TO_PDF": PRINT_TO_PDF,
    "MIN_AUDIO_SEC": MIN_AUDIO_SEC,
    "MAX_BUFFER_SEC": MAX_BUFFER_SEC,
    "DEBUG_TIMING": DEBUG_TIMING,
    "LAST_IMAGE": None,
}


def read_exact(sock: socket.socket, n: int):
    """Read exactly n bytes from socket; return None on EOF or shutdown."""
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            if stop_event.is_set():
                return None
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def open_wav_sink():
    """Optionally open a WAV file to mirror the incoming stream for debugging."""
    if not SAVE_WIFI_WAV:
        return None
    filename = f"wifi_capture_{int(time.time())}.wav"
    wav = wave.open(filename, "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    console.print(f"[blue]Saving raw WiFi audio to {filename}[/blue]")
    return wav


def queue_audio(payload: bytes, wav_writer):
    """Append incoming audio payload to the shared buffer and WAV sink."""
    global last_audio_time
    global buffer_packet_count
    if wav_writer:
        wav_writer.writeframes(payload)
    with buffer_lock:
        audio_buffer.extend(payload)
        buffer_packet_count += 1
        last_audio_time = time.time()
        secs = len(audio_buffer) / (SAMPLE_RATE * 2.0)
        if secs >= current_max_buffer_sec():
            raw = bytes(audio_buffer)
            pkt = buffer_packet_count
            audio_buffer.clear()
            buffer_packet_count = 0
            console.print(f"[yellow]Auto-flush after reaching {secs:.1f}s buffer[/yellow]")
            if debug_timing_enabled():
                console.print(f"[grey]Flush reason=max_buffer packets={pkt} bytes={len(raw)}[/grey]")
            transcribe_queue.put(raw)


def flush_buffer(reason: str = "manual", forced: bool = False):
    """Move buffered audio into the transcription queue if ready."""
    global buffer_packet_count, clip_counter
    raw = None
    with buffer_lock:
        if not audio_buffer:
            return
        raw = bytes(audio_buffer)
        audio_buffer.clear()
        pkt = buffer_packet_count
        buffer_packet_count = 0
    secs = len(raw) / (SAMPLE_RATE * 2.0)
    clip_counter += 1
    if debug_timing_enabled():
        console.print(f"[grey]Flush #{clip_counter} reason={reason} packets={pkt} bytes={len(raw)} secs={secs:.2f}s[/grey]")
    console.print(f"[cyan]Queued {secs:.2f}s clip for transcription ({reason}).[/cyan]")
    transcribe_queue.put(raw)


def flush_loop():
    """Periodically flush buffered audio after a silence gap."""
    while not stop_event.is_set():
        time.sleep(0.2)


def config_watcher():
    """Watch a JSON config file to allow runtime toggle updates."""
    last_mtime = None
    while not stop_event.is_set():
        try:
            if os.path.isfile(RUNTIME_CONFIG_FILE):
                mtime = os.path.getmtime(RUNTIME_CONFIG_FILE)
                if last_mtime is None or mtime != last_mtime:
                    last_mtime = mtime
                    with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    with runtime_lock:
                        for key in ("ENABLE_FREEPIK", "PRINT_IMAGE", "OPEN_IMAGE"):
                            if key in data:
                                runtime_flags[key] = bool(data[key])
                        if "PRINT_TO_PDF" in data:
                            runtime_flags["PRINT_TO_PDF"] = bool(data["PRINT_TO_PDF"])
                        for key in ("MIN_AUDIO_SEC", "MAX_BUFFER_SEC"):
                            if key in data:
                                try:
                                    runtime_flags[key] = float(data[key])
                                except (TypeError, ValueError):
                                    pass
                        if "DEBUG_TIMING" in data:
                            runtime_flags["DEBUG_TIMING"] = bool(data["DEBUG_TIMING"])
                        if "LAST_IMAGE" in data:
                            runtime_flags["LAST_IMAGE"] = data.get("LAST_IMAGE")
                    console.print(f"[grey]Runtime config reloaded from {RUNTIME_CONFIG_FILE}[/grey]")
        except Exception as exc:
            console.print(f"[red]Runtime config watcher error:[/red] {exc}")
        time.sleep(1.0)


def freepik_enabled() -> bool:
    with runtime_lock:
        return runtime_flags.get("ENABLE_FREEPIK", False)


def print_enabled() -> bool:
    with runtime_lock:
        return runtime_flags.get("PRINT_IMAGE", False)


def open_enabled() -> bool:
    with runtime_lock:
        return runtime_flags.get("OPEN_IMAGE", False)


def pdf_enabled() -> bool:
    with runtime_lock:
        return runtime_flags.get("PRINT_TO_PDF", PRINT_TO_PDF)


def current_min_audio_sec() -> float:
    with runtime_lock:
        return float(runtime_flags.get("MIN_AUDIO_SEC", MIN_AUDIO_SEC))


def current_max_buffer_sec() -> float:
    with runtime_lock:
        return float(runtime_flags.get("MAX_BUFFER_SEC", MAX_BUFFER_SEC))


def debug_timing_enabled() -> bool:
    with runtime_lock:
        return bool(runtime_flags.get("DEBUG_TIMING", DEBUG_TIMING))


def persist_runtime_flags():
    try:
        with runtime_lock:
            data = dict(runtime_flags)
        with open(RUNTIME_CONFIG_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception as exc:
        console.print(f"[red]Failed to persist runtime flags:[/red] {exc}")


def sanitize_audio(raw: bytes) -> bytes:
    """Ensure 16-bit alignment; drop trailing odd byte if present."""
    if not raw:
        return raw
    if len(raw) % 2 != 0:
        console.print("[yellow]Trimming 1 trailing byte to align 16-bit samples.[/yellow]")
        raw = raw[:-1]
    return raw


def transcribe_worker(model: WhisperModel):
    """Consume raw audio from the queue and run Whisper transcription."""
    while not stop_event.is_set():
        try:
            raw = transcribe_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if raw is None:
            break

        raw = sanitize_audio(raw)
        if not raw:
            continue

        secs = len(raw) / (SAMPLE_RATE * 2.0)
        min_sec = current_min_audio_sec()
        if secs < min_sec:
            console.print(f"[grey58]Skipped {secs:.2f}s clip (below MIN_AUDIO_SEC {min_sec}).[/grey58]")
            if debug_timing_enabled():
                console.print(f"[grey]Clip skipped (too short). Bytes={len(raw)}[/grey]")
            continue

        if SAVE_CLIP_WAV:
            try:
                os.makedirs(CLIP_WAV_DIR, exist_ok=True)
                ts_ms = int(time.time() * 1000)
                clip_path = os.path.join(CLIP_WAV_DIR, f"wifi_clip_{ts_ms}.wav")
                with wave.open(clip_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(raw)
                console.print(f"[blue]Saved clip to {clip_path} ({secs:.2f}s)[/blue]")
            except Exception as exc:
                console.print(f"[red]Failed to save clip WAV:[/red] {exc}")

        audio_np = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
        if peak < 0.001:
            console.print("[yellow]Warning: clip near silence (peak < 0.001).[/yellow]")

        # Resample to Whisper target rate if needed
        if audio_np.size and SAMPLE_RATE != TARGET_RATE:
            src_len = audio_np.shape[0]
            dst_len = int(round(src_len * TARGET_RATE / SAMPLE_RATE))
            if dst_len > 1:
                x = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
                audio_np = np.interp(x, np.arange(src_len, dtype=np.float32), audio_np).astype(np.float32)
            console.print(f"[blue]Resampled {src_len} -> {audio_np.shape[0]} samples ({SAMPLE_RATE} -> {TARGET_RATE} Hz)[/blue]")

        console.print(f"[magenta]Transcribing {secs:.2f}s of WiFi audio...[/magenta]")
        try:
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                language=LANGUAGE,
            )
            transcription = "".join(segment.text for segment in segments).strip()
            if transcription:
                console.print(f"[bold green]Transcription:[/bold green] {transcription}")
                if freepik_enabled():
                    threading.Thread(
                        target=send_image_generation_request,
                        args=(transcription,),
                        daemon=True,
                    ).start()
            else:
                console.print("[grey58]No speech detected.[/grey58]")
        except Exception as e:
            console.print(f"[red]Transcription error:[/red] {e}")


def wifi_listener():
    """Connect to the ESP32 stream and feed audio packets into the buffer."""
    console.print(f"[blue]Connecting to {HOST}:{PORT}...[/blue]")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect((HOST, PORT))
    console.print("[green]Connected to WiFi audio source.[/green]")

    wav_writer = open_wav_sink()

    total_audio_bytes = 0
    packet_count = 0

    try:
        while not stop_event.is_set():
            header = read_exact(sock, 3)
            if header is None:
                console.print("[red]Connection closed by device.[/red]")
                break

            pkt_type = header[0]
            length = (header[1] << 8) | header[2]
            if length == 0:
                continue

            payload = read_exact(sock, length)
            if payload is None:
                console.print("[red]Connection closed mid-packet.[/red]")
                break

            if pkt_type == ord("C"):
                text = payload.decode("utf-8", errors="ignore")
                console.print(f"[cyan][CTRL][/cyan] {text}")
                upper_text = text.upper()
                if any(key in upper_text for key in ("STOP", "END", "UP", "RELEASE")):
                    flush_buffer(reason="button event", forced=True)
            elif pkt_type == ord("A"):
                packet_count += 1
                total_audio_bytes += len(payload)
                queue_audio(payload, wav_writer)
                if packet_count % 20 == 0:
                    secs = total_audio_bytes / (SAMPLE_RATE * 2.0)
                    console.print(f"[blue]Audio packets: {packet_count}, ~{secs:.2f}s captured[/blue]")
            else:
                # Unknown packet type; ignore
                continue
    finally:
        flush_buffer(reason="disconnect", forced=True)
        stop_event.set()
        if wav_writer:
            wav_writer.close()
        sock.close()
        console.print("[yellow]WiFi listener stopped.[/yellow]")


def send_image_generation_request(prompt: str):
    """Call Freepik image generation API based on the transcript."""
    api_key = os.getenv("FREEPIK_API_KEY")
    if not api_key:
        console.print("[red]FREEPIK_API_KEY not set; skipping image generation.[/red]")
        return

    url = "https://api.freepik.com/v1/ai/gemini-2-5-flash-image-preview"
    payload = {
        "prompt": f"coloring book style image of {prompt}",
        "reference_images": [],
        "webhook_url": FREEPIK_WEBHOOK_URL,
    }
    headers = {
        "Content-Type": "application/json",
        "x-freepik-api-key": api_key,
    }

    try:
        console.print(f"[blue]Requesting Freepik image for: {prompt}[/blue]")
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("data", {}).get("task_id")
        if not task_id:
            console.print("[red]No task_id returned from Freepik; aborting image generation.[/red]")
            return

        console.print(f"[green]Freepik task id: {task_id}. Polling for completion...[/green]")
        poll_deadline = time.time() + 60
        image_url = None
        while time.time() < poll_deadline:
            poll_resp = requests.get(f"{url}/{task_id}", headers=headers, timeout=15)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("data", {}).get("status")
            console.print(f"[cyan]Freepik status: {status}[/cyan]")
            if status in ("COMPLETED", "READY"):
                generated = poll_data.get("data", {}).get("generated") or []
                if generated:
                    image_url = generated[0]
                elif "url" in poll_data:
                    image_url = poll_data["url"]
                elif "image_url" in poll_data:
                    image_url = poll_data["image_url"]
                break
            if status == "FAILED":
                console.print("[red]Freepik generation failed.[/red]")
                return
            time.sleep(5)

        if not image_url:
            console.print("[orange3]Timed out waiting for Freepik image URL.[/orange3]")
            return

        img_resp = requests.get(image_url, stream=True, timeout=30)
        img_resp.raise_for_status()
        filename = f"generated_image_{task_id}.png"
        with open(filename, "wb") as fh:
            for chunk in img_resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        console.print(f"[bold green]Image saved to {filename}[/bold green]")

        pdf_path = save_pdf_copy(filename)
        try:
            with runtime_lock:
                runtime_flags["LAST_IMAGE"] = filename
            persist_runtime_flags()
        except Exception as exc:
            console.print(f"[red]Failed to record image path:[/red] {exc}")

        if print_enabled():
            target_path = pdf_path
            if target_path is None:
                target_path = make_a4_image_copy(filename, task_id)
            try:
                subprocess.run([PRINT_COMMAND, target_path or filename], check=True)
                console.print(f"[green]Sent image to printer via {PRINT_COMMAND}: {target_path or filename}[/green]")
            except subprocess.CalledProcessError as exc:
                console.print(f"[red]Printing failed ({PRINT_COMMAND}):[/red] {exc}")
            except Exception as exc:
                console.print(f"[red]Unexpected printing error:[/red] {exc}")
        elif pdf_path:
            console.print(f"[green]PDF copy ready at {pdf_path} (printing disabled).[/green]")

    except requests.RequestException as exc:
        console.print(f"[red]Freepik request error:[/red] {exc}")
    except Exception as exc:
        console.print(f"[red]Unexpected Freepik error:[/red] {exc}")


def save_pdf_copy(image_path: str):
    """Optionally write a PDF copy of the generated image."""
    if not pdf_enabled():
        return None
    if Image is None:
        console.print("[yellow]PRINT_TO_PDF is enabled but Pillow is not installed; skipping PDF export.[/yellow]")
        return None
    try:
        os.makedirs(PRINT_PDF_DIR, exist_ok=True)
        pdf_path = Path(PRINT_PDF_DIR) / (Path(image_path).stem + ".pdf")
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Create an A4 canvas at 300 DPI and center-fit the image
            a4_w, a4_h = 2480, 3508  # pixels for 210x297 mm at 300 DPI
            margin = 120  # ~10 mm margin
            canvas = Image.new("RGB", (a4_w, a4_h), "white")
            max_w, max_h = a4_w - 2 * margin, a4_h - 2 * margin
            # Prefer filling the width (with aspect ratio) unless it exceeds page height
            target_w = max_w
            target_h = int(img.height * (target_w / img.width))
            if target_h > max_h:
                target_h = max_h
                target_w = int(img.width * (target_h / img.height))
            resized = img.resize((target_w, target_h))
            x = (a4_w - target_w) // 2
            y = (a4_h - target_h) // 2
            canvas.paste(resized, (x, y))
            canvas.save(pdf_path, "PDF", resolution=300.0)
        console.print(f"[green]Saved PDF copy to {pdf_path}[/green]")
        return str(pdf_path)
    except Exception as exc:
        console.print(f"[red]PDF export failed:[/red] {exc}")
        return None


def make_a4_image_copy(image_path: str, task_id: str = ""):
    """Build a print-friendly A4-sized PNG with the image scaled to fill width."""
    if Image is None:
        return None
    try:
        os.makedirs(PRINT_PDF_DIR, exist_ok=True)
        out_name = f"print_image_{task_id or Path(image_path).stem}.png"
        out_path = Path(PRINT_PDF_DIR) / out_name
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            a4_w, a4_h = 2480, 3508
            margin = 120
            max_w, max_h = a4_w - 2 * margin, a4_h - 2 * margin
            target_w = max_w
            target_h = int(img.height * (target_w / img.width))
            if target_h > max_h:
                target_h = max_h
                target_w = int(img.width * (target_h / img.height))
            resized = img.resize((target_w, target_h))
            canvas = Image.new("RGB", (a4_w, a4_h), "white")
            x = (a4_w - target_w) // 2
            y = (a4_h - target_h) // 2
            canvas.paste(resized, (x, y))
            canvas.save(out_path, "PNG")
        console.print(f"[green]Prepared A4 print image at {out_path}[/green]")
        return str(out_path)
    except Exception as exc:
        console.print(f"[red]A4 image prep failed:[/red] {exc}")
        return None


def main():
    console.print(f"[bold blue]Loading Whisper model: {MODEL_SIZE} ({DEVICE}, {COMPUTE_TYPE})[/bold blue]")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        console.print(f"[red]Failed to load Whisper model: {e}[/red]")
        raise SystemExit(1)

    transcriber = threading.Thread(target=transcribe_worker, args=(model,), daemon=True)
    flusher = threading.Thread(target=flush_loop, daemon=True)
    cfg_thread = threading.Thread(target=config_watcher, daemon=True)
    transcriber.start()
    flusher.start()
    cfg_thread.start()

    try:
        wifi_listener()
    except KeyboardInterrupt:
        console.print("\n[red]Stopping...[/red]")
        stop_event.set()
    finally:
        transcribe_queue.put(None)
        transcriber.join()
        flusher.join()
        cfg_thread.join()
        # Explicitly drop model and free memory
        try:
            model = None
            gc.collect()
            console.print("[grey]Model released and GC run.[/grey]")
        except Exception:
            pass
        console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
