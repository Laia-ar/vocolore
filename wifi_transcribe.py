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
SAMPLE_RATE = int(os.getenv("WIFI_SAMPLE_RATE", "8000"))  # actual WiFi source rate
TARGET_RATE = 16000  # Whisper target sample rate
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3")
LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "es")
MIN_AUDIO_SEC = float(os.getenv("MIN_AUDIO_SEC", "4.0"))       # ignore very short clips (skip noise)
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "16.0"))    # force flush if buffer grows too long
SAVE_WIFI_WAV = os.getenv("SAVE_WIFI_WAV", "0") == "1"         # set to 1 to persist raw audio
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
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

def get_config(key: str, default=None, env_var: str = None):
    """Get config value from runtime_flags (UI) first, then env var, then default."""
    with runtime_lock:
        if key in runtime_flags and runtime_flags[key] is not None:
            return runtime_flags[key]
    env_key = env_var or key
    return os.getenv(env_key, default)


# Image generation provider selection
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "freepik")  # "freepik" or "gemini"
FREEPIK_MODEL = os.getenv("FREEPIK_MODEL", "gemini-2-5-flash-image-preview")  # Freepik model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")  # Gemini model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # For Google Gemini API
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY", "")  # For Freepik API
PRINT_PAGE_SIZE = os.getenv("PRINT_PAGE_SIZE", "A4")  # A4 or A5

console = Console()
stop_event = threading.Event()
buffer_lock = threading.Lock()
audio_buffer = bytearray()
last_audio_time = None
transcribe_queue: "queue.Queue[bytes]" = queue.Queue()
buffer_packet_count = 0
clip_counter = 0
runtime_lock = threading.Lock()
wifi_socket: socket.socket = None  # Global socket reference for sound commands
runtime_flags = {
    "ENABLE_FREEPIK": ENABLE_FREEPIK,
    "PRINT_IMAGE": PRINT_IMAGE,
    "OPEN_IMAGE": OPEN_IMAGE,
    "PRINT_TO_PDF": PRINT_TO_PDF,
    "MIN_AUDIO_SEC": MIN_AUDIO_SEC,
    "MAX_BUFFER_SEC": MAX_BUFFER_SEC,
    "DEBUG_TIMING": DEBUG_TIMING,
    "LAST_IMAGE": None,
    "RUNNING": False,
    "READY": False,
    "BUTTON_STATE": "idle",
    "BATTERY_LEVEL": None,
    # Image generation settings (can be overridden by UI)
    "IMAGE_PROVIDER": IMAGE_PROVIDER,
    "FREEPIK_MODEL": FREEPIK_MODEL,
    "GEMINI_MODEL": GEMINI_MODEL,
    "PRINT_PAGE_SIZE": PRINT_PAGE_SIZE,
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
                        if "RUNNING" in data:
                            runtime_flags["RUNNING"] = bool(data["RUNNING"])
                        if "READY" in data:
                            runtime_flags["READY"] = bool(data["READY"])
                        if "BUTTON_STATE" in data and isinstance(data["BUTTON_STATE"], str):
                            runtime_flags["BUTTON_STATE"] = data["BUTTON_STATE"]
                        if "BATTERY_LEVEL" in data:
                            runtime_flags["BATTERY_LEVEL"] = data.get("BATTERY_LEVEL")
                        # Image generation settings from UI
                        for key in ("IMAGE_PROVIDER", "FREEPIK_MODEL", "GEMINI_MODEL", "PRINT_PAGE_SIZE"):
                            if key in data:
                                runtime_flags[key] = data[key]
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


def update_runtime_state(**kwargs):
    """
    Merge runtime status updates and persist if anything changed.
    Keeps user-facing UI in sync even when it is not attached to stdout.
    """
    changed = False
    with runtime_lock:
        for key, value in kwargs.items():
            if runtime_flags.get(key) != value:
                runtime_flags[key] = value
                changed = True
    if changed:
        persist_runtime_flags()
    return changed


def battery_level() -> int | None:
    with runtime_lock:
        return runtime_flags.get("BATTERY_LEVEL")


def sanitize_audio(raw: bytes) -> bytes:
    """Ensure 16-bit alignment; drop trailing odd byte if present."""
    if not raw:
        return raw
    if len(raw) % 2 != 0:
        console.print("[yellow]Trimming 1 trailing byte to align 16-bit samples.[/yellow]")
        raw = raw[:-1]
    return raw


def send_sound_command(sock: socket.socket, sound_id: int):
    """Send a sound command to the ESP32."""
    try:
        # Framed protocol: [type='S'][len=1][payload=sound_id]
        header = bytes([ord('S'), 0, 1])
        payload = bytes([sound_id])
        sock.sendall(header + payload)
    except Exception as exc:
        console.print(f"[red]Failed to send sound command:[/red] {exc}")


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
            # Sound 0 = explosion (audio too short/error)
            if wifi_socket:
                send_sound_command(wifi_socket, 0)
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
                # Sound 3 = powerUp (transcription complete)
                if wifi_socket:
                    send_sound_command(wifi_socket, 3)
                if freepik_enabled():
                    threading.Thread(
                        target=send_image_generation_request,
                        args=(transcription,),
                        daemon=True,
                    ).start()
            else:
                console.print("[grey58]No speech detected.[/grey58]")
                # Sound 0 = explosion (no speech detected - error)
                if wifi_socket:
                    send_sound_command(wifi_socket, 0)
        except Exception as e:
            console.print(f"[red]Transcription error:[/red] {e}")


def wifi_listener():
    """Connect to the ESP32 stream and feed audio packets into the buffer."""
    global wifi_socket
    console.print(f"[blue]Connecting to {HOST}:{PORT}...[/blue]")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    sock.connect((HOST, PORT))
    wifi_socket = sock  # Store for sound commands
    console.print("[green]Connected to WiFi audio source.[/green]")
    update_runtime_state(READY=True, BUTTON_STATE="idle")

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
                lower_text = text.lower()
                if "down" in lower_text or "press" in lower_text:
                    update_runtime_state(BUTTON_STATE="down")
                elif any(key in lower_text for key in ("up", "release", "stop", "end")):
                    update_runtime_state(BUTTON_STATE="up")
            elif pkt_type == ord("B"):
                text = payload.decode("utf-8", errors="ignore")
                try:
                    level = int(text)
                    update_runtime_state(BATTERY_LEVEL=level)
                    # Color code battery level
                    if level <= 20:
                        console.print(f"[red]Battery: {level}%[/red] ⚠️ Low battery!")
                    elif level <= 50:
                        console.print(f"[yellow]Battery: {level}%[/yellow]")
                    else:
                        console.print(f"[green]Battery: {level}%[/green]")
                except ValueError:
                    pass
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
        wifi_socket = None
        update_runtime_state(READY=False, BUTTON_STATE="idle")
        console.print("[yellow]WiFi listener stopped.[/yellow]")


# Freepik Model configurations
FREEPIK_MODELS = {
    "gemini-2-5-flash-image-preview": {
        "endpoint": "/v1/ai/gemini-2-5-flash-image-preview",
        "supports_aspect_ratio": False,
        "description": "Gemini 2.5 Flash (undocumented but works)",
    },
    "mystic": {
        "endpoint": "/v1/ai/mystic",
        "supports_aspect_ratio": False,
        "supports_resolution": True,
        "resolutions": ["1k", "2k", "4k"],
        "description": "Freepik Mystic - Ultra-realistic, LoRA support",
    },
    "flux-kontext-pro": {
        "endpoint": "/v1/ai/text-to-image/flux-kontext-pro",
        "supports_aspect_ratio": True,
        "aspect_ratios": ["square_1_1", "classic_4_3", "traditional_3_4", "widescreen_16_9", "social_story_9_16", "standard_3_2"],
        "description": "Flux Kontext Pro - Context-aware generation",
    },
    "flux-2-pro": {
        "endpoint": "/v1/ai/text-to-image/flux-2-pro",
        "supports_aspect_ratio": True,
        "supports_resolution": True,
        "description": "Flux 2 Pro - Professional-grade",
    },
    "flux-2-turbo": {
        "endpoint": "/v1/ai/text-to-image/flux-2-turbo",
        "supports_aspect_ratio": True,
        "description": "Flux 2 Turbo - Fast and cost-effective",
    },
    "flux-2-klein": {
        "endpoint": "/v1/ai/text-to-image/flux-2-klein",
        "supports_aspect_ratio": True,
        "supports_resolution": True,
        "resolutions": ["1k", "2k"],
        "description": "Flux 2 Klein - Sub-second generation",
    },
    "seedream-v4-5": {
        "endpoint": "/v1/ai/text-to-image/seedream-v4-5",
        "supports_aspect_ratio": False,
        "description": "Seedream 4.5 - Great for text/posters",
    },
    "seedream-v4": {
        "endpoint": "/v1/ai/text-to-image/seedream-v4",
        "supports_aspect_ratio": False,
        "description": "Seedream 4 - Next-gen text-to-image",
    },
    "z-image": {
        "endpoint": "/v1/ai/text-to-image/z-image",
        "supports_aspect_ratio": False,
        "description": "Z-Image - Fast, LoRA + ControlNet",
    },
}

# Gemini (Google) Model configurations
# Model IDs disponibles según la API de Gemini
GEMINI_MODELS = {
    # Modelos Gemini Image (usando generateContent)
    "gemini-2.5-flash-image": {
        "model_id": "gemini-2.5-flash-image",
        "api_type": "generateContent",
        "supports_aspect_ratio": True,
        "description": "Nano Banana - Modelo oficial de generación de imágenes",
    },
    "gemini-3-pro-image-preview": {
        "model_id": "gemini-3-pro-image-preview",
        "api_type": "generateContent",
        "supports_aspect_ratio": True,
        "description": "Nano Banana Pro - Versión Pro del modelo de imagen",
    },
    "nano-banana-pro-preview": {
        "model_id": "nano-banana-pro-preview",
        "api_type": "generateContent",
        "supports_aspect_ratio": True,
        "description": "Nano Banana Pro (Preview)",
    },
    "gemini-3.1-flash-image-preview": {
        "model_id": "gemini-3.1-flash-image-preview",
        "api_type": "generateContent",
        "supports_aspect_ratio": True,
        "description": "Nano Banana 2 - Versión 2 del modelo de imagen",
    },
    # Modelos Imagen (usando predict)
    "imagen-4.0-generate-001": {
        "model_id": "imagen-4.0-generate-001",
        "api_type": "predict",
        "supports_aspect_ratio": True,
        "description": "Imagen 4 - Modelo estándar",
    },
    "imagen-4.0-ultra-generate-001": {
        "model_id": "imagen-4.0-ultra-generate-001",
        "api_type": "predict",
        "supports_aspect_ratio": True,
        "description": "Imagen 4 Ultra - Mayor calidad",
    },
    "imagen-4.0-fast-generate-001": {
        "model_id": "imagen-4.0-fast-generate-001",
        "api_type": "predict",
        "supports_aspect_ratio": True,
        "description": "Imagen 4 Fast - Generación rápida",
    },
}


def get_freepik_model_config(model_name: str) -> dict:
    """Get configuration for a Freepik model."""
    return FREEPIK_MODELS.get(model_name, FREEPIK_MODELS["gemini-2-5-flash-image-preview"])


def build_freepik_payload(model_name: str, prompt: str) -> dict:
    """Build the appropriate payload for the selected model."""
    config = get_freepik_model_config(model_name)
    base_prompt = f"coloring book style image of {prompt}"
    
    # Get aspect ratio from page size setting
    page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    # Map page sizes to aspect ratios
    page_to_ratio = {
        "A4": "traditional_3_4",    # 210x297mm (portrait)
        "A5": "traditional_3_4",    # 148x210mm (portrait)
    }
    
    # Default payload structure
    payload = {"prompt": base_prompt}
    
    if model_name == "gemini-2-5-flash-image-preview":
        payload["reference_images"] = []
        payload["webhook_url"] = FREEPIK_WEBHOOK_URL
    
    elif model_name == "mystic":
        resolution = os.getenv("FREEPIK_RESOLUTION", "2k")
        payload.update({
            "resolution": resolution,
            "num_images": 1,
            "styling": {"style": "digital_art"},
        })
        if FREEPIK_WEBHOOK_URL:
            payload["webhook_url"] = FREEPIK_WEBHOOK_URL
    
    elif model_name in ["flux-kontext-pro", "flux-2-pro", "flux-2-turbo", "flux-2-klein"]:
        aspect_ratio = page_to_ratio.get(page_size, "traditional_3_4")
        if config.get("supports_aspect_ratio"):
            payload["aspect_ratio"] = aspect_ratio
        if config.get("supports_resolution"):
            resolution = os.getenv("FREEPIK_RESOLUTION", "2k")
            payload["resolution"] = resolution
        if FREEPIK_WEBHOOK_URL:
            payload["webhook_url"] = FREEPIK_WEBHOOK_URL
    
    elif model_name.startswith("seedream"):
        if FREEPIK_WEBHOOK_URL:
            payload["webhook_url"] = FREEPIK_WEBHOOK_URL
    
    else:
        # Generic fallback
        if FREEPIK_WEBHOOK_URL:
            payload["webhook_url"] = FREEPIK_WEBHOOK_URL
    
    return payload


def _call_gemini_generate_content(api_key: str, model_id: str, prompt: str, model_name: str) -> tuple[bool, bytes | None, str | None]:
    """Call Gemini API using generateContent endpoint. Returns (success, image_data, error_message)."""
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:generateContent?key={api_key}"
    
    prompt_text = (
        f"coloring book style, black and white line art outline drawing of {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"edge to edge drawing, fills the entire frame, no borders, no margins, "
        f"minimal text only, short labels or signs okay, NO paragraphs, "
        f"NO long text blocks, NO story text in the image, "
        f"no shading, no grayscale"
    )
    
    # Get aspect ratio from page size setting
    page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    aspect_ratio = "3:2" if page_size.upper() in ["A4", "A5"] else "1:1"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        "generationConfig": {
            "responseModalities": ["Text", "Image"],
            "temperature": 0.7,
            "imageConfig": {
                "aspectRatio": aspect_ratio,
            }
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        
        data = resp.json()
        
        # Extract image from response
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "inlineData" in part:
                        return True, base64.b64decode(part["inlineData"]["data"]), None
        
        return False, None, f"No image data in response: {data}"
        
    except requests.RequestException as exc:
        return False, None, f"Request error: {exc}"
    except Exception as exc:
        return False, None, f"Error: {exc}"


def _call_imagen_predict(api_key: str, model_id: str, prompt: str, model_name: str) -> tuple[bool, bytes | None, str | None]:
    """Call Imagen API using predict endpoint. Returns (success, image_data, error_message)."""
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base_url}/models/{model_id}:predict?key={api_key}"
    
    page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    aspect_ratio = "3:2" if page_size.upper() in ["A4", "A5"] else "1:1"
    
    prompt_text = (
        f"coloring book style, black and white line art outline drawing of {prompt}, "
        f"white background, clean thick lines suitable for children coloring page, "
        f"edge to edge drawing, fills the entire frame, no borders, no margins, "
        f"minimal text only, short labels or signs okay, NO paragraphs, "
        f"NO long text blocks, NO story text in the image, "
        f"no shading, no grayscale"
    )
    
    payload = {
        "instances": [{"prompt": prompt_text}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return False, None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        
        data = resp.json()
        predictions = data.get("predictions", [])
        
        if not predictions:
            return False, None, f"No predictions in response: {data}"
        
        image_data = predictions[0].get("bytesBase64Encoded")
        if not image_data:
            return False, None, "No image data in prediction"
        
        return True, base64.b64decode(image_data), None
        
    except requests.RequestException as exc:
        return False, None, f"Request error: {exc}"
    except Exception as exc:
        return False, None, f"Error: {exc}"


def send_gemini_image_request(prompt: str, model_name: str = "gemini-2.5-flash-image"):
    """Call Google Gemini/Imagen API for image generation."""
    api_key = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]GEMINI_API_KEY or GOOGLE_API_KEY not set; skipping image generation.[/red]")
        return
    
    # Default to gemini-2.5-flash-image if model not found
    if model_name not in GEMINI_MODELS:
        console.print(f"[yellow]Model {model_name} not found, using gemini-2.5-flash-image[/yellow]")
        model_name = "gemini-2.5-flash-image"
    
    config = GEMINI_MODELS[model_name]
    model_id = config["model_id"]
    api_type = config.get("api_type", "generateContent")
    
    console.print(f"[blue]Requesting Gemini image ({model_name}) for: {prompt}[/blue]")
    
    # Call appropriate API based on model type
    if api_type == "predict":
        success, image_bytes, error = _call_imagen_predict(api_key, model_id, prompt, model_name)
    else:
        success, image_bytes, error = _call_gemini_generate_content(api_key, model_id, prompt, model_name)
    
    if not success:
        console.print(f"[red]Gemini API error:[/red] {error}")
        return
    
    # Save image
    try:
        filename = f"generated_image_gemini_{int(time.time())}.png"
        with open(filename, "wb") as fh:
            fh.write(image_bytes)
        console.print(f"[bold green]Image saved to {filename}[/bold green]")
    except Exception as exc:
        console.print(f"[red]Failed to save image:[/red] {exc}")
        return
    
    # Process for printing/PDF
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
            target_path = make_print_image_copy(filename, str(int(time.time())))
        try:
            subprocess.run([PRINT_COMMAND, target_path or filename], check=True)
            console.print(f"[green]Sent image to printer via {PRINT_COMMAND}: {target_path or filename}[/green]")
            # Sound 4 = jump (printing/screen display)
            if wifi_socket:
                send_sound_command(wifi_socket, 4)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Printing failed ({PRINT_COMMAND}):[/red] {exc}")
        except Exception as exc:
            console.print(f"[red]Unexpected printing error:[/red] {exc}")
    elif pdf_path:
        console.print(f"[green]PDF copy ready at {pdf_path} (printing disabled).[/green]")


def send_image_generation_request(prompt: str):
    """Call image generation API based on the transcript and provider settings."""
    provider = get_config("IMAGE_PROVIDER", IMAGE_PROVIDER).lower()
    
    if provider == "gemini":
        model_name = get_config("GEMINI_MODEL", GEMINI_MODEL)
        send_gemini_image_request(prompt, model_name)
        return
    
    # Default to Freepik
    api_key = os.getenv("FREEPIK_API_KEY")
    if not api_key:
        console.print("[red]FREEPIK_API_KEY not set; skipping image generation.[/red]")
        return
    
    model_name = get_config("FREEPIK_MODEL", FREEPIK_MODEL)
    config = get_freepik_model_config(model_name)
    
    base_url = "https://api.freepik.com"
    url = f"{base_url}{config['endpoint']}"
    
    payload = build_freepik_payload(model_name, prompt)
    headers = {
        "Content-Type": "application/json",
        "x-freepik-api-key": api_key,
    }

    try:
        console.print(f"[blue]Requesting Freepik image ({model_name}) for: {prompt}[/blue]")
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Try different locations for task_id depending on model
        task_id = data.get("data", {}).get("task_id") or data.get("task_id") or data.get("id")
        if not task_id:
            console.print("[red]No task_id returned from Freepik; aborting image generation.[/red]")
            return

        console.print(f"[green]Freepik task id: {task_id}. Polling for completion...[/green]")
        poll_deadline = time.time() + 120  # 2 minute timeout
        image_url = None
        while time.time() < poll_deadline:
            poll_resp = requests.get(f"{url}/{task_id}", headers=headers, timeout=15)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            
            # Handle different status field locations
            status = poll_data.get("data", {}).get("status") or poll_data.get("status")
            console.print(f"[cyan]Freepik status: {status}[/cyan]")
            
            if status in ("COMPLETED", "READY", "completed", "ready"):
                # Try different locations for generated image URL
                generated = poll_data.get("data", {}).get("generated") or []
                if generated:
                    image_url = generated[0]
                elif poll_data.get("data", {}).get("result", {}).get("url"):
                    image_url = poll_data["data"]["result"]["url"]
                elif poll_data.get("data", {}).get("image", {}).get("url"):
                    image_url = poll_data["data"]["image"]["url"]
                elif "url" in poll_data:
                    image_url = poll_data["url"]
                elif "image_url" in poll_data:
                    image_url = poll_data["image_url"]
                break
            if status in ("FAILED", "failed"):
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
                target_path = make_print_image_copy(filename, task_id)
            try:
                subprocess.run([PRINT_COMMAND, target_path or filename], check=True)
                console.print(f"[green]Sent image to printer via {PRINT_COMMAND}: {target_path or filename}[/green]")
                # Sound 4 = jump (printing/screen display)
                if wifi_socket:
                    send_sound_command(wifi_socket, 4)
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


def get_page_dimensions(page_size: str = None) -> tuple[int, int]:
    """Return page dimensions in pixels at 300 DPI for A4 or A5."""
    if page_size is None:
        page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    size = page_size.upper()
    if size == "A5":
        return 1748, 2480  # 148x210 mm at 300 DPI
    return 2480, 3508  # A4 default: 210x297 mm at 300 DPI


def save_pdf_copy(image_path: str):
    """Optionally write a PDF copy of the generated image."""
    if not pdf_enabled():
        return None
    if Image is None:
        console.print("[yellow]PRINT_TO_PDF is enabled but Pillow is not installed; skipping PDF export.[/yellow]")
        return None
    page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    page_w, page_h = get_page_dimensions(page_size)
    try:
        os.makedirs(PRINT_PDF_DIR, exist_ok=True)
        pdf_path = Path(PRINT_PDF_DIR) / (Path(image_path).stem + ".pdf")
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            # Create canvas at 300 DPI and center-fit the image
            margin = 120  # ~10 mm margin
            canvas = Image.new("RGB", (page_w, page_h), "white")
            max_w, max_h = page_w - 2 * margin, page_h - 2 * margin
            # Prefer filling the width (with aspect ratio) unless it exceeds page height
            target_w = max_w
            target_h = int(img.height * (target_w / img.width))
            if target_h > max_h:
                target_h = max_h
                target_w = int(img.width * (target_h / img.height))
            resized = img.resize((target_w, target_h))
            x = (page_w - target_w) // 2
            y = (page_h - target_h) // 2
            canvas.paste(resized, (x, y))
            canvas.save(pdf_path, "PDF", resolution=300.0)
        console.print(f"[green]Saved PDF copy to {pdf_path} ({page_size})[/green]")
        return str(pdf_path)
    except Exception as exc:
        console.print(f"[red]PDF export failed:[/red] {exc}")
        return None


def make_print_image_copy(image_path: str, task_id: str = ""):
    """Build a print-friendly PNG with the image scaled to fill width (A4 or A5)."""
    if Image is None:
        return None
    page_size = get_config("PRINT_PAGE_SIZE", PRINT_PAGE_SIZE)
    page_w, page_h = get_page_dimensions(page_size)
    try:
        os.makedirs(PRINT_PDF_DIR, exist_ok=True)
        out_name = f"print_image_{task_id or Path(image_path).stem}.png"
        out_path = Path(PRINT_PDF_DIR) / out_name
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            margin = 120
            max_w, max_h = page_w - 2 * margin, page_h - 2 * margin
            target_w = max_w
            target_h = int(img.height * (target_w / img.width))
            if target_h > max_h:
                target_h = max_h
                target_w = int(img.width * (target_h / img.height))
            resized = img.resize((target_w, target_h))
            canvas = Image.new("RGB", (page_w, page_h), "white")
            x = (page_w - target_w) // 2
            y = (page_h - target_h) // 2
            canvas.paste(resized, (x, y))
            canvas.save(out_path, "PNG")
        console.print(f"[green]Prepared {page_size} print image at {out_path}[/green]")
        return str(out_path)
    except Exception as exc:
        console.print(f"[red]Print image prep failed:[/red] {exc}")
        return None


def resolve_compute_type(device: str, compute_type: str) -> str:
    """
    The pure int8 path is CPU-only; for CUDA use int8_float16 to keep quantization
    benefits without tripping unsupported kernels.
    """
    if device.lower() == "cuda" and compute_type.lower() == "int8":
        console.print("[yellow]WHISPER_COMPUTE_TYPE=int8 is CPU-only; using int8_float16 for CUDA.[/yellow]")
        return "int8_float16"
    return compute_type


def load_whisper_model() -> WhisperModel:
    """
    Try CUDA first, but fall back to CPU so the service still runs if the GPU setup
    is missing the right runtime or drivers.
    """
    attempts = [(DEVICE, resolve_compute_type(DEVICE, COMPUTE_TYPE))]
    if DEVICE.lower() == "cuda":
        attempts.append(("cpu", "int8"))

    last_error = None
    for device, compute_type in attempts:
        console.print(f"[bold blue]Loading Whisper model: {MODEL_SIZE} ({device}, {compute_type})[/bold blue]")
        try:
            return WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        except Exception as exc:
            last_error = exc
            console.print(f"[red]Failed to load Whisper model on {device}/{compute_type}:[/red] {exc}")
            if device.lower() == "cuda":
                console.print("[yellow]CUDA failed; will retry on CPU unless WHISPER_DEVICE is overridden.[/yellow]")

    raise SystemExit(f"Could not load Whisper model: {last_error}")


def main():
    model = load_whisper_model()

    transcriber = threading.Thread(target=transcribe_worker, args=(model,), daemon=True)
    flusher = threading.Thread(target=flush_loop, daemon=True)
    cfg_thread = threading.Thread(target=config_watcher, daemon=True)
    transcriber.start()
    flusher.start()
    cfg_thread.start()
    update_runtime_state(RUNNING=True, READY=False, BUTTON_STATE="idle")

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
        update_runtime_state(RUNNING=False, READY=False, BUTTON_STATE="idle")
        console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
