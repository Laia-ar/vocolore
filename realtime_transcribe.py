# transcribe_stream.py
import argparse, threading, time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rich.console import Console
from rich.live import Live
from pynput import keyboard # Import pynput keyboard

console = Console()

# Global variables for recording state
recording_active = False
recorded_audio_buffer = []

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=None, help="Input device index or name (use --list)")
parser.add_argument("--model", type=str, default="small", help="tiny/base/small/medium/large-v3")
parser.add_argument("--lang", type=str, default=None, help="Force language, e.g. es, en (None = auto)")
parser.add_argument("--block", type=float, default=2.0, help="Seconds per audio block")
parser.add_argument("--window", type=float, default=6.0, help="Rolling window seconds to transcribe")
parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
parser.add_argument("--list", action="store_true", help="List devices and exit")
parser.add_argument("--cuda", action="store_true", help="Force CUDA (otherwise CPU)")
parser.add_argument("--debug", action="store_true", help="Print debug meters and events")
args = parser.parse_args()

if args.list:
    console.print(sd.query_devices())
    raise SystemExit

# ----- device & rates -----
dev_info = sd.query_devices(args.device, kind="input")
src_rate = int(dev_info["default_samplerate"])
TARGET_RATE = 16000

def resample_to_16k(wave: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_RATE:
        return wave.astype(np.float32, copy=False)
    n_src = wave.shape[0]
    n_dst = int(round(n_src * TARGET_RATE / src_sr))
    x = np.arange(n_src, dtype=np.float64)
    x_new = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
    y = np.interp(x_new, x, wave.astype(np.float32))
    return y.astype(np.float32, copy=False)

device_str = "cuda" if args.cuda else "cpu"
model = WhisperModel(args.model, device=device_str)

# block_s and win_s are still needed for InputStream blocksize and for consistency
block_s = max(0.5, float(args.block))
win_s = max(block_s, float(args.window))
max_samples = int(TARGET_RATE * win_s) # max_samples is not strictly needed for push-to-talk but kept for consistency

# Audio callback function
def audio_cb(indata, frames, time_info, status):
    global recorded_audio_buffer
    if status:
        console.print(f"[red]Audio status:[/red] {status}")
    if recording_active:
        mono = indata[:, 0]
        recorded_audio_buffer.append(mono.copy())

# Function to handle key press
def on_press(key):
    global recording_active, recorded_audio_buffer
    try:
        if key.name == 'space' and not recording_active:
            recording_active = True
            recorded_audio_buffer = [] # Clear buffer for new recording
            console.print("[green]Recording... (Release Spacebar to Transcribe)[/green]")
    except AttributeError:
        pass # Special keys like Ctrl, Alt, etc.

# Function to handle key release
def on_release(key):
    global recording_active, recorded_audio_buffer
    try:
        if key.name == 'space' and recording_active:
            recording_active = False
            console.print("[yellow]Processing audio...[/yellow]")
            if recorded_audio_buffer:
                # Concatenate all recorded blocks
                full_audio = np.concatenate(recorded_audio_buffer)
                # Resample to 16k
                audio_16k = resample_to_16k(full_audio, src_rate)

                if audio_16k.size == 0:
                    console.print("[red]No audio recorded.[/red]")
                    return

                # Transcribe the recorded audio
                try:
                    segments, _ = model.transcribe(
                        audio_16k,
                        language=args.lang,
                        vad_filter=not args.no_vad,
                        no_speech_threshold=0.2,
                        log_prob_threshold=-1.0,
                        compression_ratio_threshold=2.6,
                        beam_size=1,
                    )
                    transcribed_text = ""
                    for seg in segments:
                        transcribed_text += seg.text.strip() + " "
                    if transcribed_text.strip():
                        console.print(f"[bold cyan]Transcription:[/bold cyan] {transcribed_text.strip()}")
                    else:
                        console.print("[grey58]No speech detected.[/grey58]")
                except Exception as e:
                    console.print(f"[red]Transcribe error:[/red] {e}")
            else:
                console.print("[red]No audio recorded.[/red]")
    except AttributeError:
        pass

# UI banner
console.print(f"[green]Mic:[/green] {dev_info['name']}")
console.print(f"[green]Source rate:[/green] {src_rate} Hz  →  [green]Target:[/green] {TARGET_RATE} Hz")
console.print(f"[green]Device:[/green] {device_str.upper()}   [green]Model:[/green] {args.model}")
console.print(f"[green]Press and hold SPACEBAR to record, release to transcribe.[/green]")

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    with sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=src_rate,
        blocksize=int(src_rate * block_s), # Keep blocksize for audio_cb
        dtype="float32",
        callback=audio_cb,
        latency="low",
    ):
        console.print("[green]Listening for Spacebar... Ctrl+C to stop.[/green]")
        # Keep the main thread alive to listen for keyboard events
        while True:
            time.sleep(0.1) # Small sleep to prevent busy-waiting
except KeyboardInterrupt:
    pass
finally:
    console.print("[yellow]Stopped.[/yellow]")
    listener.stop() # Stop the pynput listener
    listener.join() # Wait for the listener thread to finish
