# Vocolore - Agent Documentation

## Project Overview

Vocolore is a voice-to-coloring-page system designed for children. It captures voice descriptions via an ESP32 device (M5Atom Echo), transcribes the audio using OpenAI Whisper, generates coloring book images via the Freepik API, and optionally prints or opens the resulting images.

The project bridges embedded hardware (ESP32), AI speech recognition, and generative AI image creation into a kid-friendly experience.

## Technology Stack

- **Language**: Python 3.x
- **Speech Recognition**: faster-whisper (OpenAI Whisper implementation)
- **UI Framework**: Tkinter (standard Python GUI)
- **Hardware**: M5Atom Echo (ESP32-based) with Arduino firmware
- **Image Generation**: Freepik API (Gemini 2.5 Flash Image Preview)
- **Image Processing**: Pillow (PIL)
- **Networking**: Standard Python socket library
- **Audio Processing**: NumPy, sounddevice, soundfile

## Project Structure

```
.
├── wifi_transcribe.py          # Core transcription service - main backend
├── wifi_debug_ui.py            # Debug UI with live toggles and logs
├── wifi_user_ui.py             # User-friendly UI with status indicators
├── run_wifi_and_ui.py          # Launcher for both UIs
├── test.py                     # Standalone Whisper test script
├── M5Atom_WiFi_Serial_Button_Audio/
│   └── M5Atom_WiFi_Serial_Button_Audio.ino  # ESP32 Arduino firmware
├── config.env                  # Environment configuration (gitignored)
├── sample.config.env           # Sample configuration template
├── .runtime_config.json        # Runtime toggles (created dynamically)
├── pyproject.toml              # Project configuration (uv)
├── clips/                      # Saved audio clips (gitignored)
└── images/                     # Generated images (gitignored)
```

## Architecture

### Core Components

1. **wifi_transcribe.py** (Backend Service)
   - Connects to ESP32 via TCP socket (port 5005)
   - Receives audio packets using a framed protocol: `[type:1][len:2][payload:len]`
   - Type 'A' = audio (16-bit mono PCM), Type 'C' = control (button events)
   - Resamples audio from 8kHz (WiFi) to 16kHz (Whisper)
   - Runs transcription in a separate thread using faster-whisper
   - Optionally triggers Freepik image generation
   - Supports runtime configuration via `.runtime_config.json` file watcher

2. **wifi_debug_ui.py** (Debug Interface)
   - Tkinter-based UI for debugging and configuration
   - Start/stop control for the transcription service
   - Live toggles: Freepik, Open image, Print image, PDF copy, Debug timing
   - Numeric settings: MIN_AUDIO_SEC, MAX_BUFFER_SEC
   - Full log output viewer
   - Writes configuration to `.runtime_config.json`

3. **wifi_user_ui.py** (User Interface)
   - Kid-friendly UI with large status indicators
   - Shows: Ready status, Button state, Transcription status, Freepik status
   - Displays latest transcription text
   - Shows image preview of last generated image
   - Parses stdout from wifi_transcribe.py to update UI state

4. **ESP32 Firmware** (M5Atom Echo)
   - Creates WiFi Access Point (SSID: AtomEchoAP, default IP: 192.168.4.1)
   - Captures audio via I2S microphone when button is pressed
   - Sends audio packets to connected TCP client
   - Sends button events (DOWN/UP) as control packets
   - LED status indication: Red (offline), Blue (waiting), Green (connected), Orange (recording)

### Data Flow

```
[M5Atom Echo] --(TCP/audio packets)--> [wifi_transcribe.py]
                                            |
                                            v
                                    [Whisper Transcription]
                                            |
                                            v
                                    [Freepik API Call]
                                            |
                                            v
                                    [Image Generation]
                                            |
                    +-----------------------+-----------------------+
                    |                       |                       |
                    v                       v                       v
              [Save PNG]            [Print (optional)]      [PDF Export]
                    |
                    v
            [Update UI via
             runtime_config.json]
```

## Configuration

### Environment Variables (config.env)

Copy `sample.config.env` to `config.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `FREEPIK_API_KEY` | - | Required for image generation |
| `WIFI_HOST` | 192.168.4.1 | ESP32 AP IP address |
| `WIFI_PORT` | 5005 | TCP port for audio stream |
| `WIFI_SAMPLE_RATE` | 8000 | Audio sample rate from device |
| `MODEL_SIZE` | base | Whisper model (tiny, base, small, medium, large-v3) |
| `WHISPER_DEVICE` | cuda | Device for inference (cuda/cpu) |
| `WHISPER_COMPUTE_TYPE` | float16 | Compute precision (float16, int8, int8_float16) |
| `TRANSCRIBE_LANGUAGE` | es | Language code for transcription |
| `SILENCE_TIMEOUT` | 1.5 | Seconds of silence before flush |
| `MIN_AUDIO_SEC` | 4.0 | Minimum clip length to process |
| `MAX_BUFFER_SEC` | 16.0 | Force flush when buffer exceeds this |
| `ENABLE_FREEPIK` | 0 | Enable image generation (0/1) |
| `PRINT_IMAGE` | 0 | Send to printer via lp command (0/1) |
| `PRINT_TO_PDF` | 0 | Save PDF copy (0/1) |
| `OPEN_IMAGE` | 0 | Auto-open generated image (0/1) |
| `SAVE_CLIP_WAV` | 0 | Save individual audio clips (0/1) |
| `CLIP_WAV_DIR` | clips | Directory for saved clips |
| `PRINT_PAGE_SIZE` | A4 | Page size for PDF/print (A4 or A5) |
| `IMAGE_PROVIDER` | freepik | Image provider: "freepik" or "gemini" |
| `FREEPIK_MODEL` | gemini-2-5-flash-image-preview | Freepik model (see table below) |
| `GEMINI_MODEL` | gemini-2.5-flash-image | Gemini model (see table below) |

**Available Freepik Models:**

| Model | Endpoint | Description | Aspect Ratio Support |
|-------|----------|-------------|---------------------|
| `gemini-2-5-flash-image-preview` | `/v1/ai/gemini-2-5-flash-image-preview` | Gemini Flash (undocumented but works) | No |
| `mystic` | `/v1/ai/mystic` | Freepik exclusive, ultra-realistic, LoRA | No |
| `flux-kontext-pro` | `/v1/ai/text-to-image/flux-kontext-pro` | Context-aware, image input support | Yes (6 ratios) |
| `flux-2-pro` | `/v1/ai/text-to-image/flux-2-pro` | Professional-grade, up to 4 input images | Yes |
| `flux-2-turbo` | `/v1/ai/text-to-image/flux-2-turbo` | Fast and cost-effective | Yes |
| `flux-2-klein` | `/v1/ai/text-to-image/flux-2-klein` | Sub-second generation | Yes |
| `seedream-v4-5` | `/v1/ai/text-to-image/seedream-v4-5` | Great for text/posters, up to 4MP | No |
| `seedream-v4` | `/v1/ai/text-to-image/seedream-v4` | Next-gen text-to-image | No |
| `z-image` | `/v1/ai/text-to-image/z-image` | Fast, LoRA + ControlNet | No |

**Available Gemini Models (Google API):**

| Model | Description |
|-------|-------------|
| `gemini-2.0-flash-preview` | Fast image generation (recommended) |
| `gemini-1.5-flash` | Standard image generation |
| `gemini-nano-banana` | Alias for 2.0-flash-preview |
| `gemini-nano-banana-pro` | Alias for 2.0-flash-preview (higher quality settings) |

Get Gemini API key from: https://ai.google.dev/

**Note:** Gemini models don't support aspect ratio parameter. Images are generated at the model's native resolution.

### Runtime Configuration (.runtime_config.json)

This file is created dynamically and watched by `wifi_transcribe.py`. It allows runtime toggling without restarting:

- `ENABLE_FREEPIK` - Enable/disable image generation
- `PRINT_IMAGE` - Enable/disable printing
- `OPEN_IMAGE` - Enable/disable auto-open
- `PRINT_TO_PDF` - Enable/disable PDF export
- `MIN_AUDIO_SEC` - Minimum audio clip duration
- `MAX_BUFFER_SEC` - Maximum buffer duration
- `DEBUG_TIMING` - Enable timing debug logs
- `RUNNING` - Service running state
- `READY` - Connection ready state
- `BUTTON_STATE` - Current button state
- `LAST_IMAGE` - Path to last generated image

## Build and Run Commands

### Installation (using uv)

**Prerequisite:** Install uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
# Create virtual environment and install dependencies
uv sync

# Or if you want to run directly without activating:
uv run python wifi_transcribe.py
```

### Running the Application

**Recommended (both UIs):**
```bash
python run_wifi_and_ui.py
```

**Individual components:**
```bash
# Core transcription service only
python wifi_transcribe.py

# Debug UI (with controls and logs)
python wifi_debug_ui.py

# User UI (kid-friendly interface)
python wifi_user_ui.py
```

### Environment Variables for Launchers

- `AUTO_START=1` - Automatically start transcription in debug UI
- `LAUNCH_TRANSCRIBE=1` - Auto-start transcription in user UI
- `DEBUG=1` - Enable debug output

## Code Style Guidelines

- **Imports**: Standard library first, then third-party, then local
- **Type hints**: Used where beneficial (e.g., `sock: socket.socket`)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Thread safety**: Use `threading.Lock()` for shared state access
- **Logging**: Use `rich.console.Console` for colored terminal output
- **Error handling**: Try/except blocks with descriptive error messages via console.print()

## Key Implementation Details

### Audio Processing

- Audio arrives as 16-bit mono PCM at 8kHz from ESP32
- Resampled to 16kHz for Whisper using NumPy interpolation
- Buffering with min/max duration constraints
- DC offset removal and channel selection (left/right) on ESP32 side

### Threading Model

- `wifi_listener()` - Main thread, socket I/O
- `transcribe_worker()` - Daemon thread, Whisper inference
- `config_watcher()` - Daemon thread, file watching
- UI subprocesses run in separate processes

### Protocol Details

The ESP32 uses a simple framed binary protocol:
- Header: 3 bytes `[type][length_high][length_low]`
- Type 'A' (0x41): Audio payload (16-bit PCM)
- Type 'C' (0x43): Control payload (UTF-8 text: "DOWN", "UP")
- Type 'B' (0x42): Battery level (UTF-8 text: percentage, e.g., "85")

### GPU/CPU Fallback

`wifi_transcribe.py` implements automatic fallback:
1. Try CUDA with specified compute type
2. If CUDA fails, fallback to CPU with int8
3. Exit if both fail

## Testing

The `test.py` script provides standalone Whisper testing:
```bash
python test.py  # Transcribes a sample clip
```

For manual testing:
1. Flash ESP32 with Arduino sketch
2. Connect to AtomEchoAP WiFi network
3. Run `python wifi_transcribe.py`
4. Press and hold button on M5Atom to record
5. Speak, release button, wait for transcription

## Security Considerations

- Freepik API key stored in `config.env` (gitignored)
- `.runtime_config.json` contains no sensitive data
- ESP32 AP uses WPA2 with configurable password
- No authentication on TCP socket (assumed trusted local network)
- Generated images saved to local filesystem

## Hardware Requirements

- M5Atom Echo or compatible ESP32 with I2S microphone
- Computer with Python 3.x
- Optional: CUDA-capable GPU for faster transcription
- Optional: Network printer for automatic printing

## Dependencies

Dependencies are defined in `pyproject.toml`:
- faster-whisper
- sounddevice
- numpy
- rich
- pynput
- soundfile
- python-dotenv
- requests
- Pillow

Install with: `uv sync`
