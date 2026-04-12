# Vocolore

Vocolore turns kids’ voice descriptions into printed coloring pages. It captures speech over Wi‑Fi (ESP32), transcribes it with Whisper, generates an image via Freepik, and (optionally) opens/prints the page for coloring.

## Features

- **WiFi kid-friendly flow**: Capture audio from an ESP32 (Atom Echo), transcribe with Whisper, generate a Freepik image, and optionally print/open it.
- **Debug & user UIs**: Debug UI exposes live toggles (Freepik on/off, open/print, min/max clip durations, timing logs) and launches the transcriber; User UI shows readiness, button state, transcription, and last image preview.

## Setup

To set up and run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Laia-ar/vocolore.git
   cd vocolore
   ```

2. **Install dependencies** (using [uv](https://docs.astral.sh/uv/)):
   ```bash
   uv sync
   ```

## Usage

### WiFi Transcription (ESP32 stream)

- Easiest way (launches debug + user UIs together):  
  ```bash
  python run_wifi_and_ui.py
  ```
  Debug UI autostarts the transcriber and shows logs; User UI shows readiness, button state, transcription, and image preview.
- Core listener/transcriber only:  
  ```bash
  python wifi_transcribe.py
  ```
- Debug UI with live toggles (Freepik, open/print image, min/max clip durations):  
  ```bash
  python wifi_debug_ui.py
  ```
  Writes runtime config JSON watched by `wifi_transcribe.py`.
- Simple user UI (status + preview, no controls by default):  
  ```bash
  python wifi_user_ui.py
  ```

## Platform Support

### Linux / Windows (NVIDIA GPU)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- NVIDIA GPU with CUDA (optional but recommended)
- Dependencies: `openai-whisper`, `torch`, `sounddevice`, `numpy`, `rich`, `pynput`, `soundfile`, `python-dotenv`, `requests`, `Pillow`

### macOS (Apple Silicon - M1/M2/M3)
Vocolore supports macOS with Apple Silicon through MPS (Metal Performance Shaders) or CPU fallback.

**Additional requirements for Mac:**
```bash
# Install portaudio for audio support
brew install portaudio

# Install PyTorch with MPS support
uv pip install torch torchaudio

# The device will auto-detect: MPS (GPU) -> CPU fallback
```

**Note:** The `WHISPER_DEVICE=auto` setting will automatically use:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

## Freepik API Key

Some functionalities of this project may require a Freepik API key. To use these features, you need to set up your API key as an environment variable.

1.  **Create a `config.env` file**:
    Copy the `sample.config.env` file to `config.env` in the root directory of the project:
    ```bash
    cp sample.config.env config.env
    ```
2.  **Add your API key**:
    Open the newly created `config.env` file and add your Freepik API key:
    ```
    FREEPIK_API_KEY=your_freepik_api_key_here
    ```
    Replace `your_freepik_api_key_here` with your actual Freepik API key.

Optional printing options:
```
# send directly to a printer
PRINT_IMAGE=1
PRINT_COMMAND=/usr/bin/lp

# or export a PDF copy alongside the PNG
PRINT_TO_PDF=1
PRINT_PDF_DIR=printouts
# PDF export is rendered on an A4 canvas (300 DPI)
```
