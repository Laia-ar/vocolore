# Whisper Realtime Transcription

This project provides real-time audio transcription capabilities using OpenAI's Whisper model. It includes scripts for both push-to-talk and continuous real-time transcription.

## Features

- **Push-to-Talk Transcription**: Transcribe audio only when a key is pressed.
- **Real-time Continuous Transcription**: Continuously transcribe audio from your microphone.

## Setup

To set up and run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Laia-ar/vocolore.git
   cd vocolore
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Push-to-Talk Transcription

To use the push-to-talk feature, run:

```bash
python push_to_talk_transcribe.py
```
Press and hold the spacebar to record, and release to transcribe.

### Real-time Continuous Transcription

To use the real-time continuous transcription feature, run:

```bash
python realtime_transcribe.py
```
The transcription will appear in your console as you speak.

### WiFi Transcription (ESP32 stream)

- Core listener/transcriber:  
  ```bash
  python wifi_transcribe.py
  ```
- Debug UI with live toggles (Freepik, print/open image, min/max clip durations):  
  ```bash
  python wifi_debug_ui.py
  ```
  This writes a runtime config JSON watched by `wifi_transcribe.py`.
- Simple user UI (start/stop, status, latest transcript/image):  
  ```bash
  python wifi_user_ui.py
  ```

## Requirements

- Python 3.x
- `requirements.txt` dependencies (e.g., `sounddevice`, `whisper`, `numpy`, `queue`)

## Freepik API Key

Some functionalities of this project may require a Freepik API key. To use these features, you need to set up your API key as an environment variable.

1.  **Create a `.env` file**:
    Copy the `sample.config.env` file to `.env` in the root directory of the project:
    ```bash
    cp sample.config.env .env
    ```
2.  **Add your API key**:
    Open the newly created `.env` file and add your Freepik API key:
    ```
    FREEPIK_API_KEY=your_freepik_api_key_here
    ```
    Replace `your_freepik_api_key_here` with your actual Freepik API key.
