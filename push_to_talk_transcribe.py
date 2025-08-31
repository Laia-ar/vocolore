import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from pynput import keyboard
from rich.console import Console
from rich.text import Text
import threading
import collections
import queue
import sys
import os
import soundfile as sf # Added for saving audio to file
import json # Added for handling JSON payload
import subprocess # Added for executing curl command
from dotenv import load_dotenv # Added for loading environment variables from .env file
import requests # Added for downloading the generated image
import time # Added for polling delay

# Configuration
MODEL_SIZE = "base"  # or "base", "small", "medium", "large-v2"
SAMPLERATE = 48000
CHANNELS = 1
DTYPE = "int16"
KEY_TO_PRESS = keyboard.Key.space # Key to hold for push-to-talk

# Initialize console for rich output
console = Console()

# Global variables for audio recording
audio_queue = collections.deque()
recording = False
stream = None
model = None
selected_device_id = None # Added global variable for selected device ID

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        console.print(f"[red]Audio callback status: {status}[/red]")
    if recording:
        audio_queue.append(indata.copy())

def start_recording():
    global recording, stream, selected_device_id # Added selected_device_id to global
    if not recording:
        console.print("[green]Recording started... (Press and hold SPACE)[/green]")
        recording = True
        try:
            # Pass the selected_device_id to sd.InputStream if it's set
            if selected_device_id is not None:
                stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback, device=selected_device_id)
            else:
                stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype=DTYPE, callback=audio_callback)
            stream.start()
        except Exception as e:
            console.print(f"[red]Error starting audio stream: {e}[/red]")
            recording = False

def stop_recording():
    global recording, stream
    if recording:
        console.print("[yellow]Recording stopped. Transcribing...[/yellow]")
        recording = False
        if stream:
            stream.stop()
            stream.close()
            stream = None
        
        # Process audio data
        if audio_queue:
            audio_data = np.concatenate(audio_queue, axis=0)
            audio_queue.clear()
            
            # Convert to float32 for faster-whisper
            audio_data_float32 = audio_data.astype(np.float32) / 32768.0

            # Debugging: Check audio data properties
            console.print(f"[blue]Recorded audio length: {len(audio_data_float32)} samples[/blue]")
            if len(audio_data_float32) > 0:
                max_amplitude = np.max(np.abs(audio_data_float32))
                console.print(f"[blue]Max amplitude: {max_amplitude:.4f}[/blue]")
                if max_amplitude < 0.01: # Arbitrary threshold for very low volume
                    console.print("[orange]Warning: Recorded audio seems very quiet. Check microphone input.[/orange]")

            # Transcribe
            if model:
                # Save audio to a temporary WAV file
                temp_audio_file = "temp_recorded_audio.wav"
                try:
                    sf.write(temp_audio_file, audio_data_float32, SAMPLERATE)
                    console.print(f"[blue]Audio saved to {temp_audio_file}[/blue]")

                    segments, info = model.transcribe(temp_audio_file, beam_size=5, language="es")
                    transcription = ""
                    for segment in segments:
                        transcription += segment.text
                except Exception as e:
                    console.print(f"[red]Error saving or transcribing audio from file: {e}[/red]")
                    transcription = "" # Clear transcription on error
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_audio_file):
                        os.remove(temp_audio_file)
                        console.print(f"[blue]Removed temporary audio file: {temp_audio_file}[/blue]")
                
                if transcription.strip():
                    console.print(Text(f"Transcription: {transcription}", style="bold magenta"))
                    # Send image generation request after successful transcription
                    send_image_generation_request(transcription.strip())
                else:
                    console.print("[grey]No speech detected.[/grey]")
            else:
                console.print("[red]Whisper model not loaded.[/red]")
        else:
            console.print("[grey]No audio recorded.[/grey]")

def on_press(key):
    global recording
    try:
        if key == KEY_TO_PRESS and not recording:
            start_recording()
    except AttributeError:
        pass # Handle special keys if needed

def send_image_generation_request(prompt):
    freepik_api_key = os.getenv("FREEPIK_API_KEY")
    if not freepik_api_key:
        console.print("[red]Error: FREEPIK_API_KEY environment variable not set.[/red]")
        return

    payload = {
        "prompt": "coloring book style image of " + prompt, # Prepend instructions for coloring book style
        "reference_images": [], # User provided example with empty array and a URL, so keeping it flexible
        "webhook_url": "https://www.example.com/webhook" # User provided example webhook
    }

    try:
        command = [
            "curl",
            "--request", "POST",
            "--url", "https://api.freepik.com/v1/ai/gemini-2-5-flash-image-preview",
            "--header", "Content-Type: application/json",
            "--header", f"x-freepik-api-key: {freepik_api_key}",
            "--data", json.dumps(payload)
        ]
        
        console.print(f"[blue]Sending image generation request for prompt: '{prompt}'[/blue]")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        console.print(f"[green]Image generation request successful:[/green]\n{result.stdout}")
        
        response_data = json.loads(result.stdout)
        task_id = None
        # Extract task_id from 'data.task_id' as per the API response
        if 'data' in response_data and 'task_id' in response_data['data']:
            task_id = response_data['data']['task_id']

        if task_id:
            console.print(f"[blue]Image generation task ID: {task_id}. Polling for image status...[/blue]")
            
            # Polling mechanism
            timeout = 60 # seconds
            start_time = time.time()
            image_url = None

            while time.time() - start_time < timeout:
                get_image_command = [
                    "curl",
                    "--request", "GET",
                    "--url", f"https://api.freepik.com/v1/ai/gemini-2-5-flash-image-preview/{task_id}",
                    "--header", f"x-freepik-api-key: {freepik_api_key}"
                ]
                try:
                    get_result = subprocess.run(get_image_command, capture_output=True, text=True, check=True)
                    get_response_data = json.loads(get_result.stdout)
                    
                    status = None
                    if 'data' in get_response_data and 'status' in get_response_data['data']:
                        status = get_response_data['data']['status']
                    
                    console.print(f"[blue]Current image generation status: {status}[/blue]")

                    if status == "COMPLETED" or status == "READY": # Assuming "COMPLETED" or "READY" indicates image is available
                        if 'data' in get_response_data and 'generated' in get_response_data['data'] and len(get_response_data['data']['generated']) > 0:
                            image_url = get_response_data['data']['generated'][0]
                        elif 'url' in get_response_data: # Fallback for other possible structures
                            image_url = get_response_data['url']
                        elif 'image_url' in get_response_data: # Fallback for other possible structures
                            image_url = get_response_data['image_url']
                        
                        if image_url:
                            console.print(f"[green]Image generation completed. Image URL found.[/green]")
                            break # Exit polling loop
                        else:
                            console.print("[orange]Image generation completed, but no image URL found in response. Retrying...[/orange]")
                    elif status == "FAILED":
                        console.print("[red]Image generation failed.[/red]")
                        break # Exit polling loop
                    
                    time.sleep(5) # Poll every 5 seconds

                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Error querying for image status: {e}[/red]")
                    console.print(f"[red]Stderr: {e.stderr}[/red]")
                    break # Exit polling loop on error
                except json.JSONDecodeError as e:
                    console.print(f"[red]Error parsing JSON response from image status API: {e}[/red]")
                    console.print(f"[red]Raw response: {get_result.stdout}[/red]")
                    break # Exit polling loop on error
                except Exception as e:
                    console.print(f"[red]An unexpected error occurred during image status polling: {e}[/red]")
                    break # Exit polling loop on error
            
            if image_url:
                console.print(f"[blue]Downloading image from: {image_url}[/blue]")
                image_filename = f"generated_image_{task_id}.png" # Make filename dynamic
                try:
                    image_response = requests.get(image_url, stream=True)
                    image_response.raise_for_status() # Raise an exception for HTTP errors
                    with open(image_filename, 'wb') as f:
                        for chunk in image_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    console.print(f"[green]Image saved to {image_filename}[/green]")
                    # Open the image after saving
                    try:
                        subprocess.run(["xdg-open", image_filename], check=True)
                        console.print(f"[green]Opened image: {image_filename}[/green]")
                    except Exception as e:
                        console.print(f"[red]Error opening image: {e}[/red]")
                    
                    # Send image to printer
                    try:
                        subprocess.run(["/usr/bin/lp", image_filename], check=True) # Use full path to lp
                        console.print(f"[green]Sent image to printer: {image_filename}[/green]")
                    except subprocess.CalledProcessError as e:
                        console.print(f"[red]Error sending image to printer: {e}. This might be due to no default printer, printer offline, CUPS issues, or PATH problems.[/red]")
                        console.print(f"[red]Stderr: {e.stderr}[/red]")
                    except Exception as e:
                        console.print(f"[red]An unexpected error occurred while sending image to printer: {e}[/red]")
                except requests.exceptions.RequestException as e:
                    console.print(f"[red]Error downloading image: {e}[/red]")
                except Exception as e:
                    console.print(f"[red]Error saving image: {e}[/red]")
            else:
                console.print("[orange]Image generation timed out or no image URL found after polling.[/orange]")

        else:
            console.print("[orange]No task ID found in the initial API response.[/orange]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error sending initial image generation request: {e}[/red]")
        console.print(f"[red]Stderr: {e.stderr}[/red]")
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON response from initial image generation API: {e}[/red]")
        console.print(f"[red]Raw response: {result.stdout}[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred during image generation process: {e}[/red]")

def on_release(key):
    global recording
    try:
        if key == KEY_TO_PRESS and recording:
            stop_recording()
            # Optionally, restart listening for the key press
            # This allows continuous push-to-talk without restarting the script
            # console.print("[cyan]Ready for next recording... (Press and hold SPACE)[/cyan]")
    except AttributeError:
        pass

def main():
    load_dotenv('config.env') # Load environment variables from config.env
    global model, selected_device_id # Added selected_device_id to global
    console.print(f"[bold blue]Loading Whisper model: {MODEL_SIZE}...[/bold blue]")
    try:
        # Load the Whisper model
        # Using device="cpu" for broader compatibility, can change to "cuda" if GPU is available
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        console.print("[bold green]Model loaded successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading Whisper model: {e}[/bold red]")
        console.print("[bold red]Please ensure the model files are downloaded or specify a valid path.[/bold red]")
        sys.exit(1)

    # List all available input devices
    console.print("\n[bold blue]Available Audio Input Devices:[/bold blue]")
    input_devices = []
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append(device)
                console.print(f"  [cyan]ID: {i}[/cyan], [yellow]Name: {device['name']}[/yellow], [magenta]Host API: {sd.query_hostapis(device['hostapi'])['name']}[/magenta]")
    except Exception as e:
        console.print(f"[red]Error querying devices: {e}[/red]")
        sys.exit(1)

    # selected_device_id is already global, no need to declare it again
    if "SD_INPUT_DEVICE_INDEX" in os.environ:
        try:
            selected_device_id = int(os.environ["SD_INPUT_DEVICE_INDEX"])
            console.print(f"\n[bold green]Using input device specified by SD_INPUT_DEVICE_INDEX:[/bold green] [yellow]{selected_device_id}[/yellow]")
        except ValueError:
            console.print("[red]Invalid value for SD_INPUT_DEVICE_INDEX. Please provide an integer.[/red]")
            sys.exit(1)

    # Print audio device information (either default or selected)
    try:
        if selected_device_id is not None:
            device_info = sd.query_devices(selected_device_id, kind='input')
        else:
            device_info = sd.query_devices(kind='input') # Get default input device

        console.print(f"[bold yellow]Using audio input device:[/bold yellow] [yellow]{device_info['name']}[/yellow] (ID: {device_info['index']})")
        console.print(f"[bold yellow]Supported sample rates:[/bold yellow] [yellow]{device_info['default_samplerate']}[/yellow] (default)")
        
    except Exception as e:
        console.print(f"[red]Could not query audio devices or selected device is invalid: {e}[/red]")
        console.print("[red]Please ensure the selected device ID is valid or check your audio setup.[/red]")
        sys.exit(1)

    console.print("[cyan]Press and hold the SPACE bar to record, release to transcribe.[/cyan]")
    console.print("[cyan]Press Ctrl+C to exit.[/cyan]")

    # Start keyboard listener in a non-blocking way
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Keep the main thread alive
    try:
        listener.join() # This will block until the listener stops (e.g., on Ctrl+C)
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")
        if recording:
            stop_recording() # Ensure recording is stopped on exit
        if stream:
            stream.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
