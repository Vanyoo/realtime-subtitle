# Real-Time Translator 🎙️➡️🇨🇳

A high-performance real-time speech-to-text and translation application built for macOS (Apple Silicon optimized).

## Features
- **⚡️ Real-Time Transcription**: Instant streaming display using `faster-whisper` (or `mlx-whisper`).
- **🌊 Word-by-Word Streaming**: See text appear as you speak, with smart context accumulation.
- **🔄 Async Translation**: Translates text to Chinese (or target language) in the background without blocking the UI.
- **🖥️ Overlay UI**: Always-on-top, transparent, click-through window for seamless usage during meetings/videos.
- **⚙️ Hot Reloading**: Change code or config and the app restarts automatically.
- **💾 Transcript Saving**: One-click save of your session history. Can be used as subtitle or LLM analyze.

## Demo
https://github.com/Vanyoo/realtime-subtitle/raw/refs/heads/master/demo/demo%20screenshot.mp4

![Dashboard](./demo/main_dashboard.png)

## Installation

1. **Prerequisites**:
   - Python 3.10+
   - macOS (recommended for `mlx-whisper` support)
   - `ffmpeg` installed (e.g., `brew install ffmpeg`)
   - `BlackHole` installed (e.g., `brew install blackhole-2ch`, need to enter system password)
   - `BlackHole` Settings![BlackHole Settings](demo/how_to_set_blackhole.png)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   *(Ensure you have `PyQt6`, `sounddevice`, `numpy`, `openai`, `watchdog` installed)*

   **🪟 Windows Users**:
   1. Double-click `install_windows.bat` to automatically set up the environment.
   2. Ensure [FFmpeg](https://ffmpeg.org/download.html) is installed and added to your PATH.

   **🖥 MacOS Users**:
   1. Use terminal to run `install_mac.sh`

## ✨ New Features & Quick Start
- **Modern Control Center**: Manage all settings in a dark-themed Dashboard.
- **One-Click Launch**: Start the overlay translator directly from the Dashboard.
- **Auto-Dependency Check**: Automatically installs missing requirements.
- **Audio Device Selection**: Choose your specific microphone input.

## Usage

### 1. Start the Application
Run the helper script for your OS:
- **Mac/Linux**: `./start_mac.sh`
- **Windows**: `start_windows.bat`

### 2. The Dashboard
The application opens the **Real-Time Translator Control Center**.
- **Home**: Click **"▶ Launch Translator"** to start the overlay.
- **Audio**: Select your Input Device and adjust Silence Threshold.
  * <details>
     <summary>How to Set</summary>
     1. Audio MIDI Setup: create multiple devices, including `BlackHole 2ch` device, and if you want to listen too, remember adding system output device

     ![](./demo/Audio_MIDI_Setup.png)

     2. Choose target audio device to capture

     ![](./demo/Audio_configuraiton.png)
   </details>
- **Transcription**: Choose Whisper model size (tiny, base, small, medium, large-v3, [see the difference](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)).
  * <details>
     <summary>How to Set</summary>
     
     * MacOS
       * Whisper Model: base
       * Compute Device: audo
       * Quantization: float16
   </details>
- **Translation**: Set your OpenAI API Key and Target Language.
- **Save Settings**: Click "Save Settings" to persist your configuration.

### 3. The Overlay
Once launched, a transparent window appears:
- **Move**: Click and drag text to move.
- **Resize**: Drag the bottom-right handle (◢).
- **Stop**: Click **"⏹"** on the overlay or "Stop Translator" in the Dashboard.
- **Save**: Click **"💾 Save"** to export transcript.

## ⚙️ Configuration Reference
Settings are managed via the Dashboard, but stored in `config.ini`.

#### `[api]` Section
| Parameter | Description | Examples |
| :--- | :--- | :--- |
| `base_url` | API Endpoint | `https://api.openai.com/v1`, `http://localhost:11434/v1` |
| `api_key` | Auth Key | `sk-...` (or `dummy` for local) |
| `target_lang` | Output Language | `Chinese`, `English`, `Japanese` |

#### `[transcription]` Section
| Parameter | Description | Details |
| :--- | :--- | :--- |
| `whisper_model` | Model Size | `tiny` (fast), `large-v3` (accurate) |
| `device` | Compute Unit | `auto` (Apple Neural Engine), `cuda` (NVIDIA) |

#### `[audio]` Section
| Parameter | Description | Details |
| :--- | :--- | :--- |
| `silence_threshold`| Sensitivity | `0.005` (Quiet) to `0.05` (Loud) |
| `device_index` | Mic ID | `auto` or specific index `0`, `1`... |

## Troubleshooting
- **No Audio?** Check the terminal for "Audio Capture" logs. If using BlackHole, ensure it's selected in `config.ini` or auto-detected.
- **Resize not working?** Use the designated "◢" handle in the bottom-right.
- **Hot Reload**: Modify any `.py` file or save settings in the UI to trigger a reload.


## License: MIT
Copyright 2025 Van

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
