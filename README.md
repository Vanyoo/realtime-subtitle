# Real-Time Translator 🎙️➡️🇨🇳

A high-performance real-time speech-to-text and translation application built for macOS (Apple Silicon optimized).

## Features
- **⚡️ Real-Time Transcription**: Instant streaming display using `faster-whisper` (or `mlx-whisper`).
- **🌊 Word-by-Word Streaming**: See text appear as you speak, with smart context accumulation.
- **🔄 Async Translation**: Translates text to Chinese (or target language) in the background without blocking the UI.
- **🖥️ Overlay UI**: Always-on-top, transparent, click-through window for seamless usage during meetings/videos.
- **⚙️ Hot Reloading**: Change code or config and the app restarts automatically.
- **💾 Transcript Saving**: One-click save of your session history.

## Installation

1. **Prerequisites**:
   - Python 3.10+
   - macOS (recommended for `mlx-whisper` support)
   - `ffmpeg` installed (e.g., `brew install ffmpeg`)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `PyQt6`, `sounddevice`, `numpy`, `openai`, `watchdog` installed)*

## Usage

### 🚀 Start the App
Run the reloader script to enable hot-reloading:
```bash
python reloader.py
```
*(Or run `python main.py` for a single session)*

### 🎮 Controls
- **Overlay Window**:
  - **Drag**: Click and drag anywhere on the window.
  - **Resize**: Drag the bottom-right corner handles (◢).
  - **Save (💾)**: Save the current transcript to `transcripts/`.
  - **Settings (⚙️)**: Open the configuration window.

### ⚙️ Configuration
You can configure the app in two ways:

1.  **GUI**: Click the **⚙️** button in the overlay to change:
    - OpenAI API Key & URL
    - Transcription Model (tiny/base/small/medium/large)
    - Streaming Speed (Latency)

2.  **File**: Edit `config.ini` directly.
    - `streaming_step_size`: Audio chunk size in seconds (default 0.2).
    - `update_interval`: How often UI updates (default 0.5s).

## Troubleshooting
- **No Audio?** Check the terminal for "Audio Capture" logs. If using BlackHole, ensure it's selected in `config.ini` or auto-detected.
- **Resize not working?** Use the designated "◢" handle in the bottom-right.
- **Hot Reload**: Modify any `.py` file or save settings in the UI to trigger a reload.
