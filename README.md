# Real-Time Translator 🎙️➡️🇨🇳

A high-performance real-time speech-to-text and translation application built for macOS (Apple Silicon optimized).

## Features
- **⚡️ Real-Time Transcription**: Instant streaming display using `faster-whisper` (or `mlx-whisper`).
- **🌊 Word-by-Word Streaming**: See text appear as you speak, with smart context accumulation.
- **🔄 Async Translation**: Translates text to Chinese (or target language) in the background without blocking the UI.
- **🖥️ Overlay UI**: Always-on-top, transparent, click-through window for seamless usage during meetings/videos.
- **⚙️ Hot Reloading**: Change code or config and the app restarts automatically.
- **💾 Transcript Saving**: One-click save of your session history.

## Demo
https://user-images.githubusercontent.com/6877923/123006036-64e2e780-d3b7-11eb-922e-018994b32da5.mov

## Installation

1. **Prerequisites**:
   - Python 3.10+
   - macOS (recommended for `mlx-whisper` support)
   - `ffmpeg` installed (e.g., `brew install ffmpeg`)
   - `BlackHole` installed (e.g., `brew install blackhole`)
   - `BlackHole` Settings![BlackHole Settings](demo/how_to_set_blackhole.png)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   ```
   *(Ensure you have `PyQt6`, `sounddevice`, `numpy`, `openai`, `watchdog` installed)*

   **🪟 Windows Users**:
   1. Double-click `install_windows.bat` to automatically set up the environment.
   2. Ensure [FFmpeg](https://ffmpeg.org/download.html) is installed and added to your PATH.

## Usage

### 🚀 Start the App
**macOS / Linux**:
Run the installer once:
```bash
./install_mac.sh
```
Then start the app:
```bash
./start_mac.sh
```

**Windows**:
Double-click `start_windows.bat`

*(Or run `python main.py` for a single session)*

### 🎮 Controls
- **Overlay Window**:
  - **Drag**: Click and drag anywhere on the window.
  - **Resize**: Drag the bottom-right corner handles (◢).
  - **Save (💾)**: Save the current transcript to `transcripts/`.
  - **Settings (⚙️)**: Open the configuration window.

### ⚙️ Configuration
### ⚙️ Configuration Reference (`config.ini`)

#### `[api]` Section (Translation Service)
| Parameter | Description | Choices / Examples | Impact |
| :--- | :--- | :--- | :--- |
| `base_url` | API Endpoint URL | `https://api.openai.com/v1` (OpenAI)<br>`http://localhost:11434/v1` (Ollama)<br>`https://api.siliconflow.cn/v1` | Determines which server handles the translation. Localhost is free but requires hardware. |
| `api_key` | Authentication Key | `sk-...` | Required for authentication. Use `dummy` for local servers like Ollama. |
| `model` | LLM for Translation | `gpt-3.5-turbo`, `gpt-4`, `llama3:8b`, `qwen2.5:7b` | **Accuracy vs Speed**. Larger models translate better but may be slower. |
| `target_lang` | Output Language | `Chinese`, `English`, `French`, `Japanese` | The language you want to see translated text in. |

#### `[transcription]` Section (Speech-to-Text)
| Parameter | Description | Choices / Examples | Impact |
| :--- | :--- | :--- | :--- |
| `whisper_model` | Model Size | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` | **Precision vs Speed**. `base`/`small` are fast real-time. `large` is accurate but slow. |
| `device` | Processing Unit | `auto` (Recommended), `cpu`, `cuda` | `auto` uses Apple Neural Engine (Metal) on Mac. `cpu` is slow. |
| `compute_type` | Math Precision | `float16` (Mac/GPU), `int8`, `float32` | `float16` is faster on modern hardware. `int8` saves RAM but may lower quality. |
| `source_language` | Input Audio Language | `auto`, `en`, `zh`, `ja` | `auto` adds a detection delay. Set specific language (e.g., `en`) for faster response. |
| `transcription_workers`| Parallel Threads | `1` to `4` | Number of simultaneous transcriptions. Higher = smoother for fast speakers, but uses more RAM. |

#### `[audio]` Section (Microphone & Streaming)
| Parameter | Description | Choices / Examples | Impact |
| :--- | :--- | :--- | :--- |
| `device_index` | Input Device ID | `auto` (System Default), `0` , `1`... | ID of your mic or Loopback (BlackHole). Check startup logs for IDs. |
| `silence_threshold` | Sensitivity | `0.01` (Loud Env) - `0.005` (Quiet Env) | Lower (`0.005`) picks up whispers but also noise. Higher (`0.05`) needs louder speech. |
| `silence_duration` | VAD Timeout | `0.5` - `2.0` (Seconds) | How long to wait after you stop speaking to "finalize" the sentence. |
| `max_phrase_duration`| Max Sentence Length | `5` - `600` (Seconds) | Forces a split if you speak too long. **Set higher (30s+)** for lectures to avoid cutting sentences. |
| `streaming_step_size`| Audio Grain | `0.1` - `0.5` (Seconds) | **Lower (`0.1`)** = smoother "typing" effect but high CPU. **Higher (`0.5`)** = choppier updates. |
| `update_interval` | UI Refresh Rate | `0.5` - `1.0` (Seconds) | How often the gray "partial" text updates. Should be > `streaming_step_size`. |
| `streaming_mode` | Continuous Stream | `true` / `false` | `true`: Emits audio constantly (good for fast talkers). `false`: Strict VAD only. |
| `streaming_interval` | Stream Segment | `1.5` - `3.0` (Seconds) | Only used if `streaming_mode=true`. How often to force a chunk emit. |
| `streaming_overlap` | Context Overlap | `0.3` (Seconds) | Overlap between chunks to prevent cutting words at boundaries. |

#### `[display]` Section (Appearance)
| Parameter | Description | Choices / Examples | Impact |
| :--- | :--- | :--- | :--- |
| `window_width` | Overlay Width | `800`, `1024` (Pixels) | Width of the transparent window. |
| `window_height` | Overlay Height | `120`, `200` (Pixels) | Height of the visible text area. |
| `display_duration`| Persistence | `10` (Seconds) | (Legacy) How long text stays before clearing (if not using scroll history). |

## Troubleshooting
- **No Audio?** Check the terminal for "Audio Capture" logs. If using BlackHole, ensure it's selected in `config.ini` or auto-detected.
- **Resize not working?** Use the designated "◢" handle in the bottom-right.
- **Hot Reload**: Modify any `.py` file or save settings in the UI to trigger a reload.


## License: MIT
Copyright 2025 Van

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
