# Real-Time Translator: Requirements Specification

## 1. System Overview
A high-performance real-time speech-to-text and translation application for macOS (Apple Silicon optimized) and Windows. The system captures system/microphone audio, transcribes it using local Whisper models, translates it using an OpenAI-compatible API, and displays the result in a transparent overlay window.

## 2. AI & Communication Specification
This section defines strictly how the application interacts with AI models.

### 2.1 Transcription (Local AI)
- **Engine**: `faster-whisper` (CPU/CUDA) or `mlx-whisper` (Apple Silicon Metal).
- **Input**: Raw float32 audio chunks (16kHz).
- **Context Handling**:
  - The transcriber must support an `initial_prompt` argument.
  - The application must feed the **text of the previous final sentence** as the `initial_prompt` for the current audio chunk.
  - **Purpose**: To maintain capitalization and context across sentence breaks (e.g., prevent "hello world" -> "Hello. World.").
- **Hallucination Filter**:
  - Must detect and suppress "infinite loop" hallucinations common in Whisper (e.g., "Thanks. Thanks. Thanks.").
  - **Logic**:
    - Reject if >4 consecutive identical words.
    - Reject if unique word ratio < 0.4 (for segments >10 words).
- **Warmup**:
  - Upon initialization, the system must perform a "warmup" inference (1s silence) to trigger JIT compilation/model loading before the UI is shown.

### 2.2 Translation (Remote AI)
- **Engine**: OpenAI-compatible Conversation API (`/v1/chat/completions`).
- **Prompt Engineering**:
  - **System Prompt** (Contextual):
    > "You are a professional real-time translator. Translate the following user input into [Target Language]. The previous audio segment was: "[Previous Original Text]" (translated as: "[Previous Translated Text]"). Consider this context for continuity, but ONLY translate the new text below. Do not add any explanations, just output the translation of the new text."
  - **System Prompt** (No Context):
    > "You are a professional real-time translator. Translate the following user input into [Target Language]. Do not add any explanations, just output the translation."
- **Reasoning Models**:
  - The system must verify if the response contains `<think>...</think>` tags (common in reasoning models).
  - **Requirement**: Use Regex to strip all content within `<think>` tags before displaying.

## 3. Functional Requirements

### 3.1 Audio Capture & VAD
- **Input Source**: Configurable device index or "auto" (detects BlackHole on macOS).
- **Voice Activity Detection (VAD)**:
  - **RMS Threshold**: configurable (default 0.005).
  - **Silence Duration**: configurable (default 0.5s - 2.0s).
  - **Max Phrase Duration**: configurable (default 5.0s).
- **Streaming Logic**:
  - Must yield small raw chunks (0.2s) continuously.
  - **Accumulation**: The main pipeline accumulates these chunks into a growing buffer.
  - **Partial Update**: Every `update_interval` (0.8s), runs a partial transcription on the growing buffer and updates the UI (Gray text).
  - **Finalization**:
    - If `silence_duration` passed AND buffer > 2.0s: Finalize.
    - OR if buffer > `max_phrase_duration`: Force finalize.
    - On finalize: Clear buffer, promote text to "Final" (White), trigger translation.

### 3.2 Overlay User Interface
- **Window**: Frameless, transparent background, always-on-top.
- **MacOS Integration**: Must use `NSWindowCollectionBehaviorCanJoinAllSpaces` to appear on all Mission Control spaces.
- **Content**:
  - **Scrollable Log**: Displays history of segments.
  - **Segment Widget**:
    - **Original**: Small, Gray (#aaaaaa), Timestamped.
    - **Translated**: Large, White (#ffffff), Bold.
- **Controls**:
  - **Resize**: Custom `ResizeHandle` widget (◢) in bottom-right.
  - **Save**: Button to dump current session to `transcripts/*.txt`.
  - **Settings**: Button to open configuration.

### 3.3 Configuration Management
- **Persistence**: `config.ini` file.
- **Hot Reload**:
  - A watcher (`reloader.py`) monitors `.py` and `.ini` files.
  - Must restart `main.py` automatically upon changes.
- **Settings UI**:
  - API Key & Base URL input.
  - Dynamic Model List: Fetch available models from `base_url` via button.
  - Whisper Model size selection.
  - VAD/Streaming parameter tuning.

### 3.4 Cross-Platform Installers
- **Windows**:
  - `install_windows.bat`: Create venv, install reqs.
  - `start_windows.bat`: Launch reloader.
- **MacOS**:
  - `install_mac.sh`: Create venv, install reqs.
  - **Auto-Optimization**: Detect `arm64` arch and install `mlx-whisper` automatically.
  - **Dependency Check**: Warn if `ffmpeg` is missing.
  - `start_mac.sh`: Launch reloader.

## 4. Boundaries & Limitations
- **No Speaker Diarization**: The system treats all audio as a single stream; it does not distinguish between speakers.
- **No Translation Audio**: The system outputs text only; it does not perform Text-to-Speech (TTS).
- **Single Target Language**: Only one target language is active at a time.

## 5. Test Cases

### 5.1 First Run Experience
1.  **Action**: Run start script.
2.  **Verify**: Terminal shows "Warming up model...".
3.  **Verify**: Log shows "Warmup complete".
4.  **Verify**: Overlay window appears empty.

### 5.2 Contextual Translation
1.  **Action**: Speak a sentence split in two (e.g., "Hello," ...pause... "world.").
2.  **Verify**: 
    - Segment 1: "Hello," (Translated).
    - Segment 2: "world." (Translated).
    - **Check**: Segment 2 transcription should NOT be capitalized ("World.") if context worked.
    - **Check**: Translation of Segment 2 should make sense in context of Segment 1.

### 5.3 Long Speech Handling
1.  **Action**: Speak continuously for 10 seconds without stopping.
2.  **Verify**: System forces a split at `max_phrase_duration` (5s).
3.  **Verify**: No audio is lost; second half appears immediately as new segment.

### 5.4 Hot Reload
1.  **Action**: While app is running, edit `config.ini` (e.g., change `window_width` to 500).
2.  **Verify**: App closes and reopens automatically with new width.
