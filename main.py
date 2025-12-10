import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import signal
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from audio_capture import AudioCapture
from transcriber import Transcriber
from translator import Translator
from overlay_window import OverlayWindow
from config import config

class WorkerSignals(QObject):
    update_text = pyqtSignal(int, str, str)  # (chunk_id, original, translated)

class Pipeline(QObject):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.running = True
        
        # Print config for debugging
        config.print_config()
        
        # Initialize components
        self.audio = AudioCapture(
            device_index=config.device_index,
            sample_rate=config.sample_rate,
            silence_threshold=config.silence_threshold,
            silence_duration=config.silence_duration,
            chunk_duration=config.chunk_duration,
            max_phrase_duration=config.max_phrase_duration,
            streaming_mode=config.streaming_mode,
            streaming_interval=config.streaming_interval,
            streaming_step_size=config.streaming_step_size,
            streaming_overlap=config.streaming_overlap
        )
        
        # Initialize Transcriber
        print(f"[Pipeline] Initializing Transcriber with device={config.whisper_device}...")
        self.transcriber = Transcriber(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            language=config.source_language
        )
        
        # Initialize Translator
        print(f"[Pipeline] Initializing Translator (target={config.target_lang})...")
        self.translator = Translator(
            target_lang=config.target_lang,
            base_url=config.api_base_url,
            api_key=config.api_key,
            model=config.model
        )
        
        # Warmup Transcriber (Critical for MLX/GPU)
        self.transcriber.warmup()

    def start(self):
        """Start the processing pipeline in a dedicated thread"""
        self.audio.start()
        self.thread = threading.Thread(target=self.processing_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        print("\n[Pipeline] Stopping...")
        self.running = False
        self.audio.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        print("[Pipeline] Stopped.")

    def processing_loop(self):
        """Fully parallel pipeline: multiple concurrent transcription + translation"""
        print("Pipeline processing loop started (FULLY PARALLEL mode).")
        
        # Create multiple transcribers for concurrent processing
        # CHECK: If using MLX, force 1 worker (MLX is not thread-safe for parallel inference in this way)
        is_mlx = getattr(self.transcriber, 'use_mlx', False)
        
        if is_mlx:
            print("[Pipeline] MLX backend detected - forcing single worker (MLX uses GPU parallelism internaly)")
            num_transcription_workers = 1
        else:
            num_transcription_workers = config.transcription_workers
            
        print(f"[Pipeline] Using {num_transcription_workers} transcription workers...")
        
        transcribers = [self.transcriber]  # Reuse existing one
        for i in range(num_transcription_workers - 1):
            t = Transcriber(
                model_size=config.whisper_model,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
                language=config.source_language
            )
            transcribers.append(t)
        """Accumulating Buffer Processing Loop (Word-by-Word Streaming)"""
        print("[Pipeline] processing loop started (Accumulating Mode).")
        
        import numpy as np
        
        # Executors
        transcribe_executor = ThreadPoolExecutor(max_workers=1) # Serial transcription
        translate_executor = ThreadPoolExecutor(max_workers=2)
        
        # State
        buffer = np.array([], dtype=np.float32)
        chunk_id = 1
        last_update_time = time.time()
        phrase_start_time = time.time()
        
        # Generator yielding small chunks (e.g. 0.2s)
        audio_gen = self.audio.generator()
        
        # Context Management
        self.last_final_text = ""

        try:
            for audio_chunk in audio_gen:
                if not self.running:
                    break
                buffer = np.concatenate([buffer, audio_chunk])
                now = time.time()
                buffer_duration = len(buffer) / self.audio.sample_rate
                
                # Check silence for finalization
                # Use configured silence duration/threshold
                is_silence = False
                min_silence_dur = config.silence_duration # e.g. 1.0s
                
                # Only check silence if we have enough buffer
                if buffer_duration > min_silence_dur:
                     # Check tail of silence duration
                    tail = buffer[-int(self.audio.sample_rate * min_silence_dur):]
                    rms = np.sqrt(np.mean(tail**2))
                    if rms < self.audio.silence_threshold:
                        is_silence = True
                
                # DECISION: Finalize vs Partial Update
                
                # 1. Finalize if: (Long enough AND silence) OR Max duration reached
                # Require at least 2.0s duration if using silence test (prevent chopping short phrases)
                should_finalize = (
                    (is_silence and buffer_duration > 2.0) or 
                    (buffer_duration > self.audio.max_phrase_duration)
                )
                
                if should_finalize and buffer_duration > 0.5:
                    # FINALIZE
                    final_buffer = buffer.copy()
                    cid = chunk_id
                    
                    # Store current prompt to pass to task (thread safety)
                    prompt = self.last_final_text
                    
                    # Submit Final Task
                    transcribe_executor.submit(self._process_final_chunk, final_buffer, cid, prompt)
                    
                    # Reset
                    buffer = np.array([], dtype=np.float32)
                    chunk_id += 1
                    phrase_start_time = now
                    last_update_time = now
                    
                # 2. Partial Update if: Interval passed AND not finalizing
                elif now - last_update_time > config.update_interval and buffer_duration > 0.5:
                    # PARTIAL UPDATE
                    partial_buffer = buffer.copy()
                    prompt = self.last_final_text
                    transcribe_executor.submit(self._process_partial_chunk, partial_buffer, chunk_id, prompt)
                    last_update_time = now
                    
        except Exception as e:
            print(f"[Pipeline] Error in loop: {e}")
        finally:
            transcribe_executor.shutdown(wait=False)
            translate_executor.shutdown(wait=False)

    def _process_partial_chunk(self, audio_data, chunk_id, prompt=""):
        """Transcribe and update UI (No translation)"""
        try:
            # Use accumulated context as prompt
            text = self.transcriber.transcribe(audio_data, prompt=prompt)
            if text:
                self.signals.update_text.emit(chunk_id, text, "")
        except Exception as e:
            print(f"[Partial {chunk_id}] Error: {e}")

    def _process_final_chunk(self, audio_data, chunk_id, prompt=""):
        """Transcribe, Log, and Trigger Translation"""
        try:
            text = self.transcriber.transcribe(audio_data, prompt=prompt)
            if text:
                print(f"[Final {chunk_id}] Transcribed: {text}")
                # Save for context (only if meaningful)
                if len(text.split()) > 2:
                    self.last_final_text = text
                
                # Emit final transcription first (confirms text)
                self.signals.update_text.emit(chunk_id, text, "")
                
                # Translate
                translated = self.translator.translate(text)
                print(f"[Final {chunk_id}] Translated: {translated}")
                self.signals.update_text.emit(chunk_id, text, translated)
            else:
                pass
        except Exception as e:
            print(f"[Final {chunk_id}] Error: {e}")
    
    def _transcribe_chunk(self, transcriber, audio_chunk, chunk_id):
        """Transcribe a single chunk and log timing"""
        t0 = time.time()
        text = transcriber.transcribe(audio_chunk)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Transcribed in {t1-t0:.2f}s: {text if text else '(empty)'}")
        return text
    
    def _translate_and_log(self, text, chunk_id=0):
        """Translate text and log result"""
        t0 = time.time()
        translated_text = self.translator.translate(text)
        t1 = time.time()
        print(f"[Chunk {chunk_id}] Translated in {t1-t0:.2f}s: {translated_text}")
        return (text, translated_text)

# Global reference for signal handler
_pipeline = None
_app = None

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully"""
    print("\n[Main] Ctrl-C received, force killing...")
    os._exit(0)

def main():
    global _pipeline, _app
    
    # Set up signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    _app = QApplication(sys.argv)
    
    # Initialize Overlay Window
    window = OverlayWindow(
        display_duration=config.display_duration,
        window_width=config.window_width
        # window_height is omitted to let it default to 100% screen height
    )
    window.show()
    
    # Logic
    _pipeline = Pipeline()
    
    # Connect signals
    _pipeline.signals.update_text.connect(window.update_text)
    
    # Timer to let Python interpreter handle signals (Ctrl-C)
    timer = QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)
    
    # Start pipeline
    _pipeline.start()
    
    try:
        sys.exit(_app.exec())
    except SystemExit:
        pass
    finally:
        _pipeline.stop()

if __name__ == "__main__":
    main()
