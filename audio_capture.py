import sounddevice as sd
import numpy as np
import queue
import threading
import time

class AudioCapture:
    def __init__(self, device_index=None, sample_rate=16000, chunk_duration=0.1, 
                 silence_threshold=0.01, silence_duration=1.0, max_phrase_duration=5.0,
                 streaming_mode=False, streaming_interval=1.5, streaming_step_size=0.2, streaming_overlap=0.3):
        """
        Captures audio and yields segments containing speech.
        
        Args:
            device_index: Index of input device (None for default).
            sample_rate: Audio sample rate (default 16000 for Whisper).
            chunk_duration: Duration of each small read in seconds.
            silence_threshold: RMS amplitude threshold for "silence".
            silence_duration: How many seconds of silence triggers a segment cut.
            max_phrase_duration: Force processing after this many seconds even without silence.
            streaming_mode: If True, emit audio at fixed intervals (no VAD).
            streaming_interval: Seconds between audio emissions in streaming mode.
            streaming_overlap: Seconds of overlap between chunks in streaming mode.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * chunk_duration)
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.max_phrase_duration = max_phrase_duration
        
        # Streaming mode settings
        self.streaming_mode = streaming_mode
        self.streaming_interval = streaming_interval
        self.streaming_mode = streaming_mode
        self.streaming_interval = streaming_interval
        self.streaming_step_size = streaming_step_size
        self.streaming_overlap = streaming_overlap
        
        self.audio_queue = queue.Queue()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        
        # Log audio device info
        print("[AudioCapture] Starting...")
        print(f"  Sample Rate: {self.sample_rate} Hz")
        print(f"  Silence Threshold: {self.silence_threshold}")
        print(f"  Silence Duration: {self.silence_duration}s")
        
        # Get device info
        if self.device_index is None:
            default_device = sd.query_devices(kind='input')
            print(f"  Using DEFAULT input device:")
            print(f"    Name: {default_device['name']}")
            print(f"    Index: {default_device['index']}")
            print(f"    Channels: {default_device['max_input_channels']}")
        else:
            device_info = sd.query_devices(self.device_index)
            print(f"  Using device index {self.device_index}:")
            print(f"    Name: {device_info['name']}")
            print(f"    Channels: {device_info['max_input_channels']}")
        
        print("\n[AudioCapture] Available input devices:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                marker = " <-- SELECTED" if (self.device_index == i or (self.device_index is None and d == sd.query_devices(kind='input'))) else ""
                print(f"    [{i}] {d['name']} ({d['max_input_channels']} ch){marker}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        print("Audio capture stopped.")

    def generator(self):
        """Yields small raw audio chunks for external accumulation logic."""
        if self.device_index is None:
             # Just safety check, usually handled in start()
             pass
             
        # Use configured step size
        block_size = int(self.sample_rate * self.streaming_step_size)
        print(f"[Audio] Starting raw processing stream (step={self.streaming_step_size}s)")
        
        with sd.InputStream(device=self.device_index, channels=1, samplerate=self.sample_rate, 
                            blocksize=block_size, dtype='float32') as stream:
             self.running = True
             while self.running:
                 data, overflow = stream.read(block_size)
                 if overflow:
                     print("Audio overflow")
                 yield data.flatten()
        print("[Audio] Generator stopped.")

    def get_audio_stream(self):
        """Generator that yields numpy arrays of float32 audio containing speech."""
        while self.running:
            try:
                # Get a segment from the queue
                audio_segment = self.audio_queue.get(timeout=1)
                yield audio_segment
            except queue.Empty:
                continue

    def _record_loop(self):
        if self.streaming_mode:
            self._streaming_record_loop()
        else:
            self._vad_record_loop()
    
    def _streaming_record_loop(self):
        """Continuous streaming: emit audio every streaming_interval seconds with overlap"""
        print(f"[Audio] Streaming mode: interval={self.streaming_interval}s, overlap={self.streaming_overlap}s")
        
        interval_samples = int(self.sample_rate * self.streaming_interval)
        overlap_samples = int(self.sample_rate * self.streaming_overlap)
        
        # Ring buffer to hold audio with overlap
        buffer = np.array([], dtype=np.float32)
        
        with sd.InputStream(device=self.device_index, channels=1, samplerate=self.sample_rate,
                            blocksize=self.block_size, dtype='float32') as stream:
            
            last_emit_time = time.time()
            
            while self.running:
                audio_chunk, _ = stream.read(self.block_size)
                audio_chunk = audio_chunk.flatten()
                buffer = np.concatenate([buffer, audio_chunk])
                
                # Check if it's time to emit
                if time.time() - last_emit_time >= self.streaming_interval:
                    if len(buffer) > 0:
                        # Check if there's any audio (not pure silence)
                        rms = np.sqrt(np.mean(buffer**2))
                        if rms > self.silence_threshold * 0.5:  # Lower threshold for streaming
                            duration = len(buffer) / self.sample_rate
                            print(f"[Audio] Streaming emit: {duration:.2f}s, RMS={rms:.4f}")
                            self.audio_queue.put(buffer.copy())
                        
                        # Keep overlap for context, discard the rest
                        if len(buffer) > overlap_samples:
                            buffer = buffer[-overlap_samples:]
                        
                    last_emit_time = time.time()
    
    def _vad_record_loop(self):
        """VAD-based recording: wait for speech and silence"""
        # Buffer to hold current speech phrase
        current_phrase = []
        silence_start_time = None
        has_speech = False
        
        def callback(indata, frames, time_info, status):
            if status:
                print(status)
            if not self.running:
                raise sd.CallbackAbort
            
            # Make a copy of data
            audio_chunk = indata.copy().flatten()
            
            # Simple VAD: Check RMS (Root Mean Square) amplitude
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            # Communicate via non-local variables (or just process here)
            # Since callback is in a separate thread managed by sounddevice, 
            # we need to be careful. However, pure python processing in callback 
            # might block if too slow. 
            # Better strategy: push raw chunks to a processing queue, 
            # but for simplicity, let's use a shared list with lock or just a queue for raw chunks.
            pass

        # To avoid complexity with callbacks, let's use a blocking read in this thread
        # sounddevice.InputStream logic
        
        with sd.InputStream(device=self.device_index, channels=1, samplerate=self.sample_rate, 
                            blocksize=self.block_size, callback=None, dtype='float32') as stream:
            
            debug_counter = 0
            max_rms_seen = 0
            phrase_start_time = None  # Track when current phrase started
            
            while self.running:
                audio_chunk, overflow = stream.read(self.block_size)
                audio_chunk = audio_chunk.flatten()
                
                rms = np.sqrt(np.mean(audio_chunk**2))
                max_rms_seen = max(max_rms_seen, rms)
                
                # Debug logging every 2 seconds
                debug_counter += 1
                if debug_counter % 20 == 0:
                    status = "SPEECH" if has_speech else "silent"
                    phrase_dur = time.time() - phrase_start_time if phrase_start_time else 0
                    print(f"[Audio] RMS: {rms:.4f} | Max: {max_rms_seen:.4f} | Threshold: {self.silence_threshold} | {status} | Phrase: {phrase_dur:.1f}s")
                
                # Always collect audio if above threshold
                if rms > self.silence_threshold:
                    if not has_speech:
                        has_speech = True
                        phrase_start_time = time.time()
                        print(f"[Audio] Speech detected! RMS={rms:.4f}")
                    current_phrase.append(audio_chunk)
                    silence_start_time = None
                else:
                    if has_speech:
                        current_phrase.append(audio_chunk)
                        
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > self.silence_duration:
                            # Silence long enough, cut phrase
                            self._emit_phrase(current_phrase, "silence")
                            current_phrase = []
                            has_speech = False
                            silence_start_time = None
                            phrase_start_time = None
                
                # Force cut if phrase is too long (real-time requirement)
                if has_speech and phrase_start_time:
                    phrase_duration = time.time() - phrase_start_time
                    if phrase_duration >= self.max_phrase_duration:
                        self._emit_phrase(current_phrase, "max_time")
                        current_phrase = []
                        has_speech = False
                        silence_start_time = None
                        phrase_start_time = None
    
    def _emit_phrase(self, phrase_chunks, reason):
        """Helper to emit a complete phrase"""
        if not phrase_chunks:
            return
        full_phrase = np.concatenate(phrase_chunks)
        duration = len(full_phrase) / self.sample_rate
        print(f"[Audio] Phrase complete ({reason}): {duration:.2f}s")
        self.audio_queue.put(full_phrase)

if __name__ == "__main__":
    # Test
    print("Available devices:")
    print(sd.query_devices())
    
    cap = AudioCapture()
    cap.start()
    try:
        for i, segment in enumerate(cap.get_audio_stream()):
            print(f"Got audio segment {i}: length {len(segment)/16000:.2f}s")
    except KeyboardInterrupt:
        cap.stop()
