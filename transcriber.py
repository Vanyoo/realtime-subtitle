import numpy as np

class Transcriber:
    def __init__(self, model_size, device, compute_type, language=None):
        self.language = language
        self.use_mlx = False
        self.model_size = model_size
        
        try:
            import mlx_whisper
            self.use_mlx = True
            print(f"[Transcriber] Using MLX Whisper (Metal Acceleration) with model: {model_size}")
        except ImportError:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"[Transcriber] Using faster-whisper (CPU/CUDA) with model: {model_size}")

    def transcribe(self, audio_data, prompt=None):
        if self.use_mlx:
            text = self._transcribe_mlx(audio_data, prompt)
        else:
            text = self._transcribe_faster_whisper(audio_data, prompt)
            
            
        # Filter hallucinations (infinite loops, e.g. "once once once")
        if self._is_hallucination(text):
            print(f"[Transcriber] Filtered hallucination: {text[:50]}...")
            return ""

        # Filter prompt echoes (music/noise causing repetition of context)
        if prompt and self._is_prompt_echo(text, prompt):
            print(f"[Transcriber] Filtered prompt echo: {text[:50]}...")
            return ""
            
        return text

    def warmup(self):
        """Warmup the model to prevent lag on first inference"""
        print("[Transcriber] Warming up model...")
        # 1 second of silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            self.transcribe(dummy_audio)
            print("[Transcriber] Warmup complete.")
        except Exception as e:
            print(f"[Transcriber] Warmup failed (non-fatal): {e}")

    def _is_hallucination(self, text):
        """Check if text looks like a Whisper hallucination (repetitive loop)"""
        if not text:
            return False
            
        words = text.split()
        if not words:
            return False
            
        # 1. Check for immediate consecutive repetitions of the same word
        # e.g. "once once once once once"
        max_repeats = 0
        current_repeats = 1
        last_word = ""
        
        for word in words:
            if word == last_word:
                current_repeats += 1
            else:
                max_repeats = max(max_repeats, current_repeats)
                current_repeats = 1
                last_word = word
        max_repeats = max(max_repeats, current_repeats)
        
        if max_repeats > 4:
            return True
            
        # 2. Check for low information density (unique words / total words)
        # e.g. "that was that was that was that was"
        if len(words) > 10:
            unique_words = set(words)
            ratio = len(unique_words) / len(words)
            if ratio < 0.4: # Filter if less than 40% of words are unique
                return True
                
        return False

    def _is_prompt_echo(self, text, prompt):
        """Check if the transcribed text is just an echo of the prompt (common hallucination on silence/music)"""
        if not text or not prompt:
            return False
            
        import re
        def normalize(s):
            return re.sub(r'[^\w\s]', '', s.lower()).strip()
            
        norm_text = normalize(text)
        norm_prompt = normalize(prompt)
        
        if not norm_text or not norm_prompt:
            return False
            
        # Check for exact match or strong overlap
        if norm_text == norm_prompt:
            return True
            
        # Check if text is a trailing substring of prompt (e.g. Prompt="Hello world", Text="world")
        if norm_prompt.endswith(norm_text):
            return True
            
        return False

    def _transcribe_mlx(self, audio_data, prompt=None):
        import mlx_whisper
        # mlx_whisper.transcribe takes audio and other kwargs
        # We need to ensure audio_data is in the format MLX expects (usually numpy array)
        
        try:
            # Prepare kwargs
            kwargs = {
                "path_or_hf_repo": f"mlx-community/whisper-{self.model_size}-mlx",
                "language": self.language,
                "temperature": 0.0
            }
            if prompt:
                kwargs["initial_prompt"] = prompt
                
            result = mlx_whisper.transcribe(audio_data, **kwargs)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[Transcriber] MLX Error: {e}")
            return ""

    def _transcribe_faster_whisper(self, audio_data, prompt=None):
        segments, _ = self.model.transcribe(
            audio_data, 
            language=self.language, 
            beam_size=5,
            condition_on_previous_text=False, # We manage context manually if needed
            initial_prompt=prompt,
            no_speech_threshold=0.6
        )
        text = " ".join([segment.text for segment in segments]).strip()
        return text
