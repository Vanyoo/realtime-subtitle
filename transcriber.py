import numpy as np

class Transcriber:
    def __init__(self, backend="whisper", model_size="base", device="cpu", compute_type="int8", language=None):
        """
        Initialize Transcriber with multiple backend support
        
        Args:
            backend: ASR backend to use - 'whisper', 'mlx', or 'funasr'
            model_size: Model identifier (for Whisper: tiny/base/small/medium/large/turbo, for FunASR: model name)
            device: Device to use (cpu/cuda/auto)
            compute_type: Compute type for faster-whisper (int8/float16/float32)
            language: Source language code or None for auto-detect
        """
        self.backend = backend.lower()
        self.language = language
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        if self.backend == "funasr":
            self._init_funasr(model_size, device)
        elif self.backend == "mlx":
            self._init_mlx(model_size)
        else:  # default to whisper
            self._init_whisper(model_size, device, compute_type)

    def _init_whisper(self, model_size, device, compute_type):
        """Initialize faster-whisper backend"""
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"[Transcriber] Using faster-whisper (CPU/CUDA) with model: {model_size}")
    
    def _init_mlx(self, model_size):
        try:
            import mlx_whisper
            # MLX doesn't need explicit model loading here
            print(f"[Transcriber] Using MLX Whisper (Metal Acceleration) with model: {model_size}")
        except ImportError:
            print("[Transcriber] Warning: mlx_whisper not available, falling back to faster-whisper")
            self.backend = "whisper"
            self._init_whisper(model_size, "cpu", "int8")
    
    def _init_funasr(self, model_size, device):
        """Initialize FunASR backend with device support"""
        try:
            from funasr import AutoModel
            import platform
            import torch
            
            print(f"[Transcriber] Initializing FunASR with model: {model_size}")
            
            # Determine the optimal device for FunASR
            funasr_device = self._get_funasr_device(device)
            print(f"[Transcriber] FunASR using device: {funasr_device}")
            
            # Store device for later use in transcription
            self.funasr_device = funasr_device
            
            # For MPS device, monkey patch torch.tensor to force float32
            if funasr_device == "mps":
                print("[Transcriber] Applying MPS float32 compatibility patches")
                self._apply_mps_float32_patches()
            
            # Initialize FunASR model with device parameter
            # FunASR supports: 'cuda', 'cpu', or 'mps' (with float32)
            model_kwargs = {
                "model": model_size,
                "device": funasr_device,
                "disable_pbar": True,
                "disable_log": False
            }
            
            # For MPS device, we need to ensure float32 dtype
            # This mimics the fp16 parameter in FunASR but for float32
            if funasr_device == "mps":
                print("[Transcriber] Configuring FunASR for MPS with float32 dtype")
                # Set default dtype to float32 for MPS compatibility
                old_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
            
            self.model = AutoModel(**model_kwargs)
            
            # CRITICAL: For MPS, explicitly convert ALL model components to float32 after initialization
            # This is necessary because FunASR's internal operations may still use float64
            if funasr_device == "mps":
                print("[Transcriber] Converting FunASR model to float32 for MPS compatibility")
                
                def convert_to_float32_recursive(module):
                    """Recursively convert all parameters and buffers to float32"""
                    if module is None:
                        return
                    
                    # Convert the module itself if it has parameters/buffers
                    try:
                        module.to(dtype=torch.float32)
                    except:
                        pass
                    
                    # For complex objects with nested models
                    if hasattr(module, 'model'):
                        convert_to_float32_recursive(module.model)
                    
                    # Check for common FunASR submodels
                    for attr_name in ['encoder', 'decoder', 'predictor', 'frontend', 
                                     'specaug', 'normalize', 'vad_model', 'punc_model', 
                                     'spk_model', 'lm_model']:
                        if hasattr(module, attr_name):
                            attr = getattr(module, attr_name)
                            if attr is not None:
                                convert_to_float32_recursive(attr)
                
                # Convert the main model and all submodels
                convert_to_float32_recursive(self.model)
                
                # Restore default dtype
                torch.set_default_dtype(old_default_dtype)
            
            print(f"[Transcriber] FunASR model loaded successfully on {funasr_device}")
        except Exception as e:
            print(f"[Transcriber] Error loading FunASR model: {e}")
            print("[Transcriber] Falling back to faster-whisper")
            self.backend = "whisper"
            self._init_whisper("base", "cpu", "int8")
    
    def _apply_mps_float32_patches(self):
        """Apply patches to ensure float32 is used on MPS device"""
        import torch
        
        # Store original functions
        self._original_torch_tensor = torch.tensor
        self._original_torch_as_tensor = torch.as_tensor
        self._original_torch_zeros = torch.zeros
        self._original_torch_ones = torch.ones
        self._original_torch_empty = torch.empty
        self._original_torch_arange = torch.arange
        self._original_torch_linspace = torch.linspace
        self._original_torch_full = torch.full
        self._original_torch_cumsum = torch.cumsum
        self._original_torch_cumprod = torch.cumprod
        self._original_torch_cat = torch.cat
        self._original_torch_stack = torch.stack
        
        # Create patched versions that force float32 for floating point types
        def patched_tensor(*args, **kwargs):
            if 'dtype' not in kwargs and len(args) > 0:
                # Check if data looks like it would default to float64
                try:
                    import numpy as np
                    data = args[0]
                    # Handle numpy array
                    if isinstance(data, np.ndarray):
                        if data.dtype == np.float64:
                            kwargs['dtype'] = torch.float32
                    # Handle numpy scalar
                    elif isinstance(data, (np.float64, np.double)): 
                        kwargs['dtype'] = torch.float32
                    # Handle standard list/tuple
                    elif isinstance(data, (list, tuple)):
                        # optimize: check first element
                        if len(data) > 0:
                             first = data[0]
                             if isinstance(first, (np.float64, np.double)):
                                 kwargs['dtype'] = torch.float32
                except:
                    pass
            
            # Explicitly catch float64 requests and convert to float32
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
                
            return self._original_torch_tensor(*args, **kwargs)
        
        def patched_as_tensor(data, dtype=None, device=None):
            # Check for float64 dtype request
            if dtype == torch.float64:
                dtype = torch.float32
            
            # Check if input is float64 numpy array/scalar and no dtype specified
            if dtype is None:
                try:
                    import numpy as np
                    # Check for array or scalar with dtype attribute
                    if hasattr(data, 'dtype') and data.dtype == np.float64:
                         dtype = torch.float32
                    # Check explicitly for np.float64 instance (scalar)
                    elif isinstance(data, (np.float64, np.double)): 
                         dtype = torch.float32
                except:
                    pass
            
            return self._original_torch_as_tensor(data, dtype=dtype, device=device)

        def patched_from_numpy(ndarray):
            import numpy as np
            if isinstance(ndarray, np.ndarray) and ndarray.dtype == np.float64:
                return self._original_torch_from_numpy(ndarray.astype(np.float32))
            return self._original_torch_from_numpy(ndarray)
        
        def patched_zeros(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            elif 'dtype' not in kwargs and 'device' in kwargs and kwargs['device'] == 'mps':
                kwargs['dtype'] = torch.float32
            return self._original_torch_zeros(*args, **kwargs)
        
        def patched_ones(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            elif 'dtype' not in kwargs and 'device' in kwargs and kwargs['device'] == 'mps':
                kwargs['dtype'] = torch.float32
            return self._original_torch_ones(*args, **kwargs)
        
        def patched_empty(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            elif 'dtype' not in kwargs and 'device' in kwargs and kwargs['device'] == 'mps':
                kwargs['dtype'] = torch.float32
            return self._original_torch_empty(*args, **kwargs)
        
        def patched_arange(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            return self._original_torch_arange(*args, **kwargs)
        
        def patched_linspace(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            return self._original_torch_linspace(*args, **kwargs)
        
        def patched_full(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            elif 'dtype' not in kwargs and 'device' in kwargs and kwargs['device'] == 'mps':
                kwargs['dtype'] = torch.float32
            return self._original_torch_full(*args, **kwargs)
        
        def patched_cumsum(*args, **kwargs):
            # This is the critical patch for FunASR's CIF predictor
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            return self._original_torch_cumsum(*args, **kwargs)
        
        def patched_cumprod(*args, **kwargs):
            if kwargs.get('dtype') == torch.float64:
                kwargs['dtype'] = torch.float32
            return self._original_torch_cumprod(*args, **kwargs)
        
        def patched_cat(tensors, *args, **kwargs):
            """Patch torch.cat to handle device mismatches on MPS"""
            # Check if any tensor is on MPS
            has_mps = any(t.device.type == 'mps' for t in tensors if hasattr(t, 'device'))
            
            if has_mps:
                # Move all CPU tensors to MPS to avoid device mismatch
                tensors = [
                    t.to('mps') if hasattr(t, 'device') and t.device.type == 'cpu' else t
                    for t in tensors
                ]
            
            return self._original_torch_cat(tensors, *args, **kwargs)
        
        def patched_stack(tensors, *args, **kwargs):
            """Patch torch.stack to handle device mismatches on MPS"""
            # Check if any tensor is on MPS
            has_mps = any(t.device.type == 'mps' for t in tensors if hasattr(t, 'device'))
            
            if has_mps:
                # Move all CPU tensors to MPS to avoid device mismatch
                tensors = [
                    t.to('mps') if hasattr(t, 'device') and t.device.type == 'cpu' else t
                    for t in tensors
                ]
            
            return self._original_torch_stack(tensors, *args, **kwargs)

        # Apply patches
        torch.tensor = patched_tensor
        torch.as_tensor = patched_as_tensor
        torch.zeros = patched_zeros
        torch.ones = patched_ones
        torch.empty = patched_empty
        torch.arange = patched_arange
        torch.linspace = patched_linspace
        torch.full = patched_full
        torch.cumsum = patched_cumsum
        torch.cumprod = patched_cumprod
        torch.cat = patched_cat
        torch.stack = patched_stack
        
        # Also patch from_numpy as it's a common source of float64 tensors
        if hasattr(torch, 'from_numpy'):
            self._original_torch_from_numpy = torch.from_numpy
            torch.from_numpy = patched_from_numpy
        
        print("[Transcriber] Applied comprehensive torch tensor patches for MPS compatibility")
    
    def _get_funasr_device(self, device):
        """Determine the best device for FunASR based on configuration and hardware"""
        import platform
        
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False
            print("[Transcriber] PyTorch not available, using CPU for FunASR")
            return "cpu"
        
        # If user explicitly set a device, validate and respect it
        if device and device.lower() != "auto":
            # MPS is supported with float32 only
            if device.lower() in ["mps", "metal"]:
                if has_torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print("[Transcriber] Using MPS device for FunASR (ensure float32 quantization)")
                    return "mps"
                else:
                    print("[Transcriber] MPS not available, falling back to CPU")
                    return "cpu"
            
            if device.lower() == "cuda" and has_torch and torch.cuda.is_available():
                return "cuda"
            elif device.lower() == "cpu":
                return "cpu"
            elif device.lower().startswith("cuda:") and has_torch and torch.cuda.is_available():
                return device.lower()
        
        # Auto-detect based on hardware
        system = platform.system()
        machine = platform.machine()
        
        # Check for CUDA (NVIDIA GPU) - highest priority for FunASR
        if has_torch and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"[Transcriber] Detected {gpu_count} CUDA device(s)")
            return "cuda"
        
        # Apple Silicon: Use CPU by default for best compatibility
        # User can explicitly choose MPS if they want GPU acceleration with float32
        if system == "Darwin" and machine == "arm64":
            print("[Transcriber] Apple Silicon detected")
            print("[Transcriber] Using CPU (set device=mps with float32 for GPU acceleration)")
            return "cpu"
        
        # Default to CPU
        print("[Transcriber] No compatible GPU detected, using CPU")
        return "cpu"
    def transcribe(self, audio_data, prompt=None):
        """Transcribe audio using the configured backend"""
        if self.backend == "funasr":
            text = self._transcribe_funasr(audio_data, prompt)
        elif self.backend == "mlx":
            text = self._transcribe_mlx(audio_data, prompt)
        else:  # whisper
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

    def _transcribe_funasr(self, audio_data, prompt=None):
        """Transcribe using FunASR backend"""
        try:
            import torch
            
            # FunASR expects audio data in specific format
            # Convert numpy array to the format FunASR expects
            # Most FunASR models expect 16kHz audio
            
            # Ensure audio is in the right shape and format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # For MPS device, ensure audio data is float32 to avoid float64 conversion errors
            if hasattr(self, 'funasr_device') and self.funasr_device == "mps":
                # Ensure audio data is explicitly float32
                audio_data = audio_data.astype(np.float32)
                # Convert to torch tensor with explicit float32 dtype for MPS
                audio_tensor = torch.from_numpy(audio_data).to(dtype=torch.float32, device="mps")
                input_data = audio_tensor
                
                # Set default dtype to float32 during inference to prevent any float64 operations
                old_default_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                
                
                try:
                    # FunASR AutoModel.generate() accepts audio directly
                    result = self.model.generate(
                        input=input_data,
                        batch_size_s=300,  # Process in batches
                        hotword="" if not prompt else prompt
                    )
                except RuntimeError as e:
                    if "float64" in str(e):
                        print(f"[Transcriber] FunASR Error: {e}")
                        # If we still get float64 error, log detailed traceback for debugging
                        import traceback
                        print("[Transcriber] Detailed traceback:")
                        traceback.print_exc()
                        return ""
                    else:
                        raise
                finally:
                    # Restore default dtype
                    torch.set_default_dtype(old_default_dtype)
            else:
                # For CPU/CUDA, use numpy array directly
                input_data = audio_data
                # FunASR AutoModel.generate() accepts audio directly
                result = self.model.generate(
                    input=input_data,
                    batch_size_s=300,  # Process in batches
                    hotword="" if not prompt else prompt
                )
            
            # Extract text from result
            if isinstance(result, list) and len(result) > 0:
                # FunASR returns a list of results
                text_parts = []
                for item in result:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    elif isinstance(item, str):
                        text_parts.append(item)
                return " ".join(text_parts).strip()
            elif isinstance(result, dict) and 'text' in result:
                return result['text'].strip()
            else:
                return ""
                
        except Exception as e:
            error_msg = str(e)
            if "float64" in error_msg and hasattr(self, 'funasr_device') and self.funasr_device == "mps":
                print(f"[Transcriber] FunASR Error: {e}")
                import traceback
                import sys
                print("[Transcriber] Full traceback (saving to /tmp/funasr_mps_error.txt):")
                with open('/tmp/funasr_mps_error.txt', 'w') as f:
                    traceback.print_exc(file=f)
                    f.write("\n\n=== Stack ===\n")
                    traceback.print_stack(file=f)
                traceback.print_exc()
            else:
                print(f"[Transcriber] FunASR Error: {e}")
            return ""

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
            error_msg = str(e)
            # Handle unsupported language error gracefully
            if "Unsupported language" in error_msg and self.language:
                print(f"[Transcriber] Language '{self.language}' not supported, falling back to auto-detection")
                self.language = None  # Switch to auto-detect
                # Retry with auto-detection
                try:
                    kwargs["language"] = None
                    result = mlx_whisper.transcribe(audio_data, **kwargs)
                    return result.get("text", "").strip()
                except Exception as retry_error:
                    print(f"[Transcriber] MLX Error on retry: {retry_error}")
                    return ""
            else:
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
