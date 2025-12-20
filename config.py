import configparser
import os

class Config:
    """Centralized configuration loaded from config.ini"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Look for config.ini in the same directory as this script
            config_path = os.path.join(os.path.dirname(__file__), "config.ini")
        
        self.config = configparser.ConfigParser()
        
        if os.path.exists(config_path):
            self.config.read(config_path)
            print(f"[Config] Loaded from: {config_path}")
        else:
            print(f"[Config] Warning: {config_path} not found, using defaults/env vars")
        
        # API settings (env vars take precedence)
        self.api_base_url = os.getenv("OPENAI_BASE_URL") or self._get("api", "base_url") or None
        self.api_key = os.getenv("OPENAI_API_KEY") or self._get("api", "api_key", "dummy-key-for-local")
        
        # Translation settings
        self.model = self._get("translation", "model", "gpt-3.5-turbo")
        self.model = self._get("translation", "model", "gpt-3.5-turbo")
        self.target_lang = self._get("translation", "target_lang", "Chinese")
        self.translation_threads = self._getint("translation", "threads", 4)
        
        # Transcription settings
        self.asr_backend = self._get("transcription", "backend", "whisper").lower()
        self.whisper_model = self._get("transcription", "whisper_model", "base")
        self.funasr_model = self._get("transcription", "funasr_model", "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        self.whisper_device = self._get("transcription", "device", "cpu")
        self.whisper_compute_type = self._get("transcription", "compute_type", "int8")
        self.source_language = self._get("transcription", "source_language", "auto")
        if self.source_language == "auto":
            self.source_language = None  # Whisper uses None for auto-detect
        self.transcription_workers = self._getint("transcription", "transcription_workers", 2)
        
        # Diarization settings
        self.enable_diarization = self._get("diarization", "enable_diarization", "false").lower() == "true"
        self.diarization_step = self._getfloat("diarization", "step", 0.5)
        self.diarization_latency = self._getfloat("diarization", "latency", 0.5)
        
        # Audio settings
        self.sample_rate = self._getint("audio", "sample_rate", 16000)
        self.silence_threshold = self._getfloat("audio", "silence_threshold", 0.01)
        self.silence_duration = self._getfloat("audio", "silence_duration", 1.0)
        self.chunk_duration = self._getfloat("audio", "chunk_duration", 0.5)
        
        # Device index: 'auto' or empty = auto-detect BlackHole, or set a specific index
        device_idx_str = self._get("audio", "device_index", "auto")
        if device_idx_str.isdigit():
            self.device_index = int(device_idx_str)
        elif device_idx_str.lower() in ("auto", ""):
            self.device_index = self._find_blackhole_device()
        else:
            self.device_index = None
            
        # Max phrase duration - force processing after N seconds
        self.max_phrase_duration = self._getfloat("audio", "max_phrase_duration", 5.0)
        
        # Streaming mode settings
        self.streaming_mode = self._get("audio", "streaming_mode", "false").lower() == "true"
        self.streaming_interval = self._getfloat("audio", "streaming_interval", 1.5)
        self.streaming_step_size = self._getfloat("audio", "streaming_step_size", 0.2)
        self.update_interval = self._getfloat("audio", "update_interval", 0.5)
        self.streaming_overlap = self._getfloat("audio", "streaming_overlap", 0.3)
        
        # Display settings
        self.display_duration = self._getfloat("display", "display_duration", 3.0)
        self.window_width = self._getint("display", "window_width", 800)
        self.window_height = self._getint("display", "window_height", 120)
    
    def _get(self, section, key, fallback=""):
        try:
            value = self.config.get(section, key)
            return value if value else fallback
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    
    def _getint(self, section, key, fallback=0):
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def _getfloat(self, section, key, fallback=0.0):
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    
    def _find_blackhole_device(self):
        """Auto-detect BlackHole audio device index"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0 and 'blackhole' in d['name'].lower():
                    print(f"[Config] Auto-detected BlackHole device: [{i}] {d['name']}")
                    return i
            print("[Config] BlackHole not found, using default input device")
            return None
        except Exception as e:
            print(f"[Config] Error detecting audio devices: {e}")
            return None
    
    def print_config(self):
        """Print current configuration for debugging"""
        print("[Config] Current settings:")
        print(f"  API Base URL: {self.api_base_url or '(default OpenAI)'}")
        print(f"  API Key: {self.api_key[:8]}...{self.api_key[-4:] if len(self.api_key) > 12 else '***'}")
        print(f"  Model: {self.model}")
        print(f"  Target Language: {self.target_lang}")
        print(f"  ASR Backend: {self.asr_backend}")
        print(f"  Whisper Model: {self.whisper_model}")
        print(f"  FunASR Model: {self.funasr_model}")
        print(f"  Sample Rate: {self.sample_rate}")
        print(f"  Diarization Enabled: {self.enable_diarization}")

# Global config instance
config = Config()
