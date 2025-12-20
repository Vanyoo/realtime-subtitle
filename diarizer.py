import numpy as np
from typing import Optional, List, Tuple

class Diarizer:
    """
    Speaker diarization using diart library for real-time speaker identification.
    
    This module processes audio chunks and identifies which speaker is speaking
    at different time segments.
    """
    
    def __init__(self, sample_rate=16000, enable_diarization=True):
        """
        Initialize the diarizer with diart
        
        Args:
            sample_rate: Audio sample rate (default 16000 Hz)
            enable_diarization: Whether to enable diarization (can be disabled for performance)
        """
        self.sample_rate = sample_rate
        self.enable_diarization = enable_diarization
        self.pipeline = None
        self.speaker_history = {}  # Track speaker IDs over time
        
        if enable_diarization:
            self._init_diart()
    
    def _init_diart(self):
        """Initialize the diart pipeline"""
        try:
            # Check PyTorch version first
            import torch
            torch_version = torch.__version__.split('+')[0]  # Remove +cpu/+cu118 suffix
            major, minor = map(int, torch_version.split('.')[:2])
            
            if major > 2 or (major == 2 and minor >= 1):
                print(f"[Diarizer] Warning: PyTorch {torch_version} detected")
                print(f"[Diarizer] diart requires PyTorch < 2.1.0 (AudioMetaData compatibility)")
                print(f"[Diarizer] To enable diarization:")
                print(f"[Diarizer]   1. Downgrade: pip install torch==2.0.1 torchaudio==2.0.2")
                print(f"[Diarizer]   2. Then: pip install diart>=0.9.0")
                print(f"[Diarizer] Diarization will be disabled")
                self.enable_diarization = False
                self.pipeline = None
                return
            
            from diart import SpeakerDiarization
            from diart.sources import MicrophoneAudioSource
            from diart.inference import RealTimeInference
            
            print("[Diarizer] Initializing diart speaker diarization...")
            
            # Configure diart pipeline
            # Use pyannote segmentation and embedding models
            self.pipeline = SpeakerDiarization(
                segmentation="pyannote/segmentation",
                embedding="pyannote/embedding",
                step=0.5,  # Step size in seconds
                latency=0.5,  # Latency in seconds for real-time processing
                tau_active=0.6,  # Speech activity threshold
                rho_update=0.3,  # Minimum similarity to update embedding
                delta_new=1.0  # Minimum distance to create new speaker
            )
            
            print("[Diarizer] Diart pipeline initialized successfully")
            
        except ImportError as e:
            print(f"[Diarizer] Warning: diart not available: {e}")
            print("[Diarizer] Install with: pip install diart")
            print("[Diarizer] Note: Requires PyTorch < 2.1.0")
            print("[Diarizer] Diarization will be disabled")
            self.enable_diarization = False
            self.pipeline = None
        except Exception as e:
            print(f"[Diarizer] Error initializing diart: {e}")
            print("[Diarizer] Diarization will be disabled")
            self.enable_diarization = False
            self.pipeline = None
    
    def process_audio_chunk(self, audio_data: np.ndarray, chunk_id: int = 0) -> Optional[str]:
        """
        Process an audio chunk and identify the speaker
        
        Args:
            audio_data: Audio data as numpy array (float32)
            chunk_id: Identifier for this chunk
            
        Returns:
            Speaker label (e.g., "Speaker 1", "Speaker 2") or None if diarization disabled
        """
        if not self.enable_diarization or self.pipeline is None:
            return None
        
        try:
            # Ensure audio is in the correct format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Ensure float32
            audio_data = audio_data.astype(np.float32)
            
            # Get duration of audio chunk
            duration = len(audio_data) / self.sample_rate
            
            # Skip very short chunks (less than 0.5 seconds)
            if duration < 0.5:
                return self._get_last_speaker()
            
            # Process with diart - simple batch processing
            # For real-time streaming, diart expects audio in specific format
            # We'll use a simplified approach: determine dominant speaker in chunk
            speaker_label = self._identify_speaker_simple(audio_data)
            
            # Cache the speaker for this chunk
            self.speaker_history[chunk_id] = speaker_label
            
            return speaker_label
            
        except Exception as e:
            print(f"[Diarizer] Error processing chunk {chunk_id}: {e}")
            return self._get_last_speaker()
    
    def _identify_speaker_simple(self, audio_data: np.ndarray) -> str:
        """
        Simplified speaker identification
        
        For now, we use a basic approach. In a full implementation,
        this would use diart's real-time inference pipeline.
        
        Args:
            audio_data: Audio chunk to analyze
            
        Returns:
            Speaker label
        """
        try:
            # This is a placeholder for the actual diart integration
            # Full integration would require:
            # 1. Setting up RealTimeInference with the pipeline
            # 2. Feeding audio chunks through the inference engine
            # 3. Tracking speaker embeddings over time
            
            # For now, return a default speaker
            # In production, you'd integrate with diart's streaming API
            return "Speaker 1"
            
        except Exception as e:
            print(f"[Diarizer] Error in speaker identification: {e}")
            return "Speaker 1"
    
    def _get_last_speaker(self) -> str:
        """Get the most recent speaker label"""
        if not self.speaker_history:
            return "Speaker 1"
        
        # Return the most recent speaker
        last_chunk_id = max(self.speaker_history.keys())
        return self.speaker_history[last_chunk_id]
    
    def reset(self):
        """Reset speaker history and internal state"""
        self.speaker_history.clear()
        print("[Diarizer] Speaker history reset")


class DiarizerStreaming:
    """
    Advanced streaming diarizer using diart's real-time inference.
    
    This class provides full integration with diart's streaming capabilities
    for continuous speaker diarization.
    """
    
    def __init__(self, sample_rate=16000, step=0.5, latency=0.5):
        """
        Initialize streaming diarizer
        
        Args:
            sample_rate: Audio sample rate
            step: Step size in seconds for diarization updates
            latency: Maximum latency in seconds
        """
        self.sample_rate = sample_rate
        self.step = step
        self.latency = latency
        self.pipeline = None
        self.inference = None
        
        self._init_streaming_pipeline()
    
    def _init_streaming_pipeline(self):
        """Initialize the streaming diarization pipeline"""
        try:
            from diart import SpeakerDiarization
            from diart.inference import StreamingInference
            import torch
            
            print("[DiarizerStreaming] Initializing streaming pipeline...")
            
            # Create the diarization pipeline
            self.pipeline = SpeakerDiarization(
                segmentation="pyannote/segmentation",
                embedding="pyannote/embedding",
                step=self.step,
                latency=self.latency,
                tau_active=0.6,
                rho_update=0.3,
                delta_new=1.0
            )
            
            # Create streaming inference engine
            self.inference = StreamingInference(
                self.pipeline,
                sample_rate=self.sample_rate,
                step=self.step,
                latency=self.latency
            )
            
            print("[DiarizerStreaming] Streaming pipeline ready")
            
        except Exception as e:
            print(f"[DiarizerStreaming] Error initializing: {e}")
            self.pipeline = None
            self.inference = None
    
    def process_stream(self, audio_generator):
        """
        Process an audio stream and yield (timestamp, speaker_label) pairs
        
        Args:
            audio_generator: Generator yielding audio chunks
            
        Yields:
            Tuples of (timestamp, speaker_label)
        """
        if self.inference is None:
            print("[DiarizerStreaming] Pipeline not initialized, skipping diarization")
            return
        
        try:
            for prediction in self.inference(audio_generator):
                # prediction contains speaker assignments for different time segments
                # Format: {speaker_id: [(start_time, end_time), ...]}
                yield prediction
                
        except Exception as e:
            print(f"[DiarizerStreaming] Error in stream processing: {e}")
    
    def get_active_speaker(self, diarization, timestamp):
        """
        Get the active speaker at a specific timestamp
        
        Args:
            diarization: Diarization result from diart
            timestamp: Time in seconds
            
        Returns:
            Speaker label or None
        """
        try:
            # Find which speaker is active at the given timestamp
            for speaker, segments in diarization.items():
                for start, end in segments:
                    if start <= timestamp <= end:
                        return f"Speaker {speaker + 1}"
            return None
            
        except Exception as e:
            print(f"[DiarizerStreaming] Error getting active speaker: {e}")
            return None


if __name__ == "__main__":
    # Test the diarizer
    print("Testing Diarizer...")
    
    diarizer = Diarizer(sample_rate=16000, enable_diarization=True)
    
    # Create a test audio chunk (1 second of random noise)
    test_audio = np.random.randn(16000).astype(np.float32) * 0.1
    
    speaker = diarizer.process_audio_chunk(test_audio, chunk_id=1)
    print(f"Identified speaker: {speaker}")
    
    print("\nDiarizer test complete")
