# Claude Code Implementation Plan: 8D Latent Voice Synthesizer

Here's a comprehensive prompt you can give to Claude Code to build this system step-by-step:

---

## PROJECT PROMPT FOR CLAUDE CODE

```
I need you to build a real-time voice synthesizer controlled by 8-dimensional articulatory latent vectors, based on DDSP (Differentiable Digital Signal Processing) architecture.

## Project Context

I have an AI model that outputs 8D latent vectors representing articulatory positions at 100Hz. These need to be converted to audio in real-time (22050 Hz sample rate). My previous formant synthesizer failed because it couldn't handle consonants properly - consonants require parallel noise excitation, not just cascade filtering.

## 8D Latent Space Specification

Each latent vector has 8 dimensions, all in range [-1, 1] except intensity [0, 1]:

0. front-back: Place of articulation (lips to throat)
1. high-low: Tongue height
2. rounded: Lip rounding
3. voice-breath: Voiced (-1) ↔ silence (0) ↔ unvoiced (+1)
4. manner: Stop (-1) ↔ fricative (0) ↔ approximant (+1)
5. vowel-cons: Vowel (-1) ↔ consonant (+1)
6. pitch: Fundamental frequency
7. intensity: Amplitude [0, 1]

## Architecture Requirements

Build a hybrid DDSP synthesizer with:

1. **Latent-to-Parameters Decoder**: Small MLP mapping 8D → synthesis parameters
2. **Harmonic Synthesizer**: Cascade branch for voiced sounds (vowels, voiced consonants)
3. **Noise Synthesizer**: Parallel branch for unvoiced sounds (fricatives, stops)
4. **Hybrid Mixer**: Blends harmonic + noise based on voice-breath dimension
5. **Real-time Streaming**: Accepts 100Hz latent stream, outputs 22050Hz audio

The entire pipeline must be differentiable (PyTorch) for future gradient flow.

## Implementation Steps

### Phase 1: Core DDSP Components (Start Here)

Create a Python package with this structure:

```
latent_synth/
├── __init__.py
├── ddsp_core.py          # Core DDSP synthesis primitives
├── latent_decoder.py     # 8D → parameters MLP
├── synthesizer.py        # Main hybrid synthesizer
├── streaming.py          # Real-time streaming interface
└── utils.py              # Helper functions

tests/
├── test_ddsp_core.py
├── test_vowels.py        # Test static vowel synthesis
├── test_consonants.py    # Test consonant synthesis
└── test_streaming.py

examples/
├── static_synthesis.py   # Synthesize from static latents
├── playground.py         # Interactive real-time demo
└── benchmark.py          # Performance testing
```

### Phase 1.1: Implement ddsp_core.py

Build these core DDSP primitives:

```python
import torch
import torch.nn as nn
import numpy as np

class HarmonicOscillator(nn.Module):
    """
    Generates harmonic content from F0 and harmonic amplitudes.
    Uses anti-aliased oscillators to prevent artifacts.
    """
    def __init__(self, sample_rate=22050, block_size=220):
        # block_size = samples per frame at 100Hz
        # Implement additive synthesis with up to 100 harmonics
        # Use cumulative phase to avoid discontinuities
        pass
    
    def forward(self, f0, harmonic_amplitudes):
        """
        Args:
            f0: [batch, frames] - fundamental frequency in Hz
            harmonic_amplitudes: [batch, frames, n_harmonics] - amplitude per harmonic
        Returns:
            audio: [batch, samples] - synthesized audio
        """
        pass

class FilteredNoise(nn.Module):
    """
    Generates filtered noise for consonants.
    Uses time-varying FIR filtering of white noise.
    """
    def __init__(self, sample_rate=22050, n_bands=64):
        # n_bands = frequency bands for noise shaping
        pass
    
    def forward(self, noise_magnitudes):
        """
        Args:
            noise_magnitudes: [batch, frames, n_bands] - per-band energy
        Returns:
            audio: [batch, samples] - filtered noise
        """
        pass

class FormantFilter(nn.Module):
    """
    Differentiable formant filtering using time-varying IIR.
    Implements F1, F2, F3, F4 resonances.
    """
    def __init__(self, sample_rate=22050):
        pass
    
    def forward(self, audio, formant_freqs, formant_bandwidths):
        """
        Args:
            audio: [batch, samples] - input audio
            formant_freqs: [batch, frames, 4] - F1-F4 frequencies
            formant_bandwidths: [batch, frames, 4] - bandwidths
        Returns:
            filtered_audio: [batch, samples]
        """
        pass
```

**Key implementation details:**
- Use `torch.cumsum()` for phase accumulation to maintain continuity
- Implement anti-aliasing for harmonics above Nyquist
- Use overlap-add for smooth frame transitions
- Make everything differentiable - no `.numpy()` conversions

### Phase 1.2: Implement latent_decoder.py

```python
class LatentToParameters(nn.Module):
    """
    Decodes 8D articulatory latent to DDSP synthesis parameters.
    """
    def __init__(self, n_harmonics=100, n_noise_bands=64):
        super().__init__()
        
        # Small MLP: 8 → 64 → 128 → output_dim
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        
        # Separate heads for different parameter types
        self.f0_head = nn.Linear(128, 1)
        self.voicing_head = nn.Linear(128, 1)
        self.harmonic_head = nn.Linear(128, n_harmonics)
        self.noise_head = nn.Linear(128, n_noise_bands)
        self.formant_head = nn.Linear(128, 8)  # F1-F4 freqs + bandwidths
        
    def forward(self, latent_8d):
        """
        Args:
            latent_8d: [batch, frames, 8] or [batch, 8]
        Returns:
            dict with keys: f0, voicing, harmonic_amps, noise_mags, formants
        """
        # Extract individual dimensions
        front_back = latent_8d[..., 0]
        high_low = latent_8d[..., 1]
        rounded = latent_8d[..., 2]
        voice_breath = latent_8d[..., 3]
        manner = latent_8d[..., 4]
        vowel_cons = latent_8d[..., 5]
        pitch = latent_8d[..., 6]
        intensity = latent_8d[..., 7]
        
        # Decode through MLP
        hidden = self.decoder(latent_8d)
        
        # Convert pitch dimension to F0 (80-300 Hz range)
        f0_normalized = torch.sigmoid(self.f0_head(hidden))
        f0 = 80 + f0_normalized.squeeze(-1) * 220  # 80-300 Hz
        
        # Convert voice_breath to voicing probability
        # voice_breath: -1 (voiced) → +1 (unvoiced)
        voicing = torch.sigmoid(-voice_breath * 3)  # High when voiced
        
        # Harmonic amplitudes (spectral envelope)
        harmonic_amps = torch.sigmoid(self.harmonic_head(hidden))
        
        # Noise magnitudes (for consonants)
        noise_mags = torch.sigmoid(self.noise_head(hidden))
        
        # Formant frequencies from articulatory dimensions
        formants_raw = self.formant_head(hidden)
        
        # Formant frequency ranges (in Hz):
        # F1: 200-1000 (inversely related to high_low)
        # F2: 600-2800 (related to front_back)
        # F3: 1800-3500 (affected by rounded)
        # F4: ~3500 (relatively stable)
        
        f1 = 200 + (1 - (high_low + 1) / 2) * 800  # 200-1000 Hz
        f2 = 600 + (front_back + 1) / 2 * 2200     # 600-2800 Hz
        f3 = 1800 + (1 - (rounded + 1) / 2) * 1700 # 1800-3500 Hz
        f4 = torch.ones_like(f1) * 3500
        
        formant_freqs = torch.stack([f1, f2, f3, f4], dim=-1)
        
        # Formant bandwidths (wider for consonants)
        base_bw = torch.tensor([50, 70, 110, 150])  # Default bandwidths
        bw_multiplier = 1 + vowel_cons.abs()  # Wider for consonants
        formant_bws = base_bw * bw_multiplier.unsqueeze(-1)
        
        return {
            'f0': f0,
            'voicing': voicing,
            'harmonic_amplitudes': harmonic_amps,
            'noise_magnitudes': noise_mags,
            'formant_freqs': formant_freqs,
            'formant_bandwidths': formant_bws,
            'intensity': intensity,
            'voice_breath': voice_breath,  # Keep for mixing
        }
```

### Phase 1.3: Implement synthesizer.py

```python
class HybridDDSPSynthesizer(nn.Module):
    """
    Main synthesizer combining harmonic + noise branches.
    Implements Klatt-style cascade/parallel architecture.
    """
    def __init__(self, sample_rate=22050, frame_rate=100):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.block_size = sample_rate // frame_rate  # 220 samples
        
        # Core DDSP components
        self.harmonic_synth = HarmonicOscillator(sample_rate, self.block_size)
        self.noise_synth = FilteredNoise(sample_rate)
        self.formant_filter = FormantFilter(sample_rate)
        
        # Parameter decoder
        self.param_decoder = LatentToParameters()
        
    def forward(self, latent_8d):
        """
        Args:
            latent_8d: [batch, frames, 8] - latent vectors at frame_rate
        Returns:
            audio: [batch, samples] - synthesized audio at sample_rate
        """
        # Decode latents to synthesis parameters
        params = self.param_decoder(latent_8d)
        
        # CASCADE BRANCH: Harmonic synthesis for voiced sounds
        harmonic_audio = self.harmonic_synth(
            params['f0'], 
            params['harmonic_amplitudes']
        )
        
        # Apply formant filtering to harmonics
        harmonic_filtered = self.formant_filter(
            harmonic_audio,
            params['formant_freqs'],
            params['formant_bandwidths']
        )
        
        # PARALLEL BRANCH: Noise synthesis for consonants
        noise_audio = self.noise_synth(params['noise_magnitudes'])
        
        # Apply formant filtering to noise (for fricatives)
        noise_filtered = self.formant_filter(
            noise_audio,
            params['formant_freqs'],
            params['formant_bandwidths']
        )
        
        # MIX BRANCHES based on voice_breath dimension
        # voicing: high for voiced (-1), low for unvoiced (+1)
        voicing = params['voicing']
        
        # Expand voicing from frame rate to sample rate
        voicing_samples = self._upsample_to_samples(voicing)
        
        # Blend: high voicing → mostly harmonic, low voicing → mostly noise
        mixed_audio = (
            voicing_samples * harmonic_filtered +
            (1 - voicing_samples) * noise_filtered
        )
        
        # Apply intensity envelope
        intensity_samples = self._upsample_to_samples(params['intensity'])
        output_audio = mixed_audio * intensity_samples
        
        return output_audio
    
    def _upsample_to_samples(self, frame_values):
        """Upsample frame-rate values to sample-rate with linear interpolation."""
        # frame_values: [batch, frames]
        # output: [batch, samples]
        return torch.nn.functional.interpolate(
            frame_values.unsqueeze(1),
            size=frame_values.shape[-1] * self.block_size,
            mode='linear',
            align_corners=False
        ).squeeze(1)
    
    def synthesize_static(self, latent_8d, duration_sec=1.0):
        """
        Synthesize audio from a single static latent vector.
        Useful for testing.
        """
        n_frames = int(duration_sec * self.frame_rate)
        latent_repeated = latent_8d.unsqueeze(0).unsqueeze(0).repeat(1, n_frames, 1)
        return self.forward(latent_repeated)
```

### Phase 1.4: Implement streaming.py

```python
class RealtimeLatentSynth:
    """
    Real-time streaming interface for synthesizer.
    Accepts latent vectors at 100Hz, outputs audio chunks at 22050Hz.
    """
    def __init__(self, synthesizer, buffer_frames=5):
        self.synth = synthesizer
        self.buffer_frames = buffer_frames
        self.latent_buffer = []
        
    def push_latent(self, latent_8d):
        """
        Push a single 8D latent vector.
        
        Args:
            latent_8d: torch.Tensor [8] - single latent vector
        Returns:
            audio_chunk: np.ndarray or None if buffering
        """
        self.latent_buffer.append(latent_8d)
        
        if len(self.latent_buffer) >= self.buffer_frames:
            # Process buffered frames
            latent_batch = torch.stack(self.latent_buffer).unsqueeze(0)  # [1, frames, 8]
            
            with torch.no_grad():
                audio = self.synth(latent_batch)
            
            # Clear buffer (or keep overlap for smoothness)
            self.latent_buffer = self.latent_buffer[-1:]  # Keep last frame
            
            return audio.squeeze(0).cpu().numpy()
        
        return None
    
    def stream_from_generator(self, latent_generator):
        """
        Generator that yields audio chunks from latent stream.
        
        Args:
            latent_generator: yields 8D latent vectors
        Yields:
            audio_chunks: np.ndarray
        """
        for latent in latent_generator:
            audio_chunk = self.push_latent(latent)
            if audio_chunk is not None:
                yield audio_chunk
```

### Phase 2: Testing & Validation

Create comprehensive tests:

```python
# tests/test_vowels.py

import torch
from latent_synth import HybridDDSPSynthesizer
import soundfile as sf

# Standard vowel latents (from your design doc)
VOWEL_LATENTS = {
    'IY': torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7]),   # "ee" as in "beet"
    'IH': torch.tensor([0.8, 0.7, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7]),   # "i" as in "bit"
    'EH': torch.tensor([0.6, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7]),   # "e" as in "bet"
    'AE': torch.tensor([0.6, -0.7, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7]),  # "a" as in "bat"
    'AA': torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7]), # "a" as in "father"
    'AO': torch.tensor([-0.8, -0.5, 0.5, -1.0, 0.0, -1.0, 0.0, 0.7]),  # "aw" as in "caught"
    'UH': torch.tensor([-0.5, 0.5, 0.5, -1.0, 0.0, -1.0, 0.0, 0.7]),   # "oo" as in "book"
    'UW': torch.tensor([-1.0, 1.0, 1.0, -1.0, 0.0, -1.0, 0.0, 0.7]),   # "oo" as in "boot"
}

def test_vowel_synthesis():
    synth = HybridDDSPSynthesizer()
    
    for vowel_name, latent in VOWEL_LATENTS.items():
        print(f"Synthesizing {vowel_name}...")
        audio = synth.synthesize_static(latent, duration_sec=0.5)
        
        # Save to file
        sf.write(f'test_output/{vowel_name}.wav', 
                 audio.cpu().numpy(), 
                 22050)
        
        # Basic checks
        assert audio.abs().max() < 1.0, "Audio clipping detected"
        assert audio.abs().mean() > 0.01, "Audio too quiet"
        
    print("✓ All vowels synthesized successfully")

def test_vowel_formants():
    """Verify formant frequencies are in expected ranges."""
    synth = HybridDDSPSynthesizer()
    
    # Test IY (high front): expect high F2
    latent_iy = VOWEL_LATENTS['IY']
    params = synth.param_decoder(latent_iy)
    
    assert params['formant_freqs'][0] < 400, "F1 too high for IY"
    assert params['formant_freqs'][1] > 2000, "F2 too low for IY"
    
    # Test AA (low back): expect low F2
    latent_aa = VOWEL_LATENTS['AA']
    params = synth.param_decoder(latent_aa)
    
    assert params['formant_freqs'][0] > 600, "F1 too low for AA"
    assert params['formant_freqs'][1] < 1200, "F2 too high for AA"
    
    print("✓ Formant frequencies in expected ranges")
```

```python
# tests/test_consonants.py

CONSONANT_LATENTS = {
    # Stops
    'P': torch.tensor([-1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.7]),   # voiceless bilabial
    'B': torch.tensor([-1.0, 0.0, 0.0, -1.0, -1.0, 1.0, 0.0, 0.7]),  # voiced bilabial
    'T': torch.tensor([0.5, 0.5, 0.0, 1.0, -1.0, 1.0, 0.0, 0.7]),    # voiceless alveolar
    'D': torch.tensor([0.5, 0.5, 0.0, -1.0, -1.0, 1.0, 0.0, 0.7]),   # voiced alveolar
    
    # Fricatives
    'F': torch.tensor([-0.8, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0, 0.6]),    # voiceless labiodental
    'S': torch.tensor([0.7, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 0.6]),     # voiceless alveolar
    'SH': torch.tensor([0.3, 0.5, 0.5, 1.0, 0.5, 1.0, 0.0, 0.6]),    # voiceless postalveolar
    
    # Nasals
    'M': torch.tensor([-1.0, 0.0, 0.0, -1.0, -0.5, 0.5, 0.0, 0.7]),  # bilabial nasal
    'N': torch.tensor([0.5, 0.5, 0.0, -1.0, -0.5, 0.5, 0.0, 0.7]),   # alveolar nasal
}

def test_consonant_synthesis():
    synth = HybridDDSPSynthesizer()
    
    for cons_name, latent in CONSONANT_LATENTS.items():
        print(f"Synthesizing {cons_name}...")
        audio = synth.synthesize_static(latent, duration_sec=0.3)
        
        sf.write(f'test_output/{cons_name}.wav', 
                 audio.cpu().numpy(), 
                 22050)
        
    print("✓ All consonants synthesized")

def test_cv_transitions():
    """Test consonant-vowel transitions (ba, da, ga, etc.)"""
    synth = HybridDDSPSynthesizer()
    
    cv_pairs = [
        ('B', 'AA'),  # "ba"
        ('D', 'AA'),  # "da"
        ('P', 'AA'),  # "pa"
        ('T', 'AA'),  # "ta"
    ]
    
    for cons, vowel in cv_pairs:
        # Create transition: consonant (100ms) → vowel (400ms)
        latent_cons = CONSONANT_LATENTS[cons]
        latent_vowel = VOWEL_LATENTS[vowel]
        
        # Linear interpolation over 50ms
        transition_frames = 5  # 50ms at 100Hz
        latents = []
        
        # Consonant closure (5 frames = 50ms)
        latents.extend([latent_cons] * 5)
        
        # Transition (5 frames = 50ms)
        for i in range(transition_frames):
            alpha = i / transition_frames
            latent_interp = (1 - alpha) * latent_cons + alpha * latent_vowel
            latents.append(latent_interp)
        
        # Sustained vowel (40 frames = 400ms)
        latents.extend([latent_vowel] * 40)
        
        latent_seq = torch.stack(latents).unsqueeze(0)  # [1, frames, 8]
        audio = synth(latent_seq)
        
        sf.write(f'test_output/{cons}_{vowel}.wav', 
                 audio.squeeze().cpu().numpy(), 
                 22050)
        
    print("✓ CV transitions synthesized")
```

### Phase 3: Interactive Playground

```python
# examples/playground.py

"""
Real-time interactive playground for latent synthesis.
- Keyboard/mouse controls 8D latent space
- Real-time audio output
- Latent visualization
"""

import torch
import numpy as np
import sounddevice as sd
import pygame
from latent_synth import HybridDDSPSynthesizer, RealtimeLatentSynth

class LatentPlayground:
    def __init__(self):
        self.synth = HybridDDSPSynthesizer()
        self.stream_synth = RealtimeLatentSynth(self.synth, buffer_frames=3)
        
        # Current latent state
        self.latent = torch.zeros(8)
        self.latent[7] = 0.7  # Default intensity
        
        # Audio output
        self.sample_rate = 22050
        self.audio_buffer = []
        
        # Start audio stream
        self.audio_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=2205  # 100ms blocks
        )
        self.audio_stream.start()
        
        # Pygame for UI
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Latent Voice Playground")
        
    def audio_callback(self, outdata, frames, time, status):
        """Called by sounddevice to fill audio buffer."""
        if len(self.audio_buffer) >= frames:
            data = self.audio_buffer[:frames]
            self.audio_buffer = self.audio_buffer[frames:]
            outdata[:] = np.array(data).reshape(-1, 1)
        else:
            outdata[:] = np.zeros((frames, 1))
    
    def update_latent(self):
        """Update latent based on user input."""
        keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Mouse X → front-back
        self.latent[0] = (mouse_x / 800) * 2 - 1
        
        # Mouse Y → high-low
        self.latent[1] = 1 - (mouse_y / 600) * 2
        
        # Keys for other dimensions
        if keys[pygame.K_r]:
            self.latent[2] = min(1.0, self.latent[2] + 0.1)  # More rounded
        if keys[pygame.K_u]:
            self.latent[2] = max(-1.0, self.latent[2] - 0.1)  # Less rounded
        
        # Voice-breath: V for voiced, B for breathy
        if keys[pygame.K_v]:
            self.latent[3] = -1.0  # Fully voiced
        elif keys[pygame.K_b]:
            self.latent[3] = 1.0   # Unvoiced
        else:
            self.latent[3] = 0.0   # Silence
        
        # Pitch: arrow keys
        if keys[pygame.K_UP]:
            self.latent[6] = min(1.0, self.latent[6] + 0.05)
        if keys[pygame.K_DOWN]:
            self.latent[6] = max(-1.0, self.latent[6] - 0.05)
        
        # Intensity: space bar
        if keys[pygame.K_SPACE]:
            self.latent[7] = 1.0
        else:
            self.latent[7] = 0.5
    
    def synthesize_and_buffer(self):
        """Generate audio from current latent and add to buffer."""
        audio_chunk = self.stream_synth.push_latent(self.latent.clone())
        if audio_chunk is not None:
            self.audio_buffer.extend(audio_chunk.tolist())
    
    def draw_ui(self):
        """Draw latent space visualization."""
        self.screen.fill((20, 20, 30))
        
        # Draw vowel space (F1 vs F2 approximation)
        pygame.draw.circle(self.screen, (100, 100, 255), 
                          (int((self.latent[0] + 1) * 400), 
                           int((1 - self.latent[1]) * 300)), 
                          20)
        
        # Draw latent values
        font = pygame.font.Font(None, 24)
        y_offset = 350
        for i, name in enumerate(['front-back', 'high-low', 'rounded', 
                                  'voice-breath', 'manner', 'vowel-cons', 
                                  'pitch', 'intensity']):
            text = font.render(f"{name}: {self.latent[i]:.2f}", True, (255, 255, 255))
            self.screen.blit(text, (10, y_offset + i * 25))
        
        # Instructions
        instructions = [
            "Mouse: Control tongue position (X=front-back, Y=high-low)",
            "R/U: Rounded/Unrounded",
            "V: Voice ON, B: Breath/unvoiced",
            "Up/Down: Pitch",
            "Space: Intensity",
            "ESC: Quit"
        ]
        y_offset = 20
        for instruction in instructions:
            text = font.render(instruction, True, (200, 200, 200))
            self.screen.blit(text, (420, y_offset))
            y_offset += 30
        
        pygame.display.flip()
    
    def run(self):
        """Main loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.update_latent()
            self.synthesize_and_buffer()
            self.draw_ui()
            
            clock.tick(100)  # 100 Hz update rate
        
        self.audio_stream.stop()
        pygame.quit()

if __name__ == '__main__':
    playground = LatentPlayground()
    playground.run()
```

### Phase 4: Performance Optimization

```python
# examples/benchmark.py

import torch
import time
from latent_synth import HybridDDSPSynthesizer

def benchmark_synthesizer():
    synth = HybridDDSPSynthesizer()
    synth.eval()
    
    # Generate random latent sequence
    batch_size = 1
    n_frames = 100  # 1 second at 100Hz
    latent_seq = torch.randn(batch_size, n_frames, 8)
    latent_seq[..., 7] = 0.7  # Set intensity
    
    # Warmup
    for _ in range(10):
        _ = synth(latent_seq)
    
    # Benchmark
    n_iterations = 100
    start_time = time.time()
    
    for _ in range(n_iterations):
        with torch.no_grad():
            _ = synth(latent_seq)
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / n_iterations
    audio_duration = n_frames / 100.0  # seconds
    rtf = avg_time / audio_duration
    
    print(f"Benchmark Results:")
    print(f"  Average synthesis time: {avg_time*1000:.2f} ms")
    print(f"  Audio duration: {audio_duration:.2f} s")
    print(f"  Real-time factor: {rtf:.4f}")
    print(f"  Target RTF: <0.1 (10x real-time)")
    
    if rtf < 0.1:
        print("✓ PASSED: Real-time capable")
    else:
        print("✗ FAILED: Too slow for real-time")
    
    return rtf

if __name__ == '__main__':
    benchmark_synthesizer()
```

## Testing Strategy

Execute tests in this order:

1. **Unit tests**: Test individual DDSP components
   ```bash
   pytest tests/test_ddsp_core.py -v
   ```

2. **Vowel synthesis**: Verify basic functionality
   ```bash
   python tests/test_vowels.py
   # Listen to output files in test_output/
   ```

3. **Consonant synthesis**: Test parallel noise branch
   ```bash
   python tests/test_consonants.py
   ```

4. **CV transitions**: Test dynamic articulation
   ```bash
   python tests/test_consonants.py::test_cv_transitions
   ```

5. **Performance**: Ensure real-time capability
   ```bash
   python examples/benchmark.py
   ```

6. **Interactive**: Manual testing and validation
   ```bash
   python examples/playground.py
   ```

## Success Criteria

The implementation should achieve:

- [ ] All vowels synthesize with recognizable quality (even if robotic)
- [ ] Consonants produce appropriate noise characteristics
- [ ] CV transitions are smooth (no clicks/pops)
- [ ] Real-time factor < 0.1 (10x faster than real-time)
- [ ] Entire pipeline is differentiable (no .numpy() in forward pass)
- [ ] Interactive playground runs without audio dropouts

## Notes for Implementation

1. **Start simple**: Get harmonic synthesis working first, add noise later
2. **Test incrementally**: Synthesize a single vowel before moving to full system
3. **Listen carefully**: Your ears are the best debugger for audio
4. **Watch for artifacts**: Phase discontinuities, clicks, and clipping are common issues
5. **Gradients matter**: Even if not training yet, maintain differentiability

## After Phase 1 is Working

Once basic synthesis works, we'll add:

- Plosive burst generation (5-20ms noise bursts for /p/, /t/, /k/)
- Nasal pole-zero pairs (for /m/, /n/)
- Aspiration noise (for voiceless stops)
- Better formant transitions (coarticulation effects)
- Training pipeline to optimize decoder from (latent, audio) pairs

But first, get vowels working reliably.

## Questions to Address During Implementation

1. Are formant frequencies mapping correctly from latents?
2. Is harmonic amplitude rolloff natural (should decrease with frequency)?
3. Is noise properly filtered for consonant identity?
4. Are phase discontinuities causing artifacts?
5. Is real-time performance actually achieved?

Build this system incrementally, test each component, and listen to the results carefully.
```