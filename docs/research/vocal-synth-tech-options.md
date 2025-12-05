# Real-Time Voice Synthesis for 8D Articulatory Control

Combining differentiable DSP with source-filter neural networks offers the most promising path to high-quality, real-time human voice synthesis from an 8-dimensional articulatory latent space. **DDSP-based architectures achieve 15-600 MFLOPS at MOS scores above 4.0**, while physical waveguide models provide interpretable, consonant-capable alternatives at ~110 operations per sample. Your previous formant synthesis failed because it lacked the parallel excitation branch essential for consonants—solving this requires either a Klatt-style hybrid architecture or neural source-filter models that implicitly learn these dynamics.

## Physical modeling delivers interpretable articulatory control

Digital waveguide synthesis, pioneered by Kelly and Lochbaum in 1962 and refined into the framework underlying Logic Pro's Sculpture, models the vocal tract as concatenated cylindrical tubes with bidirectional pressure wave propagation. Each tube junction introduces reflection coefficients **r = (A₂ - A₁)/(A₂ + A₁)** based on adjacent cross-sectional areas. At 44.1kHz sampling, a 17cm vocal tract requires approximately 22 tube sections, yielding only ~100 multiply-adds per sample—trivially real-time on any modern CPU.

The **Liljencrants-Fant (LF) model** dominates glottal source synthesis, parameterized by fundamental period T₀, maximum excitation time Tₑ, peak flow time Tₚ, and return phase duration Tₐ. These reduce to perceptually meaningful controls: open quotient (Oq) correlating with breathiness via H1-H2 spectral difference, and speed quotient (Sq) controlling asymmetry. For your 8D latent space, the voice-breath dimension maps directly to Oq while pitch controls T₀.

**Pink Trombone** demonstrates that browser-based JavaScript can achieve real-time waveguide synthesis with parameters that map cleanly to articulatory dimensions: `tongue.index` (front-back), `tongue.diameter` (constriction degree), and `tenseness` (voice-breath ratio). The MIT-licensed codebase at github.com/zakaton/Pink-Trombone provides an accessible starting point. VocalTractLab offers a more sophisticated 33-parameter articulatory model derived from MRI data, though its computational cost currently limits real-time synthesis to near-real-time on desktop hardware.

| Latent Dimension | Physical Model Mapping |
|------------------|------------------------|
| front-back | Tongue body position (Maeda parameter 2) |
| high-low | Jaw opening + tongue body shape |
| rounded | Lip protrusion + aperture |
| voice-breath | LF Oq + aspiration noise ratio |
| manner | Constriction degree at articulation point |
| vowel-cons | Velum opening (nasal coupling) |
| pitch | Fundamental frequency → glottal period |
| intensity | Subglottal pressure / pulse amplitude |

## Neural vocoders now achieve sub-1% CPU utilization

**FARGAN** (Framewise Autoregressive GAN, 2024) represents the current frontier for CPU-efficient neural synthesis at **600 MFLOPS**—five times lighter than optimized LPCNet and 110 times lighter than CARGAN while achieving statistically equivalent quality to HiFi-GAN V1. Operating on 2.5ms subframes with 820k parameters, FARGAN consumes less than 1% of a modern laptop CPU core at real-time. The implementation lives in the Opus codec repository at gitlab.xiph.org/xiph/opus.

For maximum efficiency, the **Ultra-Lightweight DDSP Vocoder** achieves MOS 4.36 at just **15 MFLOPS**—340 times fewer FLOPs than MB-MelGAN. This approach eliminates learnable vocoder parameters entirely, instead jointly training an acoustic model to produce features that a fixed DSP vocoder can synthesize directly. The result is RTF 0.003 (vocoder only) on a 2GHz Intel Xeon, leaving substantial headroom for your 8D latent decoder network.

DDSP architectures naturally align with articulatory control because they decompose synthesis into source (harmonic additive synthesizer) and filter (subtractive noise shaper) components—mirroring the glottis-tract separation in physical models. Google Magenta's DDSP (github.com/magenta/ddsp) demonstrates end-to-end differentiability through differentiable oscillators and time-varying filters, enabling gradient flow through your self-listening loop. The conditioning pathway accepts continuous F0 and loudness at frame rate, easily extended to accept your 8D latent via an encoder network.

**Vocos** (github.com/gemelo-ai/vocos) offers an alternative path: predicting STFT coefficients and using iSTFT for upsampling eliminates transposed convolutions entirely, achieving order-of-magnitude speedups over time-domain vocoders while maintaining state-of-the-art quality. The architecture's isotropic ConvNeXt backbone maintains constant temporal resolution, simplifying conditioning from variable-rate control inputs.

## Consonant synthesis requires parallel noise excitation

Your formant synthesizer failed because standard cascade configurations model only poles—appropriate for vowels where the vocal tract transfer function is all-pole with glottal excitation. **Consonants introduce zeros** because the noise source sits at the supraglottal constriction rather than the glottis, and the front cavity between source and lips creates both resonances and antiresonances.

The Klatt synthesizer solved this with a **hybrid cascade/parallel architecture**: sonorants (vowels, nasals, liquids) route through the cascade branch with glottal excitation, while fricatives and plosive bursts route through the parallel branch where individual formant amplitudes can be controlled independently. This dual-path structure is essential for natural consonants.

Plosives require three distinct phases: closure (silence or low-frequency voicebar for voiced stops), burst (5-20ms noise shaped by front cavity resonance), and aspiration (noise through cascade for voiceless stops). Voice onset time ranges from 0-20ms for English voiced stops to 40-80ms for voiceless, increasing for posterior articulations (bilabial < alveolar < velar). The burst spectral shape depends critically on front cavity length: bilabials produce diffuse spectra, alveolars concentrate energy above 3kHz, and velars peak around 1.5-2.5kHz.

Fricatives split into obstacle sources (strident: /s/, /ʃ/) where turbulent jets impinge on teeth, and wall sources (non-strident: /f/, /θ/) with lower intensity. Noise amplitude scales with **U³** (airflow velocity cubed). Voiced fricatives require dual simultaneous excitation: periodic glottal pulses through the cascade branch with 50% amplitude modulation, plus frication noise through the parallel branch.

Nasals introduce pole-zero pairs from the nasal cavity side-branch: a fixed nasal pole around 270Hz and a variable antiresonance (nasal zero) that moves between 400-700Hz depending on oral cavity configuration. During non-nasalized sounds, setting FNP = FNZ achieves cancellation; opening the velum separates them, introducing the characteristic nasal quality through first formant amplitude reduction.

## Hybrid neural-physical architectures maximize controllability

**Neural Source-Filter (NSF) models** from NII Japan combine classical source-filter theory with neural networks: a source module generates sine-based excitation from F0, while a filter module uses non-autoregressive dilated convolutions to transform excitation to waveform. Running 100x faster than WaveNet with comparable quality, NSF implementations at github.com/nii-yamagishilab offer a BSD-licensed starting point.

**HiFi-Glot** (2024) directly addresses articulatory control by implementing differentiable resonant filters with neural glottal waveform generation. The architecture accepts 9 control parameters—F0, voicing, F1-F4, spectral tilt, spectral centroid, energy—closely matching your 8D latent dimensionality. Perceptual quality exceeds Praat while maintaining full differentiability for end-to-end training.

**SiFi-GAN** (Source-filter HiFi-GAN) integrates source-filter theory into the HiFi-GAN framework, achieving RTF 0.08 on CPU with robust F0 controllability. The hierarchical conditioning of filter network on source excitation enables pitch manipulation far outside training ranges.

The **SawSing** family of DDSP vocoders (github.com/YatingMusic/ddsp-singing-vocoders) demonstrates differentiable linear time-varying FIR filtering with neural coefficient prediction. The SawSinSub variant uses anti-aliased harmonic synthesis while CombSub preserves timbre through combtooth excitation—both train on just 3 hours of data while enforcing phase continuity to eliminate glitch artifacts.

For your system, the **GOLF** architecture (github.com/iamycy/golf) combines differentiable LPC estimation with glottal-flow-inspired wavetable synthesis, offering explicit decomposition into source and filter components while remaining fully differentiable.

## Real-time constraints are achievable with proper architecture

At 22050Hz with 10ms latency target, you have a 220-sample budget per processing frame. Frame-based vocoders naturally align with your 100Hz control rate—both operate on ~10ms windows. The critical constraint is algorithmic latency, not computational throughput.

| Architecture | MFLOPS | RTF (CPU) | Latency |
|--------------|--------|-----------|---------|
| FARGAN | 600 | <0.01 | 2.5ms subframes |
| Ultra-light DDSP | 15 | 0.003 | Frame-based |
| LPCNet | 3,000 | 0.1-0.3 | 40ms frames |
| HiFi-GAN V3 | 3,000 | <0.075 | Frame-based |
| Kelly-Lochbaum waveguide | ~5 | <0.001 | Sample-based |

The physical waveguide approach offers the lowest latency since it operates sample-by-sample without frame buffering—Pink Trombone achieves <10ms in JavaScript via Web Audio Worklet. However, quality for consonants requires careful noise source modeling.

For neural approaches, SIMD optimization (AVX2/NEON) provides 4-8x speedups, while INT8 quantization adds another 2-4x. LPCNet's optimized implementation uses 8-bit dot products specifically to achieve real-time on mobile CPUs.

## Recommended architecture for your 8D system

The optimal design combines **DDSP-style source-filter decomposition** with **differentiable formant filtering**:

**Stage 1: Latent Decoder** (10-20k parameters MLP)
Maps 8D latent → synthesis parameters: F0, voicing, F1-F4, noise amplitude, spectral tilt, nasal coupling

**Stage 2: Source Generation**
- Voiced: LF-model glottal pulses or learned neural glottal waveform
- Unvoiced: Gaussian noise with position-dependent filtering
- Mixed: Amplitude-modulated combination for voiced fricatives

**Stage 3: Filter Application**
- Cascade all-pole filter (4-5 resonators) for sonorants
- Parallel formant bank with independent amplitude control for obstruents
- Differentiable anti-resonator for nasal zero

**Stage 4: Waveform Synthesis**
Either direct sample generation (waveguide-style) or iSTFT-based upsampling (Vocos-style)

This architecture remains fully differentiable: gradients flow through filter coefficients, source amplitude, and pitch period back to the 8D latent space. Train on parallel articulatory-acoustic data (EMA recordings with audio) or use self-supervised learning with articulatory analysis as a bottleneck.

## Conclusion

Your 8D articulatory latent space aligns remarkably well with established synthesis parameterizations—the Maeda model derived 7 statistically optimal articulatory parameters from X-ray data, explaining 88% of tongue shape variance. Neural prosthetics research demonstrates that 7-10 articulatory parameters suffice for intelligible speech with 100-200Hz control rates.

**Start with DDSP-SVC** (github.com/yxlllc/DDSP-SVC) as a proven real-time baseline, then extend the conditioning pathway to accept your 8D latent rather than extracted acoustic features. For consonant quality, augment with explicit noise source modeling following Klatt's parallel branch design—or train on sufficient consonant-vowel data for the neural network to implicitly learn these dynamics. The differentiable filter coefficients enable gradient flow through the self-listening loop while physical interpretability ensures the 8D space maps meaningfully to articulation.

The computational budget is generous: at 600 MFLOPS for FARGAN or 15 MFLOPS for ultra-lightweight DDSP, you have substantial headroom for your latent decoder network while remaining well under 10ms latency on any modern CPU.