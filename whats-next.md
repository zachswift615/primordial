# Session Handoff: Speech Production Learning

**Created:** 2025-12-02
**Purpose:** Continue training the speaking agent in a fresh context

---

## What We Built Today

### Phase 2: Self-Listening Speech Production
Agent produces phonemes → hears itself via TTS → adjusts to match target. Like a baby babbling to learn speech.

**Results:**
| Test | Phonemes | Epochs | Match Rate |
|------|----------|--------|------------|
| Vowels | IY,AA,EH,UW,AH | 30 | **100%** |
| +Consonants | +B,P,D,T,M,S,L,N | 100 | **100%** |

Key: Distinguished voiced/unvoiced pairs (B↔P, D↔T) - single dimension flip in 6D latent space.

### Architecture
```
Audio → CNN Encoder → Fourier Mixing → Production Head → 6D Latent
                                                            ↓
                                                    Snap to nearest anchor
                                                            ↓
                                                    Piper TTS → Audio
                                                            ↓
                                                Perception Head classifies
                                                            ↓
                                                   Loss → Update weights
```

---

## Continue Training

### Resume from 10-phoneme checkpoint:
```bash
python -m primordial.scripts.train_production_interactive \
  --checkpoint checkpoints/production/production_best.pt \
  --epochs 300 \
  --play-every 30 \
  --phonemes IY,IH,EY,EH,AE,AA,AH,AO,OW,UW,UH,ER,AY,AW,OY,B,P,D,T,G,K,M,N,NG,F,V,S,Z,SH,TH,DH,L,R,W,Y,HH,CH,JH
```

### Or start fresh on all 41:
```bash
python -m primordial.scripts.train_production_interactive \
  --epochs 300 \
  --play-every 30 \
  --phonemes IY,IH,EY,EH,AE,AA,AH,AO,OW,UW,UH,ER,AY,AW,OY,B,P,D,T,G,K,M,N,NG,F,V,S,Z,SH,TH,DH,L,R,W,Y,HH,CH,JH
```

---

## Key Files

| File | Purpose |
|------|---------|
| `primordial/speech/latent.py` | 6D phoneme anchors (articulatory features) |
| `primordial/scripts/train_production_interactive.py` | Training with audio playback |
| `primordial/scripts/decode_latent.py` | Interpret model outputs |
| `docs/plans/2025-12-02-phase2-speech-production.md` | Full design doc |
| `docs/SESSION-2025-12-02.md` | Session summary |
| `checkpoints/production/production_best.pt` | Best 10-phoneme model |

---

## Latent Space Design

6 dimensions based on articulatory phonetics:

| Dim | Feature | Example |
|-----|---------|---------|
| 0 | Front-Back | IY (front) ↔ UW (back) |
| 1 | High-Low | IY (high) ↔ AA (low) |
| 2 | Rounded | IY (unrounded) ↔ UW (rounded) |
| 3 | Voiced | P (unvoiced) ↔ B (voiced) |
| 4 | Manner | P (stop) ↔ S (fricative) ↔ M (nasal) |
| 5 | Type | Vowels (-1) ↔ Consonants (+1) |

---

## Technical Notes

1. **Audio pops/clicks** - TTS output lacks fade in/out. Model learns despite this.
2. **MPS issues** - Complex tensor ops fail on Apple Silicon. Use `--device cpu`.
3. **Sample rates** - Piper: 22050Hz, Model: 16000Hz. Resampling handled.
4. **Not progressive** - Each run starts fresh unless `--checkpoint` used.

---

## Future Work

1. Fix audio artifacts (add envelope to TTS output)
2. Sequence production (syllables → words)
3. Continuous latent decoder (smooth inter-phoneme sounds)
4. Cross-modal grounding (sounds ↔ meanings)
5. Run overnight on full 41 phonemes
