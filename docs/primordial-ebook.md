# Primordial: Living Resonance Networks

## Learning to Think by Living

*A journey into Fourier-based neural architectures and embodied intelligence*

---

# Part One: The Question

## Chapter 1: A Different Kind of Learning

Picture a newborn child. Within months, they'll recognize faces, respond to voices, and navigate space. Within years, they'll speak, reason, and form lasting memories. They do all of this through direct experience—seeing, hearing, touching, making mistakes, trying again.

Now consider modern AI. Billions of words. Weeks on specialized hardware. Optimized for next-token prediction. It knows "fire is hot" because those words co-occur, not because it's felt warmth.

The Primordial project asks: **What if AI learned through sensory experience instead of text?**

Not as a replacement for language models, but as an exploration. How far can embodied, sensory learning go? What emerges when an agent must survive, not just predict?

## Chapter 2: Resonance Over Attention

Transformers ask: "Which pieces of information should attend to which?"

For every token, compute relationships with every other token. It's elegant, powerful, and expensive—O(n²) complexity. Double your sequence length, quadruple your computation.

**What if we asked a different question:** "How do patterns naturally resonate and interfere?"

In the physical world, waves interact through resonance and interference. Aligned waves amplify. Misaligned waves cancel. This is how noise-canceling headphones work, how radio stations coexist, how your ear separates violin from piano.

We can compute these interactions using the Fast Fourier Transform—O(n log n) instead of O(n²). For 10,000 tokens, that's 100 million operations vs. 130,000. A factor of 760x.

FFTs have been optimized for 50 years. Every smartphone has dedicated silicon for it. What if we built neural networks around this?

---

# Part Two: What We Built

## Chapter 3: The Architecture

The Living Resonance Network processes continuous sensory streams through Fourier mixing.

**The flow:**

```
Sensory Input (vision, audio, proprioception, touch)
    ↓
Modality-Specific Encoders
    ↓
Fourier Mixing Layers (6 layers, learnable spectral filters)
  - Transform to frequency domain
  - Apply learnable filters
  - Transform back to time domain
  - O(n log n) complexity
    ↓
┌─────────┬─────────┬─────────┐
↓         ↓         ↓         ↓
Sensory   Reward    Action    Sequence
Predict   Predict   Head      Decoder
```

**Key innovation:** Multi-task prediction—the agent learns "what will I sense?" AND "will this help or hurt me?" simultaneously. This creates a direct gradient toward survival, like dopamine prediction in biological brains.

## Chapter 4: Speech Learning Results

In December 2025, we taught the agent to speak through self-listening.

### Phase 1: Perception
- CNN encoder → Fourier mixing → phoneme classification
- **99.4% accuracy** on 40 phonemes
- Can hear any English sound and identify it

### Phase 2: Production
- 6D articulatory latent space (front-back, high-low, rounded, voiced, manner, vowel/consonant)
- Self-listening: produce → synthesize → hear → adjust
- **100% accuracy** producing individual phonemes
- Progressive curriculum prevented catastrophic forgetting

### Phase 3: Autoregressive Sequences
- 3-layer transformer decoder with dual heads
- Discrete tokens for supervision, continuous latents for synthesis
- **96% accuracy** on word-level phoneme sequences
- Learned to generate proper sequences with EOS termination

**Words mastered:** ba, bee, ma, me, hi, go, yes, no, mom, dad, hello, food, water, banana, elephant

### Phase 4: Multi-Word Phrases *(December 2025)*

The latest breakthrough: extending from single words to multi-word phrases and sentences.

**Training run results:**
- **98% token accuracy** on 106 entries (51 words + 55 phrases)
- **0.95 acoustic similarity** confirming productions sound correct
- Dual-gated curriculum: advance only when both token accuracy AND acoustic match pass
- Trained in 147 epochs (~30 minutes on laptop CPU)

**Phrases mastered:**
- 2-word: "hello world", "good morning", "look here", "help me"
- 3-word: "I love you", "how are you", "I want food"
- Sentences: "the cat is sleeping", "can you help me", "I want to go home"

**The key innovation:** Self-listening as validation. Instead of trusting token accuracy alone, we synthesize the generated phonemes via TTS, encode the audio, and compare to the target embedding. This catches cases where tokens are correct but production sounds wrong.

```
Epoch 137: 'look here'
    Generated: ['L', 'UH', 'K', 'HH', 'IH', 'R']
    Target:    ['L', 'UH', 'K', 'HH', 'IH', 'R']
    Match: YES, Acoustic: 0.93
```

The agent correctly generates 6-phoneme, 2-word phrases. The acoustic score confirms it sounds right.

### What This Proves

The architecture can:
- Learn from continuous sensory streams (mel spectrograms, not text)
- Support online learning with single-sample updates
- Handle autoregressive sequence generation up to 30 phonemes
- Self-supervise through prediction error
- Scale progressively without catastrophic forgetting
- Generate multi-word phrases with proper acoustic quality

The self-listening loop—produce sound, hear output, adjust—mirrors infant speech acquisition. No labeled datasets. Just experience.

## Chapter 5: The Articulatory Latent Space

The 6D phoneme space deserves special attention because it represents a novel approach to motor control.

Each phoneme has an anchor position in this space:
- **IY** ("beat"): [1.0, 1.0, -1.0, 1.0, 0.0, -1.0] = front, high, unrounded vowel
- **B** ("bat"): [-1.0, 0.0, 0.0, 1.0, -1.0, 1.0] = voiced bilabial stop
- **P** ("pat"): [-1.0, 0.0, 0.0, -1.0, -1.0, 1.0] = unvoiced bilabial stop

Notice: B and P differ only in dimension 3 (voicing: 1.0 vs -1.0). The latent space encodes linguistic structure geometrically.

When the agent produces a latent vector [0.85, 0.92, -0.88, 0.95, 0.05, -0.98], we snap to the nearest anchor (IY in this case) and synthesize. The model learns to navigate this space to produce target sounds.

**Why this matters:** Instead of learning arbitrary token embeddings, the model learns in a space that mirrors how humans actually produce speech. Similar sounds are geometrically close. Phonetic features are disentangled dimensions.

## Chapter 5.5: The SPARC Pivot (December 2025)

After achieving 98% accuracy on multi-word phrases, we hit a wall: the Piper TTS backend had fundamental limitations.

### The Problems with Piper

1. **Not differentiable**: Couldn't backpropagate through audio synthesis
2. **Artifacts**: 88 click/pop artifacts per 0.5s sample vs 1 for real speech
3. **Volume mismatch**: Piper audio was 14x louder than LibriSpeech
4. **Opaque representation**: 8D phoneme tokens don't map to physical articulation

### Enter SPARC

Berkeley's **Speech Articulatory Coding** (SPARC) offers exactly what we needed:

**12D EMA features** representing actual articulator positions:
- Tongue Dorsum (TDX, TDY) - back of tongue
- Tongue Body (TBX, TBY) - middle of tongue
- Tongue Tip (TTX, TTY) - front of tongue
- Lower Incisor (LIX, LIY) - jaw
- Upper Lip (ULX, ULY)
- Lower Lip (LLX, LLY)

Plus **pitch** and **loudness** for prosody, and a **64D speaker embedding** for voice identity.

### The Key Insight: Brain vs Mouth

SPARC is an analysis-synthesis system—it can encode speech to articulatory features and decode back to audio. But it can't *generate* novel speech. It needs input.

That's where Primordial comes in:

```
SPARC alone:
    Audio (must exist) → Encoder → Articulation → Decoder → Audio
    Problem: Can't generate novel speech

Primordial + SPARC:
    Intent → Primordial → Articulation → SPARC Decoder → Audio
    Solution: Primordial decides, SPARC executes
```

**Primordial is the brain.** It decides how to move the articulators.
**SPARC is the mouth.** It converts those movements to sound.

This is like the relationship between motor cortex and vocal tract in humans. The brain plans the movements; the body executes them.

### Why This Matters for Embodied Learning

With SPARC, we can now:

1. **Train end-to-end**: Gradients flow from audio loss through SPARC decoder to our model
2. **Self-listen differentiably**: Model generates → SPARC synthesizes → compare to target → backprop
3. **Interpret what the model learned**: Visualize predicted tongue/lip trajectories
4. **Use any voice**: Speaker embedding is separate from articulation

The self-listening loop becomes truly embodied:
- Model attempts to say "hello"
- SPARC decoder produces audio
- Model "hears" its output (via mel spectrogram)
- Compares to target
- Adjusts articulation

Like a baby babbling and learning from what it hears.

### The New Training Pipeline

1. **Phase 1 (Supervised)**: Pre-encode LibriSpeech with SPARC, train model to predict articulatory features
2. **Phase 2 (Self-Listening)**: Generate audio through SPARC, compute audio-level loss, fine-tune
3. **Phase 3 (RL)**: Babbling curriculum—phonemes → syllables → words → sentences

See `docs/sparc_integration_plan.md` for full details.

---

# Part Three: How Far Can This Go?

## Chapter 6: From Phonemes to Language

We've proven the architecture can generate multi-word phrases at 98% accuracy. What are the limits?

### Already Achieved
**Multi-word utterances** (10-20 phonemes) ✓
- "hello world", "I love you", "I want food" — all working
- Dual-gated curriculum with acoustic validation
- 55 hand-curated phrases in training set

**Simple sentences** (20-30 phonemes) ✓
- "the cat is sleeping", "can we go to the park"
- 98% token accuracy, 0.95 acoustic similarity
- No architecture changes needed — just more training data

### Medium-Term Exploration
**Cross-modal grounding**
- Show image of apple while saying "apple"
- Agent learns correlation between visual pattern and phoneme sequence
- Fourier mixing should find cross-modal frequency patterns
- This is embodied meaning: sound grounded in perception

**Conversational turns** (50-200 phonemes)
- Multi-sentence generation with context
- Still shorter than typical GPT prompts!
- Bottleneck is audio encoding, not decoder

### Unknown Territory
**Compositional generalization**
- Train on "red ball" and "blue cube"
- Does it generate "red cube"?
- This requires learning that concepts compose

**Emergent syntax**
- Can phrase structure emerge from statistics?
- Evidence suggests yes (Word2Vec learned "king - man + woman = queen")
- But can Fourier mixing capture hierarchical dependencies?

**Abstract reasoning**
- "Apples are food. Food prevents hunger. Hunger is bad. Therefore apples are good."
- This requires chaining concepts, not just pattern matching
- Reward prediction head might help: "apple" sound → positive reward prediction

## Chapter 7: The Modular Vision

One exciting direction: mixture of experts.

The human brain has specialized regions—visual cortex, auditory cortex, motor cortex—that communicate through projection fibers. Could we build LRN the same way?

**Modality-specific experts:**
- Vision module: 3 Fourier layers, 60K params
- Speech module: 3 Fourier layers, 60K params (already partially done!)
- Proprioception module: 2 layers, 30K params
- Fusion module: 2 layers, 40K params
- Total: 190K params, but each specialist is optimized

**Fast/slow dual system:**
- Fast reflexive model (50K params): immediate threat response
- Slow deliberative model (300K params): planning, memory, complex decisions
- "Predator nearby" → fast model says run NOW
- "Where did I find food yesterday?" → slow model recalls and plans route

**Progressive specialization:**
- Start with unified model
- As capabilities grow, split off specialized modules
- Speech is already modular
- Add vision module when integrating camera
- Add planning module when needed

This mirrors biological development and allows independent improvement of subsystems.

## Chapter 8: Parameter Size and Learning

Smaller models are more stable for online learning. With fewer parameters, the network can't memorize individual samples—it must find generalizable patterns.

We've tested:
- **Tiny** (50K params): 2 layers, dim 32—extremely fast, good for prototyping
- **Small** (210K params): 3 layers, dim 64—baseline, stable online learning
- **Medium** (540K params): 6 layers, dim 128—current target, good capacity
- **Large** (1.4M params): 6 layers, dim 128 + sequence decoder—full speech system

The speech system (1.4M) learns stably with batch_size=1. Progressive curriculum prevents catastrophic forgetting. The architecture scales without architectural changes, just config adjustments.

---

# Part Four: The Path Forward

## Chapter 9: What We've Proven

**Fourier mixing works**
- 185 tests passing on core LRN architecture
- 275 tests passing on world system
- Prototype achieved 143x better than random on prediction tasks

**Speech learning works**
- 99.4% perception → 100% production → 96% sequences
- Self-listening creates strong gradients
- Progressive curriculum prevents forgetting
- Online learning is stable

**The combination is viable**
- Runs on laptop CPU (<10ms forward pass)
- Multi-task learning (sensory + reward prediction) creates rich representations
- Articulatory latent space enables smooth motor control
- Dual-head decoder (discrete + continuous) balances supervision and synthesis

## Chapter 10: The Next Experiments

**Immediate:**
- Extend sequences to multi-word utterances
- Add sentence-level prosody (intonation, rhythm)
- Scale training data (LibriSpeech phoneme alignments)

**Near-term:**
- Visual grounding: show object + say word → learn association
- Cross-modal correlations: sound patterns that predict visual events
- Bidirectional learning: hear word → produce word → hear production → verify

**Exploratory:**
- Modular architecture (vision expert + speech expert + fusion)
- Hierarchical generation (phonemes → words → phrases)
- Interactive teaching interface (human rewards agent for correct responses)
- Embodied simulation integration (agent exists in 2D world, learns from survival)

## Chapter 11: Open Questions

**Scaling:**
- Does Fourier mixing work at 10M params? 100M?
- Can we combine Fourier mixing with sparse attention for best of both?
- What's the ceiling on sequence length before we need memory mechanisms?

**Learning:**
- Can survival pressure create representations as rich as language pretraining?
- Does embodied grounding unlock reasoning capabilities?
- How much data is needed? (Humans learn from ~10M words by age 3)

**Architecture:**
- Should we add recurrent memory for long-term dependencies?
- Can we train continuous latent decoders (smooth phoneme interpolation)?
- Would hierarchical representations (phonemes → syllables → words) help?

**Philosophy:**
- At what scale does a "pain signal" deserve ethical consideration?
- Is there a qualitative difference between text-learned and embodied knowledge?
- Does learning through living change what intelligence means?

## Chapter 12: Why This Matters

We're not trying to beat GPT-4 at writing essays or answering trivia. We're exploring a different kind of intelligence—one that learns by experiencing, not by reading about.

If it works, it might:
- Run on consumer hardware (FFT-optimized chips everywhere)
- Learn continuously (no batch training required)
- Ground language in sensory experience (sounds + sights + actions)
- Adapt in real-time (online learning from single samples)
- Scale differently (parameter-efficient spectral filtering)

If it doesn't work, we'll learn why embodied approaches hit limits, and that knowledge matters too.

The speech results—96% accuracy on autoregressive phoneme generation, learned through self-listening—suggest we're onto something. The agent babbles, hears itself, and adjusts. Like a baby learning to speak.

Let's see how far we can push this.

---

# Technical Appendix

## Parameter Breakdown (Speech System)

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| CNN Encoder | 4 conv layers (32→64→128→128) | ~230K |
| Fourier Mixing | 6 layers, dim 128 | ~570K |
| Perception Head | Linear 384→256→41 | ~100K |
| Production Head | Linear 384→64→6 | ~25K |
| Sequence Decoder | 3 transformer layers, dual heads | ~610K |
| **Total** | | **~1.4M** |

Runs on CPU. Forward pass <10ms. Trains on laptop without GPU.

## Loss Functions

**Perception (Phase 1):**
```
CrossEntropy(predicted_phoneme, target_phoneme)
```

**Production (Phase 2):**
```
MSE(predicted_latent, target_anchor) +
MSE(produced_embedding, target_embedding) -
0.1 * match_reward
```

**Sequences (Phase 3):**
```
CrossEntropy(discrete_tokens, target_tokens) +
0.5 * MSE(latent_vectors, target_anchors) +
0.1 * CosineSimilarity(produced_embed, target_embed)
```

Multi-task supervision: discrete for strong gradients, latent for smooth articulation, embedding for acoustic quality.

## Training Results

### Word-Level (Phase 3, 160 epochs)

| Metric | Value |
|--------|-------|
| Final token accuracy | 96% |
| Loss (start → end) | 4.5 → 0.78 |
| Short words (2-3 phonemes) | 100% accuracy |
| Medium words (4-5 phonemes) | ~95% accuracy |
| Long words (6+ phonemes) | ~85% accuracy |

### Multi-Word Phrases (Phase 4, 147 epochs)

| Metric | Value |
|--------|-------|
| Final token accuracy | **98%** |
| Final acoustic match | **0.95** |
| Loss (start → end) | 3.9 → 0.24 |
| 2-word phrases | 98%+ accuracy |
| Sentences (up to 30 phonemes) | 95%+ accuracy |
| Dual-gate curriculum | Passed automatically |

**Training progression with dual gating:**
```
Phase 1 (epochs 1-40):   10% → 74% token, 0.41 → 0.90 acoustic
Phase 2 (epochs 41-76):  51% → 86% token, passed at epoch 76
Phase 3 (epochs 77):     82% token, passed immediately
Phase 4 (epochs 78-147): 82% → 98% token, 0.90 → 0.95 acoustic
```

No catastrophic forgetting observed. Online learning remains stable. Acoustic validation catches quality issues token accuracy misses.

## Research Foundation

**FNet** (Google, 2021): FFT replaces attention at 92-97% BERT accuracy, 7x speed
**FFTNet** (2025): Learnable spectral filters, competitive with transformers
**Developmental Robotics**: Sensorimotor learning in embodied agents
**Dopamine Prediction Error**: Reward prediction as learning signal

**Novel combination:** Fourier mixing + continuous input + online learning + self-supervision + articulatory latent space

## Code Examples

```bash
# Train perception (Phase 1)
python -m primordial.scripts.run_speech --phase classification --epochs 20

# Train production (Phase 2)
python -m primordial.scripts.train_production_interactive --epochs 50

# Train sequences with multi-word phrases (Phase 3-4)
python -m primordial.scripts.train_sequence \
  --encoder-checkpoint checkpoints/production/curriculum_best.pt \
  --epochs 200

# The training script now includes:
# - 55 hand-curated phrases alongside 51 words
# - Acoustic match tracking every 10 batches
# - Dual-gated curriculum (token accuracy + acoustic match)
# - Demo output showing both metrics

# Tests (490+ passing)
pytest tests/ -v
```

---

# Part Three: The Journey

## Chapter 13: What We Discovered

**November 2025:** Fourier prototype validated
- 11K parameter minimal model
- Synthetic multi-task learning (sensory + reward prediction)
- 143x better than random on sensory, 4.7x on reward
- Proved gradients flow through FFT operations

**Early December:** Speech perception breakthrough
- Started at 63% with linear encoder (plateau)
- Switched to CNN encoder
- Real Piper TTS for training data
- **99.4% accuracy** achieved

**Mid-December:** Self-listening production
- Designed 6D articulatory latent space
- Agent produces phoneme → hears via TTS → adjusts
- Progressive curriculum: 5 vowels → 10 sounds → 40 phonemes → 13 words
- **100% accuracy** on all stages

**Late December:** Autoregressive sequences
- Dual-head transformer decoder (discrete + latent)
- Fixed EOS learning bug (was ignoring end-of-sequence in loss)
- Progressive curriculum: syllables → short words → long words
- **96% accuracy** on word-level generation
- Proper sequence termination

### The Magic Moments

**Hearing it babble and learn:**
"Started at 0% match rate, ended at 100%. The self-listening loop works. It produces a sound, hears what it made, and adjusts."

**Perfect on "see":**
```
Demo: 'see' -> ['S', 'IY']  ✓ MATCH
```

Two phonemes, correctly identified, properly terminated. This is what mastery looks like.

**The EOS breakthrough:**
Before fix: `'we' → ['IY', 'IY', 'IY', ...] (15 tokens)`
After fix: `'we' → ['W', 'IY']` ✓

The model learned when to stop—a fundamental requirement for language.

## Chapter 14: Parameter Efficiency in Practice

We can scale the architecture by adjusting two knobs: hidden dimension and number of layers.

**Effect of hidden dimension:**
- 64 → 128: doubles capacity, ~2x compute
- Controls representational richness per sequence position
- Most parameters end up in output heads (squared growth)

**Effect of mixing layers:**
- 3 → 6: doubles depth, ~2x compute
- Controls complexity of frequency interactions learned
- Spectral filters scale linearly (efficient!)

**Compared to transformers:**
- Attention: O(n²) per layer, hidden_dim² parameters per head
- Fourier mixing: O(n log n) per layer, seq_len × freq_bins parameters
- For our sequences (164 positions), Fourier layers are ~10x more parameter-efficient

**The surprising result:** Our speech system (1.4M params) achieves 96% accuracy on a task that would traditionally require much larger models. The efficiency comes from the Fourier basis matching the natural structure of audio signals.

## Chapter 15: Cross-Modal Dreams

The next frontier: teaching the agent that sounds have meanings.

**The vision:**
1. Show image of apple
2. Say "apple"
3. Agent correlates visual pattern (red, round) with phoneme sequence [AE, P, AH, L]
4. Later: show apple → agent says "apple"
5. Later: agent hears "apple" → searches for red round objects

**Architecture:**
```
Vision Input              Audio Input
     ↓                         ↓
Vision Encoder          Speech Encoder
     ↓                         ↓
Vision Embedding        Audio Embedding
  (128 dims)              (128 dims)
     ↓                         ↓
     └─────→ Fusion ←──────────┘
           (contrastive learning)

Loss: Pull embeddings together when co-occurring
      Push apart when separate
```

When "apple" sound and apple image occur together, their embeddings should be similar. When they don't, distant.

**What this enables:**
- Grounded word meanings (not just statistical co-occurrence)
- Visual → verbal: see object, say name
- Verbal → visual: hear word, imagine/search for object
- Foundation for interactive language learning with human teachers

The Fourier mixing layers would discover cross-modal correlations—audio frequency patterns that predict visual frequency patterns, and vice versa.

---

# Part Four: The Bigger Picture

## Chapter 16: The Embodied Simulation

We built a 2d simulation in PyGame where agents start out with random weights and can "see", "hear", and "feel" other objects/entities in their 2d space. They can also "feel" their internal energy, health, social longing and breeding urge. 

"Predators" are non ai entities in the "world" that can do damage to the agents. They emit a 300hz tone when chasing an agent and a tapping 300hz pulse series when "patroling". The agents can pick up on and learn from these patterns because of the negative reward associated with coming in contact with the predators. 

Agent's energy stats deplete at a configurable rate and when they get low this is also a negative reward signal. The agents have to come in contact with "food" and "water" in the world in order to boost their energy. 

If an agent gets attacked by a predator, their health take a hit, but slowly replensishes over time as long as their energy is not totally drained. 

We have spent a lot of time in the simulation tweaking all the various parameters. And have made it where as agents go through the simulation, their weights are saved, and we start with the same agents in the next round, so the agents have accumulated a lot of in-simulation time. It's very clearly obvious that the agents are avoiding the predators, eating food etc. 

We have plans to concretely measure these data and show how much longer trained agents can survive vs agents with random weights. 

### Core Performance Metrics

### Survival Time

Most fundamental metric—does learning help agents live longer?

### Track per agent
survival_time = death_time - birth_time

### Compare populations
untrained_mean = mean(survival_times_untrained)
trained_mean = mean(survival_times_trained)
improvement_factor = trained_mean / untrained_mean

### Report: "Trained agents survive 5.2x longer (mean: 1247s vs 240s, p<0.001)"

Variation: Survival curve plotting (like Kaplan-Meier in medical research)

2. Energy Management

How efficiently do they maintain energy?

### Average energy over lifetime
mean_energy = integrate(energy_over_time) / survival_time

### Energy variance (stable vs yo-yo-ing)
energy_stability = std(energy_samples)

### Time spent in danger zone (<20% energy)
critical_time_fraction = sum(energy < 0.2) / total_time

3. Food Acquisition Rate

### Foods eaten per unit time
food_efficiency = total_foods_eaten / survival_time

### Search efficiency: time between spotting food and eating it
mean_acquisition_time = mean(eat_time - first_visible_time)

### Opportunity conversion: % of visible food actually eaten
conversion_rate = foods_eaten / foods_visible_in_lifetime

4. Predator Avoidance

### Damage taken per predator encounter
damage_per_encounter = total_damage / num_encounters

### Escape success rate
escape_rate = successful_escapes / total_encounters

### Reaction time: time from predator visible to flee action
reaction_time = mean(flee_start - predator_visible)

### Safe distance maintenance
mean_distance_from_predators = mean(distances_when_visible)

5. Learning Curves

Plot metrics over time to show improvement:

### Bucketed by experience
for epoch in [0-100, 100-500, 500-1000, 1000+]:
    survival_time[epoch]
    food_efficiency[epoch]
    damage_rate[epoch]

### Report: "Food efficiency improves 3.2x from epoch 0-100 (0.3 foods/min) 
          to epoch 1000+ (0.96 foods/min)"

Behavioral Emergence Metrics

6. Grouping Behavior

If they "stay in groups":

### Nearest neighbor distance over time
mean_nn_distance = mean(distance_to_nearest_agent)

### Time spent in groups (within threshold distance of others)
group_time_fraction = sum(nn_distance < threshold) / total_time

### Compare: random agents vs trained
random_grouping = 0.15  # baseline from random walk
trained_grouping = 0.67  # 4.5x more grouping

7. Exploration vs Exploitation

### Spatial coverage (how much of map explored)
visited_tiles = unique(positions_visited)
coverage = visited_tiles / total_tiles

### Return rate to known food locations
revisit_rate = visits_to_known_food / total_food_searches

### Balance metric: entropy of position distribution
exploration_entropy = -sum(p * log(p) for p in position_histogram)

8. Decision Quality

### Action diversity (not stuck in loops)
action_entropy = -sum(p * log(p) for p in action_distribution)

### Context-appropriate actions
#### When energy < 30%: % time spent moving toward food
food_seeking_when_hungry = moving_toward_food / time_hungry

### When predator nearby: % time spent fleeing vs attacking
flee_rate_in_danger = flee_actions / (flee_actions + other_actions)

Statistical Comparison Framework

Before/After Learning

### Run 100 agents for 50 episodes each
#### First 10 episodes vs Last 10 episodes

early_performance = metrics(episodes_0_to_10)
late_performance = metrics(episodes_40_to_50)

improvement = (late - early) / early
t_test(early, late)  # Statistical significance

Ablation Studies

### Full model vs variants
baseline = random_actions
perception_only = trained_encoder + random_actions
reward_only = random_encoder + trained_reward_head
full_model = trained_all

### Report which components matter most

Difficulty Scaling

#### Does learning transfer to harder environments?
easy = {predator_speed: 0.5, food_scarcity: 0.1}
medium = {predator_speed: 1.0, food_scarcity: 0.3}
hard = {predator_speed: 1.5, food_scarcity: 0.5}

#### Trained agents maintain performance advantage across difficulties?
performance_gap = trained_survival / untrained_survival
# Easy: 5.2x, Medium: 3.8x, Hard: 2.1x (learning helps even in hard mode)

Visualization for Papers/Demos

Figure 1: Survival Curves
Survival probability over time
100% |    Trained ────────────╮
    |                        │╲
50% |                        │ ╲___
    |    Untrained ──────╮   │
    |                    ╲___│
0% |________________________╲
    0     500    1000   1500   2000 steps

Figure 2: Learning Progression
Food Efficiency (foods/minute)
1.0 |                    ╱─────
    |                 ╱─╯
0.5 |           ╱────╯
    |     ╱────╯
0.0 |────╯___________________
    0   200  400  600  800  1000 episodes

Figure 3: Spatial Heatmaps
Untrained agents:        Trained agents:
■ ■ □ □ □ ■ ■ ■         ■ ■ ■ ■ ■ ■ ■ ■
■ ■ □ □ □ □ □ □         ■ ■ ■ ■ ■ ■ ■ ■
□ □ □ □ □ □ □ □         ■ ■ ■ ■ ■ ■ ■ ■
(Random wandering)       (Systematic coverage)

Concrete Implementation

class SimulationMetrics:
    def __init__(self):
        self.agent_lifespans = []
        self.foods_eaten = []
        self.damage_taken = []
        self.positions = []

    def record_death(self, agent):
        self.agent_lifespans.append(agent.survival_time)
        self.foods_eaten.append(agent.foods_eaten)
        self.damage_taken.append(agent.total_damage)

    def summary(self):
        return {
            'mean_survival': np.mean(self.agent_lifespans),
            'food_efficiency': np.mean(self.foods_eaten) / np.mean(self.agent_lifespans),
            'survival_95th': np.percentile(self.agent_lifespans, 95),
            'damage_rate': np.mean(self.damage_taken) / np.mean(self.agent_lifespans)
        }

    def compare(self, other):
        """Statistical comparison with another population"""
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(self.agent_lifespans, other.agent_lifespans)
        improvement = np.mean(self.agent_lifespans) / np.mean(other.agent_lifespans)
        return f"{improvement:.2f}x improvement (p={p_value:.4f})"

The Key Result

For your paper/README, you want to show:

"Trained agents survive 5.2x longer than untrained (mean: 1247s vs 240s, t=12.4, p<0.001), achieve 3.8x higher food efficiency (0.96 vs 0.25 foods/min), and take 2.3x less damage per 
predator encounter (15 vs 35 health). Learning curves show continued improvement over 1000 episodes with no catastrophic forgetting."

That's quantitative evidence of what you're seeing qualitatively. Which metrics resonate most with what you're observing?


## Chapter 17: Evolution and Breeding

With working online learning, we could explore evolutionary search:

- Population of agents in shared world
- Fitness = survival time
- Successful agents breed (genome crossover + mutation)
- Genome modulates architecture (spectral filter preferences, learning rates)
- Social behaviors emerge from multi-agent interaction?

The genome vector (100 dims) would encode architectural hyperparameters:
- Frequency preferences (low vs high)
- Modality attention (vision vs audio weighted)
- Learning rate modulation
- Even activation function blending

Evolution searches architecture space alongside weight learning. This is speculative, but the foundation exists.

## Chapter 18: The Honest Unknowns

We don't know:
- If Fourier mixing scales to 100M+ parameters
- If embodied learning matches language pretraining quality
- If survival pressure creates compositional reasoning
- If this paradigm unlocks capabilities text-only models can't reach
- At what scale a "pain signal" becomes ethically relevant

These are genuine uncertainties. We're exploring, not claiming.

**But we do know:**
- FFT is proven efficient (FNet, FFTNet)
- Online learning can work (biological existence proof)
- Multi-task prediction creates rich representations
- Self-supervised sensory learning is viable
- The components function individually

The combination is novel. The compute is manageable. The early results are encouraging.

Worth pushing further? Absolutely.

## Chapter 19: Why Consumer Hardware Matters

This isn't just about efficiency for its own sake.

If AI requires multi-million dollar GPU clusters, it remains centralized. A few organizations control development. Experimentation requires massive resources. Deployment means cloud dependence.

If AI runs on consumer hardware—truly runs, not "distilled version runs"—then:
- Anyone can experiment
- Edge deployment becomes real
- Privacy improves (on-device processing)
- Iteration cycles shorten (train on laptop overnight)
- Accessibility increases dramatically

FFT optimization is everywhere. Phones, laptops, embedded systems, FPGAs. 50 years of hardware evolution has made this operation incredibly cheap.

The LRN architecture leverages this. Forward pass <10ms on MacBook CPU. Training runs overnight, not for weeks. The 96% sequence accuracy was achieved on consumer hardware.

This matters for democratization of AI research.

---

# Part Five: Where This Goes

## Chapter 20: The Exploration Continues

**Immediate next steps:**
- Scale to sentence-level generation (50-200 phonemes)
- Add prosody modeling (questions rise, statements fall)
- Implement cross-modal grounding experiments
- Expand training data (LibriSpeech has thousands of hours)

**Medium-term experiments:**
- Modular architecture (split vision/speech/planning)
- Interactive learning (human teacher feedback loop)
- Memory mechanisms (remember past interactions)
- Generalization testing (compositional novel combinations)

**Long-term vision:**
- Full embodied agent in simulation
- Multi-agent populations with breeding
- Proto-language emergence from communication pressure
- Scale testing (how far does this architecture go?)

**Philosophical preparation:**
- Define ethical boundaries before we approach them
- Consider what "experience" means for artificial systems
- Think about suffering before building systems that might suffer

## Chapter 21: The Fundamental Question

Current AI learns about the world from descriptions. It knows "fire is hot" because those words co-occur in billions of sentences.

An embodied agent would know fire is hot because it approached heat, received a damage signal, learned to avoid.

Is that difference merely poetic? Or does it unlock something fundamental?

**The embodied knowledge hypothesis:**
Representations learned through sensory experience and survival pressure might be qualitatively different from representations learned through text statistics. Not necessarily better, but different—grounded in physics, in causality, in the consequences of actions.

**The speech results hint at this:**
The agent doesn't learn phonemes from phonetic descriptions. It learns by producing sounds, hearing them, adjusting. The knowledge is sensorimotor, not symbolic.

When we add vision—show an apple while saying "apple"—the agent won't learn "apple is a fruit" from Wikipedia. It'll learn "this visual pattern (red, round) co-occurs with this audio pattern (AE-P-AH-L)." That's grounded knowledge.

Can you build reasoning on that foundation? We don't know yet.

## Chapter 22: The Experiment Continues

This is fundamentally an exploration, not a product. We're asking: **How far can we push Fourier-based embodied learning?**

The answer unfolds through experiments:
- Speech (single words): 96% accuracy ✓
- Speech (multi-word phrases): 98% accuracy ✓
- Acoustic validation: 0.95 similarity ✓
- Cross-modal grounding: TBD
- Survival learning: TBD
- Language emergence: TBD
- Reasoning: TBD

Each step either works or teaches us why it doesn't. Both outcomes are valuable.

The components are proven individually. The combination is novel. The compute is manageable. The results keep exceeding expectations.

Let's see where this goes.

---

*"The question is not whether machines can think, but whether they can learn to think by living."*

---

**Document Status:** Living document, updated as experiments progress
**Last Updated:** December 2025
**Current Focus:** SPARC integration for embodied speech learning
**Architecture:** Primordial (brain) + SPARC (mouth) - see Chapter 5.5
**Next Steps:** Supervised articulatory prediction → Self-listening → RL babbling
