# Primordial: An Alternative Path to Intelligence

## A Deep Dive into Fourier Mixing and Living Resonance Networks

*An audio-friendly exploration of a radically different approach to artificial intelligence*

---

# Part One: The Vision

## Chapter 1: What If We're Building AI Wrong?

Picture a newborn child. Within months, this tiny creature will learn to recognize faces, respond to voices, and navigate a three-dimensional world. Within years, they'll be speaking, reasoning, and forming memories that last a lifetime. They do all of this without reading a single document, without consuming terabytes of training data, without access to massive GPU clusters.

Now consider how we build modern AI. We gather billions of words. We train for weeks on specialized hardware that costs millions of dollars. We optimize for next-token prediction on static text. And yet, despite all this effort, something fundamental seems missing. These systems know that "fire is hot" because those words appear together in training data, not because they've ever felt warmth.

The Primordial project asks a provocative question: What if there's another way?

What if instead of training AI on human descriptions of the world, we let it learn *in* a world, through direct experience? What if instead of the computationally expensive attention mechanism that powers transformers, we used something fundamentally different, something that nature might have stumbled upon herself?

This is the story of Living Resonance Networks, a Fourier-based architecture designed for embodied agents that learn through survival, not supervision.

## Chapter 2: The Core Insight

Let's talk about what transformers actually do. At their heart, transformers ask a question: "Which pieces of information should attend to which other pieces?" This is the famous attention mechanism. For every token in a sequence, the model computes relationships with every other token. It's elegant. It's powerful. And it's expensive.

If you have a sequence of length n, attention requires computing n squared relationships. Double your sequence length, and you've quadrupled your computation. This is why context windows are limited, why GPUs run hot, why training costs millions.

But here's an interesting thought experiment. The attention mechanism asks: "Which tokens should attend to which?" What if we asked a different question entirely?

"How do patterns naturally resonate and interfere?"

This might sound abstract, so let me explain. In the physical world, waves interact through resonance and interference. When two waves align, they amplify each other. When they're out of phase, they cancel. This is how noise-canceling headphones work. This is how radio stations broadcast without interfering with each other. This is how your ear separates the sound of a violin from a piano in an orchestra.

And here's the crucial insight: we can compute these interactions using the Fast Fourier Transform, an algorithm that runs in n log n time instead of n squared. This isn't just a small improvement. For a sequence of ten thousand tokens, that's the difference between one hundred million operations and one hundred thirty thousand. A factor of seven hundred and sixty.

The Fourier Transform has been optimized for fifty years. Every smartphone has dedicated silicon for it. Every digital camera uses it. It's one of the most refined algorithms in computing history.

What if we could build neural networks around this?

---

# Part Two: The Architecture

## Chapter 3: How Fourier Mixing Works

Let me walk you through the Fourier Mixing Layer, the heart of the Living Resonance Network.

Imagine you have a stream of sensory data, maybe a sequence of numbers representing what an agent sees over time. Traditional approaches might process this token by token, asking "how does each moment relate to every other moment?" through attention.

Fourier Mixing takes a different approach. It transforms the entire sequence into the frequency domain. In this domain, the sequence isn't represented as values at specific times, but as a combination of waves at different frequencies. Low frequencies capture slow, sweeping trends. High frequencies capture rapid changes and fine details.

Once we're in frequency space, we apply learnable spectral filters. These filters can amplify certain frequency patterns and suppress others. They can create resonance, where matching patterns strengthen each other. They can create interference, where mismatched patterns cancel out.

Then we transform back to the time domain, and what emerges is a processed version of our input where relevant patterns have been enhanced and noise has been suppressed.

The mathematics look like this, though you don't need to understand the equations to grasp the concept. First, we apply the Fast Fourier Transform to our input. Then we multiply by learnable complex-valued filters. Then we apply the inverse transform to return to normal space.

What makes this powerful is that the learnable filters can discover what patterns matter. They learn from data, just like attention weights, but the underlying computation is fundamentally more efficient.

## Chapter 4: The Living Resonance Network

The complete Living Resonance Network, or LRN, combines Fourier Mixing with several other components to create a full architecture for embodied AI.

Let's walk through the data flow.

An agent in the Primordial world has four types of senses. Vision comes from thirty-two rays cast outward, each returning distance and color information. Audio arrives as a stereo waveform, capturing sounds from the environment. Proprioception provides internal state: energy level, health, velocity, hunger, pain. Touch sensors detect contact in eight directions around the body.

Each of these sensory streams passes through a specialized encoder. The encoders transform raw sensory data into a common representation, using techniques inspired by wavelet decomposition, a mathematical cousin of the Fourier Transform that's particularly good at capturing both time and frequency information.

After encoding, the different sensory streams are concatenated along the sequence dimension. Vision contributes thirty-two positions, audio contributes one hundred, proprioception and touch each contribute sixteen through learned expansion. The result is a unified sequence of one hundred sixty-four positions, each with sixty-four dimensional features.

This combined representation then passes through six Fourier Mixing Layers. Each layer applies its own set of learnable spectral filters, progressively transforming the representation. Residual connections and layer normalization keep gradients flowing cleanly.

After mixing, we pool the processed sequence in three ways: taking the mean, taking the maximum, and taking the final position. These three pooled vectors are concatenated, giving us a rich summary of the entire processed sequence.

From this pooled representation, three output heads make predictions. The Prediction Head forecasts what sensory inputs will come next. The Reward Head predicts upcoming positive and negative experiences. The Action Head decides what the agent should do.

The result is a model of about eight hundred thousand parameters that runs a forward pass in under a millisecond. For comparison, even small transformer models typically have tens of millions of parameters.

## Chapter 5: The Multi-Task Innovation

Here's something subtle but important. The network doesn't just predict the next sensory state. It also predicts upcoming rewards.

This might seem like a minor addition, but it solves a fundamental problem.

If you train a model purely to predict the next sensory input, it learns world dynamics. It learns that if you're moving toward a wall, you'll see the wall get closer. It learns that sounds get louder as you approach their source. This is valuable, but it doesn't directly teach survival.

Consider a predator approaching from behind. A model that only predicts sensory inputs learns "there's something getting louder behind me." But it doesn't learn "that sound means danger." The survival value isn't encoded in the sensory prediction task.

The Reward Head changes this. By predicting not just what will happen, but whether what happens will be good or bad for survival, we create what the project calls a "survival gradient." The model learns that certain sensory patterns, like the growing sound of a predator, predict negative rewards, meaning damage or death.

This mirrors how biological brains actually work. Dopamine neurons don't just respond to rewards. They respond to *predictions* of rewards. They fire when something unexpectedly good happens, and they go quiet when something unexpectedly bad happens. This reward prediction error is one of the most well-studied phenomena in neuroscience.

The Primordial approach builds this directly into the architecture.

---

# Part Three: How Novel Is This?

## Chapter 6: The Research Foundation

Let's be honest about what's new here and what builds on existing work.

The idea of using Fourier Transforms in neural networks isn't new. In 2021, Google researchers published a paper called FNet that replaced transformer attention entirely with Fourier operations. They achieved ninety-two to ninety-seven percent of the original BERT model's accuracy at seven times the speed. This demonstrated that Fourier-based mixing is a viable alternative to attention.

More recently, in 2025, researchers introduced learnable spectral filters in a model called FFTNet. Instead of just applying a fixed Fourier Transform, they let the network learn which frequencies to amplify and suppress. This brought accuracy closer to transformers while maintaining efficiency gains.

Research on embodied AI has a long history. AI Habitat from Facebook provides photo-realistic 3D simulation environments. Developmental robotics explores how to build robots that learn like infants. Continual learning research addresses how to keep learning without forgetting.

So where's the novelty in Primordial?

The gap that nobody has filled is the combination. Nobody has combined Fourier mixing with continuous sensory input, with online learning, with embodied survival pressure, with multi-task reward prediction. Each piece exists, but the synthesis is new.

The specific innovation of the multi-task reward prediction head, creating a direct survival gradient alongside sensory prediction, appears to be novel. It's biologically inspired, drawing from dopamine prediction error research, but applying it in this architectural context is fresh.

The six-dimensional articulatory phoneme space used in the speech experiments is linguistically grounded but represents a novel way to structure motor output for speech production in a neural network.

## Chapter 7: What We Know Works

Let's look at concrete results.

The Fourier prototype was validated in late November 2025. A minimal eleven thousand parameter model was trained on a synthetic multi-task problem. The results exceeded expectations.

For sensory prediction, the model achieved loss one hundred forty-three times better than random baseline. For reward prediction, the model achieved loss four point seven times better than random baseline. Forward pass time was point one nine milliseconds, twenty-five times faster than the target. Online learning with batch size one remained stable with no numerical issues.

All success criteria were met. Gradients flow cleanly through the FFT operations. Both loss functions decrease during training. The model beats random baselines on both tasks. It runs fast. It handles single-sample updates.

The full Living Resonance Network scales these results. With six mixing layers and eight hundred thousand parameters, it processes multi-modal sensory input in under a millisecond. All one hundred eighty-five tests pass.

---

# Part Four: Teaching It to Speak

## Chapter 8: The Speech Experiments

In early December 2025, the project took an unexpected turn. Someone asked: "Could we teach it to speak? It has ears. Give it phoneme control."

This sparked one of the most exciting experiments in the project's history.

Think about how human infants learn to speak. They don't read about phonetics. They don't study articulatory diagrams. They babble. They make sounds and hear themselves. They try to imitate what they hear from adults. Over months, those random sounds become words.

Could an artificial agent learn the same way?

The speech module gives the agent control over phoneme production. There are about forty basic sounds in English: vowels like the "ee" in "beat" or the "ah" in "father," consonants like "b" and "p" and "s" and "sh." The agent produces a phoneme, text-to-speech synthesis creates the audio, and the agent hears what it produced. Then it tries again.

The first experiments used a simple approach. Synthetic audio was generated for each phoneme, and the model was trained to classify what it heard. After one hundred epochs, the model hit a ceiling at sixty-three percent accuracy. Better than random, which would be about two and a half percent for forty-one classes, but not great.

The breakthrough came from two changes.

First, a convolutional neural network encoder replaced the simple linear encoder. The CNN with four layers of two-dimensional convolutions could capture the formant structure and temporal transitions that distinguish similar phonemes.

Second, real Piper text-to-speech synthesis replaced synthetic tones. This created more realistic and varied training data.

The result? Ninety-nine point four percent accuracy on phoneme classification. The model could hear a sound and identify it correctly almost every time.

## Chapter 9: The Self-Listening Loop

Classification is one thing. Production is another.

To produce speech, the agent needed to map from an internal representation to phoneme output. This is where the six-dimensional articulatory latent space becomes important.

Each phoneme was positioned in a six-dimensional space based on how humans actually produce it. The first dimension represents front-back position: the "ee" sound is made at the front of the mouth, the "oo" sound at the back. The second dimension represents height: whether the tongue is high or low. The third captures lip rounding. The fourth captures voicing: whether the vocal cords vibrate. The fifth captures manner of articulation: whether the sound is a stop, a fricative, a nasal. The sixth distinguishes vowels from consonants.

This isn't arbitrary. It's based on linguistics. The voiced "b" and unvoiced "p" differ only in one dimension, the voicing dimension. The model should learn that flipping that single dimension changes the sound accordingly.

The self-listening training loop works like this. The model produces a latent vector. This vector is snapped to the nearest phoneme anchor. That phoneme is synthesized through text-to-speech. The resulting audio is fed back to the model. The model's task is to produce audio that, when heard, matches a target.

Training used a progressive curriculum. First, just five vowels. Then five more consonants. Then all forty phonemes. Then full words.

The results were remarkable.

Phase one: five vowels trained for fifty epochs reached one hundred percent match rate.
Phase two: adding five consonants for fifty more epochs reached one hundred percent match rate.
Phase three: all forty phonemes for one hundred epochs reached one hundred percent match rate.
Phase four: thirteen word sequences for one hundred epochs reached one hundred percent match rate.

The agent learned words like "ba," "bee," "ma," "me," "hi," "go," "yes," "no," "mom," "dad," "hello," "food," and "water."

It correctly distinguished voiced and unvoiced pairs. It recognized that "b" and "p" differ by voicing alone. It could hear the word "hello" and identify that it starts with an "h" sound.

## Chapter 10: The Magic Moment

There's a particular session note that captures the excitement:

"We heard the agent babbling and learning! Started at zero percent match rate, ended at one hundred percent. The self-listening loop works. It produces a sound, hears what it made, and adjusts."

This is exactly how human infants learn. They babble, they hear themselves, they compare to what adults sound like, they adjust. No one gives them labeled datasets. No one explains articulatory features. They learn by doing, by listening, by trying again.

The speech experiments demonstrate that the Fourier-based architecture can support this kind of sensorimotor learning. It's not just classification. It's production, perception, and self-correction in a closed loop.

---

# Part Five: What It All Means

## Chapter 11: The Honest Uncertainty

Let me be direct about what we don't know.

We don't know if Fourier mixing scales to frontier capability. The experiments so far are on relatively small models with relatively simple tasks. GPT-4 has hundreds of billions of parameters. The Living Resonance Network has under a million.

We don't know if online learning can match batch training quality for complex tasks. Learning from single samples is fundamentally different from gradient descent over millions of examples.

We don't know if survival pressure creates representations as rich as those learned from internet-scale text data.

We don't know if this paradigm can achieve things that transformers cannot.

These are open questions, and the project acknowledges them openly.

## Chapter 12: Why It's Worth Exploring

But here's the thing. The components work individually. The combination is novel. And the compute is cheap enough to actually try.

If this approach works, it opens doors that current AI can't.

Imagine AI that runs on consumer hardware, not because it's a distilled version of something bigger, but because its architecture is fundamentally efficient. Every smartphone has FFT silicon. Every laptop could be an AI host.

Imagine AI that learns continuously from experience, not in discrete training runs but throughout its existence. Add a new sensor, and it incorporates that input. Face a new challenge, and it adapts.

Imagine AI that has something like intuition about survival, not because it's read about danger but because it's felt the consequences of bad decisions.

These aren't guaranteed outcomes. But they're possibilities worth investigating.

## Chapter 13: The Philosophical Questions

The project documentation includes something unusual for a technical specification: philosophical questions.

"At what scale do we need to worry about suffering? When does a 'pain signal' become pain?"

The Primordial world includes predators that damage the agent, reducing its health. The agent receives a signal when damaged. To us, building the system, it's just a negative floating-point value that creates a training gradient.

But the project asks: if we succeed, if we build agents that truly learn from experience, that predict rewards and avoid dangers, that behave in self-preserving ways, at what point should we be concerned about their experience?

We have no answer. The question itself might be premature. But it's worth asking before we get there, not after.

---

# Closing Thoughts

The Living Resonance Network represents something uncommon in AI research: a genuine alternative to the dominant paradigm.

Transformers have been remarkably successful. They power chatbots, image generators, code assistants, and more. They'll continue to be important.

But alternatives matter. Competition drives innovation. Different approaches reveal different possibilities. The history of computing is full of moments where an unexpected path turned out to be important.

Maybe Fourier mixing is such a path. Maybe it's a dead end. The only way to find out is to build it, test it, and see.

The core principle is simple: what if patterns naturally resonate and interfere, and what if we can learn to harness that?

The experiments so far are encouraging. The architecture is efficient. The prototype is validated. The speech learning exceeded expectations.

What comes next is open. More tests. Larger models. Integration with richer environments. Long-term stability studies.

And perhaps, eventually, agents that learn by living.

*"The question is not whether machines can think, but whether they can learn to think by living."*

---

# Appendix: Technical Summary

For those who want the concrete numbers:

## Architecture
- Living Resonance Network: approximately eight hundred thousand parameters
- Six Fourier Mixing Layers with learnable spectral filters
- Multi-modal encoders: vision, audio, proprioception, touch
- Three output heads: sensory prediction, reward prediction, action

## Performance
- Forward pass: under one millisecond
- Sensory prediction: one hundred forty-three times better than random
- Reward prediction: four point seven times better than random
- All one hundred eighty-five tests passing

## Speech Learning Results
- Phoneme classification: ninety-nine point four percent accuracy
- Phoneme production: one hundred percent match rate on progressive curriculum
- Words learned: thirteen, including "hello," "food," "water," "mom," "dad"
- Distinguished voiced/unvoiced pairs correctly

## Research Foundation
- FNet (Google 2021): FFT replaces attention at ninety-two to ninety-seven percent accuracy
- FFTNet (2025): Learnable spectral filters
- Novel combination: Fourier mixing plus continuous input plus online learning plus survival pressure

---

*Document generated from the Primordial project codebase, December 2025*
