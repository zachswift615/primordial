"""Autoregressive transformer decoder for phoneme sequence generation."""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .config import SpeechConfig
from .latent import TOTAL_VOCAB, LATENT_DIM
from .training import SpeechLRN
from .phonemes import index_to_phoneme


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequenceDecoder(nn.Module):
    """Transformer decoder for autoregressive phoneme sequence generation.

    Takes pooled audio encoding and generates a sequence of phonemes
    autoregressively, with dual output heads for discrete tokens and
    continuous latent positions.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        # Dimensions
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.ffn_dim = 256
        self.pooled_dim = 384  # From existing encoder

        # Embeddings
        self.phoneme_embed = nn.Embedding(TOTAL_VOCAB, self.hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=32, dropout=0.1
        )

        # Memory projection (384 -> 128)
        self.memory_proj = nn.Linear(self.pooled_dim, self.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Dual output heads
        self.discrete_head = nn.Linear(self.hidden_dim, TOTAL_VOCAB)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, LATENT_DIM),
            nn.Tanh(),
        )

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal attention mask.

        Returns:
            Upper triangular matrix where True = masked (future tokens)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with teacher forcing.

        Args:
            input_ids: (batch, seq_len) phoneme token indices
            memory: (batch, 384) pooled audio encoding from encoder
            tgt_mask: Optional causal mask, generated if not provided

        Returns:
            discrete_logits: (batch, seq_len, 43) logits over vocabulary
            latent: (batch, seq_len, 6) latent positions in [-1, 1]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed and add positional encoding
        x = self.phoneme_embed(input_ids)  # (batch, seq_len, 128)
        x = self.pos_encoding(x)

        # Project memory and expand for cross-attention
        memory = self.memory_proj(memory)  # (batch, 128)
        memory = memory.unsqueeze(1)       # (batch, 1, 128)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len, device)

        # Decode
        output = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (batch, seq_len, 128)

        # Dual heads
        discrete_logits = self.discrete_head(output)
        latent = self.latent_head(output)

        return discrete_logits, latent


class SpeechSequenceLRN(nn.Module):
    """Full model: CNN Encoder + Fourier Mixing + Sequence Decoder.

    Combines the existing speech encoder with the new autoregressive
    decoder for word-level speech production.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        # Reuse existing encoder
        self.encoder = SpeechLRN(config)

        # New sequence decoder
        self.decoder = SequenceDecoder(config)

        # Token constants
        from .latent import SOS_TOKEN, EOS_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to pooled features.

        Args:
            mel: (batch, 80, n_frames) mel spectrogram

        Returns:
            (batch, 384) pooled audio encoding
        """
        return self.encoder.encode(mel)

    def forward(
        self,
        mel: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Args:
            mel: (batch, 80, n_frames) mel spectrogram
            target_tokens: (batch, seq_len) target phoneme indices

        Returns:
            discrete_logits: (batch, seq_len, 43)
            latent: (batch, seq_len, 6)
        """
        pooled = self.encode(mel)
        return self.decoder(target_tokens, memory=pooled)

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_length: int = 15,
        temperature: float = 0.0,
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """Autoregressive phoneme sequence generation.

        Args:
            mel: (1, 80, n_frames) single mel spectrogram
            max_length: Maximum phonemes to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            phonemes: List[str] - generated phoneme sequence
            latents: (seq_len, 6) - latent vectors, or None if empty
        """
        device = mel.device
        pooled = self.encode(mel)  # (1, 384)

        # Start with SOS
        generated_tokens = [self.sos_token]
        generated_latents = []

        for _ in range(max_length):
            # Prepare input
            input_ids = torch.tensor([generated_tokens], device=device)

            # Decode
            discrete_logits, latent_pred = self.decoder(input_ids, memory=pooled)

            # Get prediction from last position
            next_logits = discrete_logits[0, -1]  # (43,)
            next_latent = latent_pred[0, -1]      # (6,)

            # Sample or greedy
            if temperature == 0:
                next_token = next_logits.argmax().item()
            else:
                import torch.nn.functional as F
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Stop on EOS
            if next_token == self.eos_token:
                break

            generated_tokens.append(next_token)
            generated_latents.append(next_latent)

        # Convert to phoneme strings (skip SOS)
        phonemes = [index_to_phoneme(t) for t in generated_tokens[1:]]
        latents = torch.stack(generated_latents) if generated_latents else None

        return phonemes, latents
