"""Tests for latent phoneme space."""
import torch
from primordial.speech.latent import (
    SOS_TOKEN, EOS_TOKEN, TOTAL_VOCAB,
    PHONEME_ANCHORS, get_anchor, snap_to_nearest_anchor
)


def test_token_constants():
    """Verify token indices are correctly defined."""
    assert SOS_TOKEN == 41
    assert EOS_TOKEN == 42
    assert TOTAL_VOCAB == 43


def test_sos_eos_anchors():
    """SOS and EOS should have anchors at origin."""
    sos_anchor = get_anchor('SOS')
    eos_anchor = get_anchor('EOS')

    assert sos_anchor.shape == (6,)
    assert eos_anchor.shape == (6,)
    assert torch.allclose(sos_anchor, torch.zeros(6))
    assert torch.allclose(eos_anchor, torch.zeros(6))


def test_snap_excludes_sos_eos():
    """Snapping should not return SOS or EOS for normal latents."""
    # A latent near the IY anchor
    latent = torch.tensor([1.0, 1.0, -1.0, 1.0, 0.0, -1.0])
    phoneme, dist = snap_to_nearest_anchor(latent)

    assert phoneme == 'IY'
    assert phoneme not in ('SOS', 'EOS')
