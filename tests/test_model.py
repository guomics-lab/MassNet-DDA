import pytest
import torch
from XuanjiNovo.denovo.model import Spec2Pep
from XuanjiNovo.components.transformers import SpectrumEncoder, PeptideEncoder, PeptideDecoder

def test_model_initialization():
    """Test basic model initialization with default parameters"""
    model = Spec2Pep(
        vocab_size=26,
        hidden_dim=256,
        n_head=4,
        dim_feedforward=1024,
        n_layers=3,
        dropout=0.1,
        max_length=50,
        refine_iters=3,
        mask_schedule={
            "initial_peek": 0.93,
            "epoch_decay": 0.01,
            "min_peek": 0.00
        }
    )
    assert isinstance(model, Spec2Pep)
    assert model.hparams.vocab_size == 26
    assert model.hparams.refine_iters == 3

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda():
    """Test model CUDA compatibility"""
    model = Spec2Pep(
        vocab_size=26,
        hidden_dim=256,
        n_head=4,
        dim_feedforward=1024,
        n_layers=3
    )
    model = model.cuda()
    assert next(model.parameters()).is_cuda

def test_spectrum_encoder():
    """Test spectrum encoder with dummy input"""
    encoder = SpectrumEncoder(
        d_model=256,
        n_head=4,
        dim_feedforward=1024,
        n_layers=3,
        dropout=0.1
    )
    batch_size = 2
    seq_len = 10
    feature_dim = 256
    x = torch.randn(batch_size, seq_len, feature_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    output = encoder(x, mask)
    assert output.shape == (batch_size, seq_len, feature_dim)

def test_peptide_decoder():
    """Test peptide decoder with dummy input"""
    decoder = PeptideDecoder(
        vocab_size=26,
        d_model=256,
        n_head=4,
        dim_feedforward=1024,
        n_layers=3,
        dropout=0.1
    )
    batch_size = 2
    seq_len = 10
    feature_dim = 256
    tgt = torch.randint(0, 26, (batch_size, seq_len))
    memory = torch.randn(batch_size, seq_len, feature_dim)
    tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    memory_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    output, _, _ = decoder(tgt, None, memory, tgt_mask, memory_mask)
    assert output.shape == (batch_size, seq_len, 26)

def test_config_parameters():
    """Test model configuration parameter handling"""
    model = Spec2Pep(
        vocab_size=26,
        hidden_dim=256,
        refine_iters=5,
        mask_schedule={
            "initial_peek": 0.95,
            "epoch_decay": 0.02,
            "min_peek": 0.10
        }
    )
    assert model.hparams.refine_iters == 5
    assert model.hparams.mask_schedule["initial_peek"] == 0.95
    assert model.hparams.mask_schedule["epoch_decay"] == 0.02
    assert model.hparams.mask_schedule["min_peek"] == 0.10
