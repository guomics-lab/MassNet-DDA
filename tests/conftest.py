import pytest
import torch

@pytest.fixture(autouse=True)
def cuda_init():
    """Initialize CUDA for tests if available"""
    if torch.cuda.is_available():
        torch.cuda.init()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def dummy_spectrum_batch():
    """Create a dummy spectrum batch for testing"""
    batch_size = 2
    seq_len = 10
    feature_dim = 256
    return {
        'peaks': torch.randn(batch_size, seq_len, feature_dim),
        'precursors': torch.randn(batch_size, 2),  # m/z and charge
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }

@pytest.fixture
def dummy_peptide_batch():
    """Create a dummy peptide batch for testing"""
    batch_size = 2
    seq_len = 10
    return {
        'sequence': torch.randint(0, 26, (batch_size, seq_len)),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
