import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from model import GPT2Model

# Dummy config
class DummyConfig:
    vocab_size = 100
    hidden_dim = 64
    num_layers = 2
    num_heads = 8
    max_seq_len = 32
    dropout = 0.1

@pytest.fixture
def dummy_model():
    return GPT2Model(DummyConfig())

def test_forward_pass(dummy_model):
    input_ids = torch.randint(0, 100, (2, 16))
    logits = dummy_model(input_ids)
    assert logits.shape == (2, 16, 100)
