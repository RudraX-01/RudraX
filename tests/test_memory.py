import torch
from simulation.recurrent_memory import AssociativeRecurrentMemory

def test_memory_initialization():
    mem = AssociativeRecurrentMemory(8)
    assert mem.get_memory_size() == 64
    assert mem.get_state().shape == (8, 8)

def test_memory_update_and_retrieve():
    mem = AssociativeRecurrentMemory(8)
    k = torch.randn(8)
    v = torch.randn(8)
    
    mem.update(k, v)
    retrieved = mem.retrieve(k)
    
    # Should be highly correlated with v
    sim = torch.nn.functional.cosine_similarity(v, retrieved, dim=0)
    assert sim.item() > 0.99

def test_memory_reset():
    mem = AssociativeRecurrentMemory(8)
    mem.update(torch.randn(8), torch.randn(8))
    mem.reset()
    assert torch.all(mem.get_state() == 0)
