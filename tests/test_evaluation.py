import torch
from simulation.recurrent_memory import AssociativeRecurrentMemory
from simulation.evaluation import evaluate_memory

def test_evaluation_perfect_recall():
    mem = AssociativeRecurrentMemory(16)
    val_dict = {"A": torch.randn(16)}
    k = torch.randn(16)
    pairs = [("Q", "A", k, val_dict["A"])]
    
    mem.update(k, val_dict["A"])
    acc, results = evaluate_memory(mem, pairs, val_dict)
    
    assert acc == 100.0
    assert results[0]["Correct"] == True
