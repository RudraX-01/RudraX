from simulation.experiments import run_interference_sweep, run_capacity_sweep, run_state_size_sweep

def test_interference_sweep():
    res = run_interference_sweep(10, 8, [0, 80], num_seeds=2)
    assert len(res) == 2
    assert "Mean Recall (%)" in res[0]
    assert "Std Dev (%)" in res[0]

def test_capacity_sweep():
    res = run_capacity_sweep([5, 10], 8, 50, num_seeds=2)
    assert len(res) == 2
    assert "Facts" in res[0]

def test_state_size_sweep():
    res = run_state_size_sweep([4, 8], 10, 50, num_seeds=2)
    assert len(res) == 2
    assert "State Dimension" in res[0]
