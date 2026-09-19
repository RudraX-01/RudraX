import numpy as np
from simulation.recurrent_memory import AssociativeRecurrentMemory
from simulation.tasks import create_task
from simulation.evaluation import evaluate_memory

def run_interference_sweep(facts_count, dim, shared_components, num_seeds=10):
    results = []
    for s in shared_components:
        accs, m_sims = [], []
        for seed in range(num_seeds):
            p, _, v, m_sim = create_task(facts_count, dim, s, seed)
            m = AssociativeRecurrentMemory(dim)
            for _, _, kv, vv in p: m.update(kv, vv)
            a, _ = evaluate_memory(m, p, v)
            accs.append(a)
            m_sims.append(m_sim)
        results.append({
            "Shared Component (%)": s, 
            "Measured Correlation": np.mean(m_sims), 
            "Mean Recall (%)": np.mean(accs), 
            "Std Dev (%)": np.std(accs)
        })
    return results

def run_capacity_sweep(fact_counts, dim, shared_component, num_seeds=10):
    results = []
    for f in fact_counts:
        accs = []
        for seed in range(num_seeds):
            p, _, v, _ = create_task(f, dim, shared_component, seed)
            m = AssociativeRecurrentMemory(dim)
            for _, _, kv, vv in p: m.update(kv, vv)
            a, _ = evaluate_memory(m, p, v)
            accs.append(a)
        results.append({
            "Facts": f, 
            "Mean Recall (%)": np.mean(accs), 
            "Std Dev (%)": np.std(accs), 
            "State Size": f"{dim}x{dim}"
        })
    return results

def run_state_size_sweep(dims, facts_count, shared_component, num_seeds=10):
    results = []
    for d in dims:
        accs = []
        for seed in range(num_seeds):
            p, _, v, _ = create_task(facts_count, d, shared_component, seed)
            m = AssociativeRecurrentMemory(d)
            for _, _, kv, vv in p: m.update(kv, vv)
            a, _ = evaluate_memory(m, p, v)
            accs.append(a)
        results.append({
            "State Dimension": d, 
            "Mean Recall (%)": np.mean(accs), 
            "Std Dev (%)": np.std(accs)
        })
    return results
