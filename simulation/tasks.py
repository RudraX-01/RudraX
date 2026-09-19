import torch
import math
import itertools

def generate_vocab_vectors(vocab, dim):
    """Assign a random vector to each word in a vocab."""
    return {word: torch.randn(dim) for word in vocab}

def create_task(num_pairs, dim, shared_component_pct, seed=None):
    """
    Generates key-value pairs with controlled correlation.
    Returns: pairs, key_dict, val_dict, measured_mean_similarity
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    pairs = []
    
    base_facts = [
        ("Capital of France", "Paris"),
        ("Largest planet", "Jupiter"),
        ("Author of Hamlet", "Shakespeare"),
        ("Freezing point of water", "0°C"),
        ("Red planet", "Mars"),
        ("Chemical symbol for Gold", "Au"),
        ("Speed of light", "3e8 m/s"),
        ("First prime number", "2"),
        ("Square root of 144", "12"),
        ("Tallest mountain", "Everest")
    ]
    
    facts = []
    for i in range(num_pairs):
        if i < len(base_facts):
            facts.append(base_facts[i])
        else:
            facts.append((f"Synthetic Fact {i+1}", f"Synthetic Answer {i+1}"))
            
    keys = [f[0] for f in facts]
    vals = [f[1] for f in facts]
    
    key_dict = {}
    val_dict = generate_vocab_vectors(sorted(set(vals)), dim)
    
    base_vector = torch.randn(dim)
    base_vector = base_vector / torch.norm(base_vector)
    
    rho = shared_component_pct / 100.0
    
    for i in range(num_pairs):
        z = torch.randn(dim)
        z = z / torch.norm(z)
        
        k_vec = rho * base_vector + math.sqrt(1 - rho**2) * z
        k_vec = k_vec / torch.norm(k_vec)
        
        key_dict[keys[i]] = k_vec
        pairs.append((keys[i], vals[i], k_vec, val_dict[vals[i]]))
        
    # Measure actual pairwise cosine similarity among all unique keys
    all_k_vecs = [p[2] for p in pairs]
    sims = []
    if len(all_k_vecs) > 1:
        for v1, v2 in itertools.combinations(all_k_vecs, 2):
            sims.append(torch.nn.functional.cosine_similarity(v1, v2, dim=0).item())
        measured_mean_sim = sum(sims) / len(sims)
    else:
        measured_mean_sim = 1.0
        
    return pairs, key_dict, val_dict, measured_mean_sim
