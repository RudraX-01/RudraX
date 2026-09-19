import torch

def evaluate_memory(memory_model, pairs, val_dict):
    """
    Tests the memory model on a set of pairs.
    Returns the accuracy percentage and a log of predictions.
    """
    correct = 0
    results = []
    
    for key_str, true_val_str, k_vec, _ in pairs:
        pred_vec = memory_model.retrieve(k_vec)
        
        best_match = None
        best_sim = -float('inf')
        
        for v_str, v_vec in val_dict.items():
            sim = torch.nn.functional.cosine_similarity(pred_vec, v_vec, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_match = v_str
                
        is_correct = (best_match == true_val_str)
        if is_correct:
            correct += 1
            
        results.append({
            "Query": key_str,
            "Ground Truth": true_val_str,
            "Model Prediction": best_match,
            "Similarity": best_sim,
            "Correct": is_correct
        })
        
    accuracy = (correct / len(pairs)) * 100 if len(pairs) > 0 else 0
    return accuracy, results
