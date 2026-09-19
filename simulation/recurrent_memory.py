import torch

class AssociativeRecurrentMemory:
    """
    A toy model demonstrating fixed-size continuous recurrent state.
    Mathematical convention: M_{t+1} = M_t + v_t k_t^T
    """
    def __init__(self, state_dim, dtype=torch.float32):
        self.state_dim = state_dim
        self.dtype = dtype
        self.state = torch.zeros((state_dim, state_dim), dtype=dtype)
        
    def reset(self):
        """Clears the memory state."""
        self.state = torch.zeros((self.state_dim, self.state_dim), dtype=self.dtype)
        
    def update(self, key_vec, val_vec):
        """s_{t+1} = s_t + v * k^T"""
        k_norm = key_vec / (torch.norm(key_vec) + 1e-8)
        self.state += torch.outer(val_vec, k_norm)
        
    def retrieve(self, key_vec):
        """y_hat = M * k"""
        k_norm = key_vec / (torch.norm(key_vec) + 1e-8)
        return torch.mv(self.state, k_norm)
        
    def get_state(self):
        """Returns a copy of the current state matrix."""
        return self.state.clone()
        
    def get_memory_size(self):
        """Returns the number of parameters."""
        return self.state_dim * self.state_dim
        
    def get_memory_bytes(self):
        """Returns the physical memory footprint in bytes."""
        return self.state.element_size() * self.get_memory_size()
