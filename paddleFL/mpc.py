import torch
import numpy as np

def init(protocol="aby3", role=0, endpoints=""):
    print(f"✅ MPC Initialized | Protocol: {protocol} | Role: {role}")

def make_shares(data):
    """
    Mimics Secret Sharing: Splits a tensor into 3 random shares 
    such that share1 + share2 + share3 = data.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    
    # Generate random noise for the first two shares
    s1 = torch.randn_like(data)
    s2 = torch.randn_like(data)
    # The third share is the remainder
    s3 = data - (s1 + s2)
    return [s1, s2, s3]

def reconstruct(shares):
    """Combines shares back into the original data."""
    return sum(shares)

class MPCOptimizer:
    """Mimics an optimizer that operates on encrypted shares."""
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, shares, grads_shares):
        # In MPC, we update each share independently
        updated_shares = []
        for s, g in zip(shares, grads_shares):
            updated_shares.append(s - self.lr * g)
        return updated_shares