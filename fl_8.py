import torch
from paddleFL import mpc

# --- 1. Setup ---
mpc.init(role=0) # Acting as the aggregator
lr = 0.1

# --- 2. Private Data at Client ---
# Let's say a client has a private weight gradient
private_gradient = torch.tensor([0.5, -0.2, 0.1])
print(f"Original Private Gradient: {private_gradient}")

# --- 3. Secret Sharing (MPC) ---
# The client splits the gradient into 3 encrypted shares
shares = mpc.make_shares(private_gradient)
print(f"Sent Share 1 to Party A: {shares[0]}")
print(f"Sent Share 2 to Party B: {shares[1]}")
print(f"Sent Share 3 to Party C: {shares[2]}")

# --- 4. Secure Aggregation ---
# The parties (or server) can sum shares without knowing the original value
# If multiple clients sent shares, we would sum them share-wise here.
global_model_shares = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
optimizer = mpc.MPCOptimizer(lr=lr)

# Simulate an update step on the shares
new_shares = optimizer.step(global_model_shares, shares)

# --- 5. Reconstruction ---
# Only at the very end do we combine shares to reveal the result
final_update = mpc.reconstruct(new_shares)
print(f"\nFinal Decrypted Global Update: {final_update}")

