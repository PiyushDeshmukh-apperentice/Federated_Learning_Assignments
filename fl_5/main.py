import torch
import matplotlib.pyplot as plt
import os

from model import RNNModel
from client import train_client
from server import fedavg_aggregate
from dataset import ShakespeareDataset, get_client_dataloader

# 🔷 Configuration for Convergence
TARGET_ACCURACY = 0.50  # 40% accuracy threshold for "Acceptable Level"
MAX_ROUNDS = 50         # Prevent infinite loops
num_clients = 5

# 🔷 Load dataset
dataset = ShakespeareDataset("/mnt/StorageHDD/Coding/FL_Assignment/fl_4/data/shakespeare2.txt")
vocab_size = len(dataset.chars)

# 🔷 Initialize global model
global_model = RNNModel(vocab_size=vocab_size)

# 🔷 Metrics storage
round_losses = []
round_accuracies = []

def evaluate(model, dataloader):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            loss = criterion(output, y)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

# 🔷 CONVERGENCE LOOP
current_accuracy = 0.0
r = 0

print(f"🚀 Starting Convergence Analysis (Target: {TARGET_ACCURACY*100}%)")

while current_accuracy < TARGET_ACCURACY and r < MAX_ROUNDS:
    r += 1
    print(f"\n--- Round {r} ---")

    client_updates = []
    client_losses = []

    for client_id in range(num_clients):
        # Local copy of global weights
        client_model = RNNModel(vocab_size=vocab_size)
        client_model.load_state_dict(global_model.state_dict())

        # Load client-specific partition
        dataloader = get_client_dataloader(dataset, client_id, num_clients, batch_size=32)

        # Local training
        weights, num_samples, loss = train_client(client_model, dataloader)
        client_updates.append((weights, num_samples))
        client_losses.append(loss)

    # 🔷 Federated Averaging (Aggregation)
    global_model = fedavg_aggregate(global_model, client_updates)

    # 🔷 Validation
    eval_loader = get_client_dataloader(dataset, 0, 1, batch_size=64)
    current_accuracy, val_loss = evaluate(global_model, eval_loader)

    # Record metrics
    avg_train_loss = sum(client_losses) / len(client_losses)
    round_losses.append(avg_train_loss)
    round_accuracies.append(current_accuracy)

    print(f"Status: Acc {current_accuracy*100:.2f}% | Loss {val_loss:.4f}")

# 🔷 FINAL ANALYSIS & PLOTTING
print(f"\n✅ Convergence reached in {r} rounds!")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, r+1), round_losses, color='red', label="Train Loss")
plt.title("Convergence: Loss")
plt.xlabel("Rounds")

plt.subplot(1, 2, 2)
plt.plot(range(1, r+1), [a*100 for a in round_accuracies], color='blue', label="Accuracy")
plt.axhline(y=TARGET_ACCURACY*100, color='green', linestyle='--', label="Target")
plt.title("Convergence: Accuracy")
plt.xlabel("Rounds")

plt.tight_layout()
plt.savefig("convergence_plot.png")
torch.save(global_model.state_dict(), "converged_model.pth")