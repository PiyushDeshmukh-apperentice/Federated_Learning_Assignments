import torch
import matplotlib.pyplot as plt

from model import RNNModel
from client import train_client
from server import fedavg_aggregate
from dataset import ShakespeareDataset, get_client_dataloader

# 🔷 Load dataset
dataset = ShakespeareDataset("/mnt/StorageHDD/Coding/FL_Assignment/fl_4/data/shakespeare2.txt")

num_clients = 5
rounds = 5

# 🔷 Initialize model
vocab_size = len(dataset.chars)
global_model = RNNModel(vocab_size=vocab_size)

# 🔷 Metrics storage
round_losses = []
round_accuracies = []

# 🔷 Evaluation function
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    total_loss = 0

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


for r in range(rounds):
    print(f"\n--- Round {r+1} ---")

    client_updates = []
    client_losses = []

    for client_id in range(num_clients):
        # 🔷 Copy model
        client_model = RNNModel(vocab_size=vocab_size)
        client_model.load_state_dict(global_model.state_dict())

        # 🔷 Client data
        dataloader = get_client_dataloader(
            dataset,
            client_id,
            num_clients,
            batch_size=32
        )

        # 🔷 Train
        weights, num_samples, loss = train_client(client_model, dataloader)

        client_updates.append((weights, num_samples))
        client_losses.append(loss)

    # 🔷 Aggregate
    global_model = fedavg_aggregate(global_model, client_updates)

    # 🔷 Average training loss
    round_loss = sum(client_losses) / len(client_losses)
    round_losses.append(round_loss)

    # 🔷 Evaluate on full dataset (global performance)
    eval_loader = get_client_dataloader(dataset, 0, 1, batch_size=64)
    acc, val_loss = evaluate(global_model, eval_loader)

    round_accuracies.append(acc)

    print(f"Train Loss: {round_loss:.4f}")
    print(f"Eval Loss : {val_loss:.4f}")
    print(f"Accuracy  : {acc*100:.2f}%")

# 🔷 Plot graphs
plt.figure()

plt.plot(range(1, rounds+1), round_losses, marker='o', label="Train Loss")
plt.plot(range(1, rounds+1), round_accuracies, marker='x', label="Accuracy")

plt.xlabel("Rounds")
plt.title("Federated Learning Metrics")
plt.legend()
plt.grid()

plt.savefig("Federated Learning Metrics")

# 🔷 Save model
torch.save(global_model.state_dict(), "global_model.pth")

print("\n✅ Training Complete. Model saved as global_model.pth")