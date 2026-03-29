import torch
import torch.nn as nn

from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.clients.base_client import ClientConfig
from flsim.optimizers.local_optimizers import LocalOptimizerConfig
from flsim.optimizers.server_optimizers import FedAvgOptimizerConfig


# ---------------------------
# 1. Create simple client datasets
# ---------------------------
def get_data(num_clients=5, samples_per_client=50):
    data = []
    for _ in range(num_clients):
        x = torch.randn(samples_per_client, 10)
        y = (torch.sum(x, dim=1) > 0).long()
        data.append((x, y))
    return data


# ---------------------------
# 2. Model
# ---------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------
# 3. Config
# ---------------------------
def get_config():
    return SyncTrainerConfig(
        epochs=3,
        users_per_round=2,
        client=ClientConfig(
            epochs=1,
            optimizer=LocalOptimizerConfig(
                optimizer="SGD",
                lr=0.1,
            ),
        ),
        server_optimizer=FedAvgOptimizerConfig(lr=1.0),
    )


# ---------------------------
# 4. Train
# ---------------------------
def main():
    model = SimpleModel()
    data = get_data()

    trainer = SyncTrainer(
        model=model,
        train_data=data,   # ✅ directly pass list
        eval_data=data,
        config=get_config(),
    )

    trainer.train()
    print("✅ Training successful!")


if __name__ == "__main__":
    main()