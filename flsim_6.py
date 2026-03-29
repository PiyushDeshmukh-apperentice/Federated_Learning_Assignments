import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from flsim.data.data_sharder import SequentialSharder
from flsim.utils.example_utils import DataLoader, DataProvider, FLModel
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.utils.example_utils import MetricsReporter
from flsim.interfaces.metrics_reporter import Channel

# --- 1. Manual DP Mechanism ---
def apply_manual_dp(model, clipping_value=1.0, noise_multiplier=0.8):
    """
    Manually mimics Differential Privacy by attaching hooks to gradients.
    This bypasses the need for the native FLSim privacy modules.
    """
    def dp_hook(grad):
        # Step A: L2 Norm Clipping
        # Limits the influence of any single gradient update
        grad_norm = grad.norm(2)
        clip_coef = clipping_value / (grad_norm + 1e-6)
        if clip_coef < 1:
            grad.mul_(clip_coef)
            
        # Step B: Gaussian Noise Injection
        # Adds noise proportional to the sensitivity (clipping value)
        noise = torch.randn_like(grad) * (noise_multiplier * clipping_value)
        return grad + noise

    # Register the hook for every trainable parameter
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(dp_hook)
    return model

# --- 2. Setup Data Pipeline ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

sharder = SequentialSharder(examples_per_shard=500)
fl_data_loader = DataLoader(train_dataset, test_dataset, test_dataset, sharder, batch_size=32)
data_provider = DataProvider(fl_data_loader)

# --- 3. Model Definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        return self.fc1(x)

# --- 4. Initialize Model with Manual DP ---
raw_model = SimpleCNN(num_classes=10)

# APPLY DP MIMICRY HERE
# clipping_value=1.0, noise_multiplier=0.8
privatized_model = apply_manual_dp(raw_model, clipping_value=1.0, noise_multiplier=0.8)

# Wrap for FLSim (using the manual attribute injection to avoid constructor errors)
fl_model = FLModel(privatized_model)
fl_model.fl_criterion = nn.CrossEntropyLoss()

# --- 5. Configure Trainer ---
trainer_config = SyncTrainerConfig()
trainer_config.cuda = False
trainer_config.do_eval = True
trainer_config.epochs = 1
trainer_config.train_rounds = 10
trainer_config.users_per_round = 5

# Notice: We NO LONGER need trainer_config.privacy_setting
trainer = SyncTrainer(model=fl_model)
trainer.cfg = trainer_config

metrics_reporter = MetricsReporter(channels=[Channel.STDOUT])

# --- 6. Run Training ---
print("Starting Manual DP-Federated Learning")
try:
    final_model, eval_results = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=100,
        distributed_world_size=1
    )
    print("Training Complete!")
except Exception as e:
    print(f"An error occurred: {e}")