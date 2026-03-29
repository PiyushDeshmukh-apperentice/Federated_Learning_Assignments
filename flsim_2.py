import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from flsim.data.data_sharder import SequentialSharder
from flsim.utils.example_utils import DataLoader, DataProvider, FLModel
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelector,
    UniformlyRandomActiveUserSelectorConfig
)
from flsim.utils.example_utils import MetricsReporter
from flsim.interfaces.metrics_reporter import Channel

# --- 1. Setup Data Pipeline ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

sharder = SequentialSharder(examples_per_shard=500)
fl_data_loader = DataLoader(train_dataset, test_dataset, test_dataset, sharder, batch_size=32)
data_provider = DataProvider(fl_data_loader)

# --- 2. Define a Simple CNN ---
# --- 2. Define a Simple CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # 32x32 -> 28x28 (conv) -> 14x14 (pool)
        self.fc1 = nn.Linear(6 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        return self.fc1(x)

# Create the base PyTorch model
model = SimpleCNN(num_classes=10)

# --- THE FIX: Minimalist FLModel Wrapping ---
# Initialize with ONLY the model to avoid keyword errors
fl_model = FLModel(model)

# Manually attach the loss function (FLSim often looks for 'fl_criterion')
fl_model.fl_criterion = nn.CrossEntropyLoss()

# --- 3. Configure Trainer ---
cfg = SyncTrainerConfig()
cfg.cuda = False
cfg.do_eval = True
cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
cfg.epochs = 1
cfg.train_rounds = 5  # Keep this low for testing on CPU
cfg.users_per_round = 2

# Initialize Trainer and then inject config
trainer = SyncTrainer(model=fl_model)
trainer.cfg = cfg 

# --- 4. Initialize Metrics Reporter ---
# Reduced to just STDOUT to avoid potential TensorBoard path errors on local disk
metrics_reporter = MetricsReporter(channels=[Channel.STDOUT])

# --- 5. Run Training Rounds ---
print("Starting Federated Learning on CPU...")
try:
    final_model, eval_results = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=100,
        distributed_world_size=1
    )
    print("Training Complete!")
    print(f"Final Evaluation Results: {eval_results}")
except Exception as e:
    import traceback
    traceback.print_exc() # This will show the exact line in the library that fails
    print(f"An error occurred during training: {e}")