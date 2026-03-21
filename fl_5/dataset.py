import torch
from torch.utils.data import Dataset, DataLoader, Subset

class ShakespeareDataset(Dataset):
    def __init__(self, file_path, seq_length=50):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        self.chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}

        encoded = [self.char2idx[c] for c in text]

        self.inputs = []
        self.targets = []

        for i in range(len(encoded) - seq_length):
            self.inputs.append(encoded[i:i+seq_length])
            self.targets.append(encoded[i+seq_length])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


# 👇 ADD THIS FUNCTION HERE
def get_client_dataloader(dataset, client_id, num_clients, batch_size=32):
    data_per_client = len(dataset) // num_clients

    start = client_id * data_per_client
    end = start + data_per_client

    subset = Subset(dataset, list(range(start, end)))

    return DataLoader(subset, batch_size=batch_size, shuffle=True)