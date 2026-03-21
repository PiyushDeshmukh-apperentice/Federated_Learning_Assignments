import torch
import torch.nn as nn

def train_client(model, dataloader, epochs=1):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    total_samples = 0
    total_loss = 0

    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_samples += x.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return model.state_dict(), total_samples, avg_loss