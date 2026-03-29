"""pytorchexample: A Flower / PyTorch app refactored for Progressive Training."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from pytorchexample.task import Net, load_data, train, test

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, num_partitions, batch_size):
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Retrieve Progressive Params from server
        lr = config.get("lr", 0.01)
        epochs = config.get("local_epochs", 1)
        
        # Load weights
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        trainloader, _ = load_data(self.partition_id, self.num_partitions, self.batch_size)
        
        print(f"  [Client {self.partition_id}] Training: {epochs} epochs, LR: {lr}")
        train_loss = train(self.model, trainloader, int(epochs), float(lr), self.device)
        
        return self.get_parameters(config={}), len(trainloader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        _, valloader = load_data(self.partition_id, self.num_partitions, self.batch_size)
        loss, accuracy = test(self.model, valloader, self.device)
        
        return loss, len(valloader.dataset), {"accuracy": accuracy}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    return FlowerClient(partition_id, num_partitions, batch_size).to_client()

app = ClientApp(client_fn=client_fn)