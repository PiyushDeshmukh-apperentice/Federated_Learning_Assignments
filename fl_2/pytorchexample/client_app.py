"""pytorchexample: A Flower / PyTorch app."""

import torch
import logging
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    
    # 1. VISIBILITY: Identify the Client and Round
    partition_id = context.node_config["partition-id"]
    server_round = msg.content.get("config", {}).get("server_round", "?")
    
    # PERSISTENT LOGGING: Mark the start of a round for this specific client
    with open("fl_results.txt", "a") as f:
        f.write(f"\n>>> [ROUND {server_round}] CLIENT {partition_id} ACTIVE <<<\n")

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content.get("config", {}).get("lr", 0.01),
        device,
    )

    # PERSISTENT LOGGING: Confirm update sent
    with open("fl_results.txt", "a") as f:
        f.write(f"[Client {partition_id}] Update packaged and sent to server.\n")

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    partition_id = context.node_config["partition-id"]
    
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # PERSISTENT LOGGING: Detailed Evaluation Result
    with open("/mnt/StorageHDD/Coding/FL_Assignment/fl_2/fl_results.txt", "a") as f:
        f.write(f"[EVALUATION] Client {partition_id} | Accuracy: {eval_acc*100:.2f}% | Loss: {eval_loss:.4f}\n")

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)