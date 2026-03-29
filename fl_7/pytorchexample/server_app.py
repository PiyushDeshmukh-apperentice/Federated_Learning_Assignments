"""pytorchexample: A Flower / PyTorch app refactored for Progressive Training with logging."""

import torch
from logging import INFO
from flwr.common import Metrics, Context, log, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from pytorchexample.task import Net, load_centralized_dataset, test

def get_on_fit_config_fn():
    def fit_config(server_round: int):
        if server_round <= 10:
            lr, epochs = 0.1, 1
        elif server_round <= 20:
            lr, epochs = 0.01, 3
        else:
            lr, epochs = 0.01, 5
            
        log(INFO, "🚀 PROGRESSIVE SHIFT | Round %d | Config: LR=%s, Epochs=%d", 
            server_round, lr, epochs)
            
        return {"lr": lr, "local_epochs": epochs}
    return fit_config

def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    agg_accuracy = sum(accuracies) / sum(examples)
    log(INFO, "📈 CLIENT AGGREGATION | Aggregated Accuracy: %.4f", agg_accuracy)
    return {"accuracy": agg_accuracy}

def global_evaluate(server_round: int, parameters, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()
    
    # Convert parameters back to torch state_dict
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    
    test_loader = load_centralized_dataset()
    loss, accuracy = test(model, test_loader, device)
    log(INFO, "📊 GLOBAL EVALUATION | Round %d | Loss: %.4f | Accuracy: %.4f", 
        server_round, loss, accuracy)
    return loss, {"accuracy": accuracy}

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config.get("num-server-rounds", 30)
    
    # Initialize parameters on server to avoid get_parameters error
    net = Net()
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in net.state_dict().items()]
    )

    strategy = FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        initial_parameters=initial_parameters,
        on_fit_config_fn=get_on_fit_config_fn(),
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=global_evaluate,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)