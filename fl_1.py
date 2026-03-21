import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.client import ClientApp, NumPyClient
from flwr.simulation import run_simulation

# 1. Define the Client Logic
class TaskOneClient(NumPyClient):
    def fit(self, parameters, config):
        print(" -> Client training in progress...")
        return parameters, 1, {}

    def evaluate(self, parameters, config):
        print(" -> Client evaluating...")
        return 0.0, 1, {"accuracy": 1.0}

def client_fn(context: Context):
    return TaskOneClient().to_client()

# 2. Define the Server Logic
def server_fn(context: Context):
    strategy = fl.server.strategy.FedAvg()
    config = ServerConfig(num_rounds=3) # We'll do 3 rounds to show it working
    return ServerAppComponents(strategy=strategy, config=config)

# 3. Create the Apps
client = ClientApp(client_fn=client_fn)
server = ServerApp(server_fn=server_fn)

# 4. Run the Simulation
if __name__ == "__main__":
    print("\n" + "="*30)
    print("STARTING FLOWER SIMULATION")
    print("="*30 + "\n")
    
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=fl.server.strategy.FedAvg(),
    )
    print(hist)
    
    print("\n" + "="*30)
    print("ASSIGNMENT 1: SETUP COMPLETE")
    print("="*30)