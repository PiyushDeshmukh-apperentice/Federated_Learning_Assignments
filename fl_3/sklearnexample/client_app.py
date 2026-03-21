"""sklearnexample: A Flower / sklearn app."""

import warnings
import os
import requests  # Ensure you have run: pip install requests
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import log_loss

from sklearnexample.task import (
    UNIQUE_LABELS,
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    load_data,
    set_model_params,
)

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and transmit updates."""
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    penalty = context.run_config["penalty"]

    # 1. Initialize Model
    model = create_log_reg_and_instantiate_parameters(penalty)

    # 2. Apply RECEIVED global parameters from Central Server
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # 3. Load the Local Data (On-device data)
    X_train, y_train, _, _ = load_data(partition_id, num_partitions)

    # 4. Local Training Execution
    print(f"\n[Client {partition_id}] 🟢 Starting Local Training...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # 5. Compute Training Metrics
    y_train_pred_proba = model.predict_proba(X_train)
    train_logloss = log_loss(y_train, y_train_pred_proba, labels=UNIQUE_LABELS)
    accuracy = model.score(X_train, y_train)

    # 6. Extract UPDATED parameters for transmission (Assignment 3 requirement)
    updated_ndarrays = get_model_params(model)
    
    # --- VISIBILITY: Send updates to Localhost FastAPI (Optional) ---
    try:
        # We send only the coefficients for visibility
        payload = {
            "partition_id": partition_id,
            "weights": updated_ndarrays[0].tolist(),
            "accuracy": float(accuracy)
        }
        requests.post("http://127.0.0.1:8000/receive_update", json=payload, timeout=1)
        print(f"[Client {partition_id}] 📤 Transmission to FastAPI successful.")
    except Exception:
        # If FastAPI isn't running, we just proceed
        pass

    # --- LOGGING: Record execution to persistent file ---
    log_path = os.path.expanduser("assignment3_execution.txt")
    with open(log_path, "a") as f:
        f.write(f"Round Completed for Client {partition_id} | Accuracy: {accuracy:.4f}\n")

    # 7. Construct and return reply Message back to Flower Central Server
    model_record = ArrayRecord(updated_ndarrays)
    metrics = {
        "num-examples": len(X_train),
        "train_logloss": train_logloss,
        "train_accuracy": accuracy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    print(f"[Client {partition_id}] ✅ Local model update sent to Central Server.")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)

    # Apply received parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    set_model_params(model, ndarrays)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, X_test, y_test = load_data(partition_id, num_partitions)

    # Evaluate the model on local data
    y_test_pred_proba = model.predict_proba(X_test)
    accuracy = model.score(X_test, y_test)
    loss = log_loss(y_test, y_test_pred_proba, labels=UNIQUE_LABELS)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "test_logloss": loss,
        "accuracy": accuracy,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)