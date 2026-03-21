import numpy as np
import os
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression

# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS = [0, 1, 2]
FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]

def get_model_params(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    
    # --- VISIBILITY FOR ASSIGNMENT 3 ---
    # Print the first few coefficients to the console for verification
    print(f"\n[LOCAL UPDATE] Extracting weights for transmission...")
    print(f" -> Coefficients shape: {model.coef_.shape}")
    print(f" -> Sample Weights (first 2): {model.coef_[0][:2]}")
    
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklearn LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros."""
    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def create_log_reg_and_instantiate_parameters(penalty):
    # Assignment requirement: Ensure warm_start is True to allow incremental updates
    model = LogisticRegression(
        penalty=penalty,
        max_iter=5,  # Increased local epochs slightly for better convergence
        warm_start=True,
        solver="saga",
    )
    set_initial_params(model, n_features=len(FEATURES), n_classes=len(UNIQUE_LABELS))
    return model


fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="hitorilabs/iris", partitioners={"train": partitioner}
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    
    # Shuffle data locally to ensure better training
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    X = dataset[FEATURES]
    y = dataset["species"]
    
    # Split the on-edge data: 80% train, 20% test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # --- LOGGING TO FILE ---
    # This creates a record of the data size being used locally
    log_path = os.path.expanduser("assignment3_execution.txt")
    with open(log_path, "a") as f:
        f.write(f"Partition {partition_id}: Loaded {len(X_train)} training samples.\n")
        
    return X_train.values, y_train.values, X_test.values, y_test.values