def fedavg_aggregate(global_model, client_updates):
    """
    client_updates = [(state_dict, num_samples), ...]
    """

    new_weights = {}
    total_samples = sum(n for _, n in client_updates)

    for key in global_model.state_dict().keys():
        new_weights[key] = sum(
            client_state[key] * (num_samples / total_samples)
            for client_state, num_samples in client_updates
        )

    global_model.load_state_dict(new_weights)

    return global_model