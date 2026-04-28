import flwr as fl
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from labels import NUM_LABELS

MODEL_PATH = "./base_model"
previous_weights = None

def compute_weight_change(old, new):
    change = 0.0
    for w1, w2 in zip(old, new):
        change += np.sum(np.abs(w1 - w2))
    return change

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global previous_weights
        print("\n==============================")
        print(f"Federated Round {server_round}")
        print("==============================")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is None:
            return None, {}
        new_weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
        if previous_weights is not None:
            change = compute_weight_change(previous_weights, new_weights)
            print(f"Global weight change: {change:.6f}")
        previous_weights = new_weights
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=NUM_LABELS
        )
        params_dict = zip(model.state_dict().keys(), new_weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.save_pretrained("global_model")
        print("Global model saved.")
        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)