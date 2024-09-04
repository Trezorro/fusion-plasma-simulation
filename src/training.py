import torch
from src.config import get_current_config

C = get_current_config()


def validate(model, data_loader, criterion):
    n_samples = len(data_loader.dataset)
    loss = 0
    model.eval()
    with torch.inference_mode():
        for batch_idx, (shot_number, controls, observables) in enumerate(data_loader):
            partial_observables = torch.zeros_like(observables)
            partial_observables[:, :-C["forecast_horizon"]] = observables[:, :-C["forecast_horizon"]]
            # Input: (batch_size, seq_length, variables)
            # inputs = torch.cat((controls, partial_observables), dim=2)
            outputs = model(controls, observables)[:, -C["forecast_horizon"]:]
            loss += criterion(outputs, observables[:, -C["forecast_horizon"]:]).item()

        mean_loss = loss / n_samples
    return mean_loss
