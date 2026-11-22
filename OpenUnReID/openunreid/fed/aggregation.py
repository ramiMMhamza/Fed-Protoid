"""Server aggregation of Fed-Protoid, Eq. (2) of the paper.

    theta = alpha * theta_s + (1 - alpha) * sum_i w_i * theta_i

theta_s is the pseudo-client (source) model, theta_i the model of client i and
w_i its weight (N_i / sum N for FedAvg, 1 / n for FedFix). The classifier heads
are personalised (Sec. 3.1: "we add to each local device i a personalized
classifier head C_i") and are not aggregated.
"""
import torch

# names of the classifier modules: `classifier` in the ResNet model,
# `fc` in the ViT model (not to be confused with `mlp.fc1` / `mlp.fc2`)
_HEADS = ("classifier", "fc")


def is_classifier_head(name):
    """True for the parameters and buffers of a classifier head."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in _HEADS


def aggregate_models(models, weights=None, alpha=0.2, average_buffers=True):
    """Aggregate the models of one round with Eq. (2).

    Args:
        models: ``[source, client_1, ..., client_n]``, modules with the same
            architecture (the classifier heads may differ in size).
        weights: the n client weights, summing to 1. ``None`` gives each
            client the same weight (FedFix).
        alpha: weight of the source model.
        average_buffers: also average the floating-point buffers, i.e. the
            BatchNorm running statistics. If False they are those of the last
            client.

    Returns:
        ``models[-1]`` holding the aggregated tensors. Its classifier head is
        left as it is. The other models are not modified.
    """
    clients = models[1:]
    if not clients:
        raise ValueError("aggregation needs the source model and at least one client")
    if weights is None:
        coefs = [(1 - alpha) / len(clients)] * len(clients)
    else:
        if len(weights) != len(clients):
            raise ValueError(f"{len(weights)} weights for {len(clients)} clients")
        if abs(sum(weights) - 1) > 1e-6:
            raise ValueError(f"client weights sum to {sum(weights)}, not 1")
        coefs = [(1 - alpha) * w for w in weights]

    target = models[-1]
    parameters = {name for name, _ in target.named_parameters()}
    states = [model.state_dict() for model in models]
    aggregated = {}
    with torch.no_grad():
        for name, value in states[-1].items():
            if is_classifier_head(name) or not value.is_floating_point():
                continue
            if name not in parameters and not average_buffers:
                continue
            total = states[0][name] * alpha
            for state, coef in zip(states[1:], coefs):
                total = total + state[name] * coef
            aggregated[name] = total
        target.load_state_dict(aggregated, strict=False)
    return target
