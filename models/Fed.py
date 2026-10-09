import copy
import torch


def FedAvg(states):
    return FedWeightedAvg(states, [1.0] * len(states))


def FedWeightedAvg(states, weights):
    total = float(sum(weights))
    out = copy.deepcopy(states[0])
    for name in out:
        if torch.is_floating_point(out[name]):
            out[name] = sum(
                state[name] * (weight / total)
                for state, weight in zip(states, weights)
            )
        else:
            out[name] = copy.deepcopy(states[0][name])
    return out
