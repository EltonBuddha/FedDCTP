import copy
import math
import os
import random
import numpy as np
import torch

import ClientBuild
import ConductorData
from models.Fed import FedWeightedAvg
from models.Nets import CNNMnist, CNNCifar, Fin_CNNMnist, Fin_CNNCifar, ResNet20, Fin_ResNet20
from models.Update import LocalUpdate
from utils.options import args_parser


def _blend_state(current_state, local_state, weight):
    updated = copy.deepcopy(current_state)
    for name, value in current_state.items():
        if torch.is_floating_point(value):
            updated[name] = value + weight * (local_state[name] - value)
        else:
            updated[name] = copy.deepcopy(local_state[name])
    return updated


def _subset_state(state, prefixes):
    prefixes = tuple(prefixes)
    return {k: copy.deepcopy(v) for k, v in state.items() if k.startswith(prefixes)}


def _load_subset(state, subset):
    merged = copy.deepcopy(state)
    for name, value in subset.items():
        merged[name] = copy.deepcopy(value)
    return merged


def _weighted_subset_average(states, weights):
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




def _reset_parameter_modules(net, prefixes):
    """Freshly initialize the parameter-owning modules under selected prefixes."""
    roots = tuple(prefix.rstrip(".") for prefix in prefixes)
    for module_name, module in net.named_modules():
        if not module_name:
            continue
        in_target = any(
            module_name == root or module_name.startswith(root + ".")
            for root in roots
        )
        if in_target and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return net


def train_stage1(net_glob, dp, fe_groups, args):
    global_state = copy.deepcopy(net_glob.state_dict())

    for round_idx in range(args.stage1_rounds):
        # One learning rate per communication round, shared by all clients.
        round_lr = args.lr * (args.lr_decay ** round_idx)
        c_total = len(fe_groups)
        c_sample = math.floor(args.rho1 * c_total)
        if c_sample < 1:
            raise ValueError("floor(rho1 * C) must be at least 1")
        selected_ids = random.sample(range(c_total), c_sample)
        conductor_states = []
        conductor_sizes = []

        for c in selected_ids:
            # The first client (and the complete sequential trajectory) is
            # randomly chosen afresh in each round; seed set in main().
            clients = sorted(fe_groups[c])
            random.shuffle(clients)
            group_size = sum(len(dp.local_train_index[k]) for k in clients)
            conductor_state = copy.deepcopy(global_state)

            for k in clients:
                client_size = len(dp.local_train_index[k])
                local = LocalUpdate(args, dp, dp.local_train_index[k])
                client_model = copy.deepcopy(net_glob).to(args.device)
                client_model.load_state_dict(conductor_state)
                local_state, _, _ = local.train(client_model, round_lr)
                conductor_state = _blend_state(
                    conductor_state,
                    local_state,
                    client_size / group_size,
                )

            conductor_states.append(conductor_state)
            conductor_sizes.append(group_size)

        global_state = FedWeightedAvg(conductor_states, conductor_sizes)
        net_glob.load_state_dict(global_state)

    return net_glob


def prepare_stage2_model(stage1_model, args):
    if args.model == "res":
        model = Fin_ResNet20(args).to(args.device)
        model.conv1.load_state_dict(stage1_model.conv1.state_dict())
        model.bn1.load_state_dict(stage1_model.bn1.state_dict())
        model.layer1.load_state_dict(stage1_model.layer1.state_dict())
        model.layer2.load_state_dict(stage1_model.layer2.state_dict())
        model.fc1.load_state_dict(stage1_model.fc.state_dict())
        return model, ("layer3.",), ("fc2.",)

    if args.model == "cnn" and args.dataset_name in {"cifar10", "cifar100"}:
        model = Fin_CNNCifar(args).to(args.device)
        model.conv1.load_state_dict(stage1_model.conv1.state_dict())
        model.bn1.load_state_dict(stage1_model.bn1.state_dict())
        model.conv2.load_state_dict(stage1_model.conv2.state_dict())
        model.bn2.load_state_dict(stage1_model.bn2.state_dict())
        model.conv3.load_state_dict(stage1_model.conv3.state_dict())
        model.bn3.load_state_dict(stage1_model.bn3.state_dict())
        model.fc2.load_state_dict(stage1_model.fc2.state_dict())
        return model, ("fc1.",), ("fc3.",)

    model = Fin_CNNMnist(args).to(args.device)
    model.conv1.load_state_dict(stage1_model.conv1.state_dict())
    model.conv2.load_state_dict(stage1_model.conv2.state_dict())
    model.fc2.load_state_dict(stage1_model.fc2.state_dict())
    return model, ("fc1.",), ("fc3.",)


def train_stage2(model, dp, rc_groups, args, refiner_prefixes, phead_prefixes):
    base_state = copy.deepcopy(model.state_dict())

    # Algorithm 1 initializes one private P-Head per client and reinitializes
    # one group-shared refiner per RC conductor.  Do not clone one identical
    # random draw across all clients/conductors.
    client_states = {}
    for k in range(dp.size_device):
        fresh_client = _reset_parameter_modules(copy.deepcopy(model), phead_prefixes)
        private_phead = _subset_state(fresh_client.state_dict(), phead_prefixes)
        client_states[k] = _load_subset(base_state, private_phead)

    conductor_refiners = {}
    for c in range(len(rc_groups)):
        fresh_group = _reset_parameter_modules(copy.deepcopy(model), refiner_prefixes)
        conductor_refiners[c] = _subset_state(fresh_group.state_dict(), refiner_prefixes)

    for round_idx in range(args.stage2_rounds):
        # Continue the same per-round schedule instead of restarting it.
        global_round_idx = args.stage1_rounds + round_idx
        round_lr = args.lr * (args.lr_decay ** global_round_idx)
        k_sample = math.floor(args.rho2 * dp.size_device)
        if k_sample < 1:
            raise ValueError("floor(rho2 * N) must be at least 1")
        sampled_clients = set(random.sample(range(dp.size_device), k_sample))

        for c, group in enumerate(rc_groups):
            active_clients = sorted(sampled_clients.intersection(group))
            if not active_clients:
                continue

            refiner_updates = []
            data_sizes = []

            for k in active_clients:
                start_state = _load_subset(client_states[k], conductor_refiners[c])
                client_model = copy.deepcopy(model).to(args.device)
                client_model.load_state_dict(start_state)
                local = LocalUpdate(args, dp, dp.local_train_index[k])
                local_state, _, _ = local.train_stage2(
                    client_model,
                    round_lr,
                    refiner_prefixes,
                    phead_prefixes,
                )
                client_states[k] = copy.deepcopy(local_state)
                refiner_updates.append(_subset_state(local_state, refiner_prefixes))
                data_sizes.append(len(dp.local_train_index[k]))

            conductor_refiners[c] = _weighted_subset_average(refiner_updates, data_sizes)

    for c, group in enumerate(rc_groups):
        for k in group:
            client_states[k] = _load_subset(client_states[k], conductor_refiners[c])

    return client_states, conductor_refiners


def build_stage1_model(args):
    if args.model == "res":
        return ResNet20(args).to(args.device)
    if args.model == "cnn" and args.dataset_name in {"cifar10", "cifar100"}:
        return CNNCifar(args).to(args.device)
    return CNNMnist(args).to(args.device)


def main():
    args = args_parser()
    if args.stage1_rounds < 1 or args.stage2_rounds < 1:
        raise ValueError("stage1_rounds and stage2_rounds must be positive")
    if args.num_FE < 1 or args.num_users < 1:
        raise ValueError("num_FE and num_users must be positive")
    if args.num_RC < 1:
        raise ValueError("num_RC must be positive")
    if args.threshold < 0:
        raise ValueError("threshold must be non-negative")
    if args.alpha <= 0:
        raise ValueError("alpha must be positive")
    if args.local_ep < 1 or args.local_bs < 1:
        raise ValueError("local_ep and local_bs must be positive")
    if args.lr <= 0 or args.lr_decay <= 0:
        raise ValueError("lr and lr_decay must be positive")
    if not (0 < args.rho1 <= 1 and 0 < args.rho2 <= 1):
        raise ValueError("rho1 and rho2 must lie in (0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )

    supported_datasets = {"mnist", "FMNIST", "cifar10", "cifar100"}
    if args.dataset_name not in supported_datasets:
        raise NotImplementedError(
            "This compact Main.py only wires MNIST/FMNIST/CIFAR experiments. "
            "The paper's OfficeCaltech10, DomainNet, and AGNews experiment recipes "
            "are intentionally not bundled in this reference entry point."
        )
    if args.model == "res" and args.dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError("The bundled ResNet20 entry point is configured for CIFAR-10/100.")

    dp = ClientBuild.DataProcessor(args)
    dp.get_input(args.dataset_name)
    if args.num_classes is None:
        args.num_classes = dp.size_class
    if args.num_channels is None:
        args.num_channels = 1 if args.dataset_name in {"mnist", "FMNIST"} else 3
    dp.gen_local_imbalance_hsu(args.num_users, args.alpha)
    dp.type = "train"

    stage1_model = build_stage1_model(args)
    fe = ConductorData.ConductorData(dp, args)
    fe.assign_clients()
    stage1_model = train_stage1(stage1_model, dp, fe.conductor, args)

    stage2_model, refiner_prefixes, phead_prefixes = prepare_stage2_model(stage1_model, args)
    rc = ConductorData.ConductorData(dp, args)
    rc.assign_clients_to()
    client_states, conductor_refiners = train_stage2(
        stage2_model,
        dp,
        rc.conductor,
        args,
        refiner_prefixes,
        phead_prefixes,
    )

    os.makedirs("saved_params", exist_ok=True)
    torch.save(stage1_model.state_dict(), "saved_params/global_model.pth")
    for k, state in client_states.items():
        torch.save(state, f"saved_params/client_{k}.pth")

    return stage1_model, client_states, conductor_refiners


if __name__ == "__main__":
    main()
