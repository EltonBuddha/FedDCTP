import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate:
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=args.local_bs,
            shuffle=True,
        )
        self.loss_func = nn.CrossEntropyLoss()

    @staticmethod
    def _current_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    def train(self, net, lr_glob):
        net.train()
        for param in net.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr_glob,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        epoch_losses = []

        for _ in tqdm(range(self.args.local_ep), desc="Local Epochs", leave=False):
            batch_losses = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.long().to(self.args.device)
                optimizer.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        return copy.deepcopy(net.state_dict()), sum(epoch_losses) / len(epoch_losses), self._current_lr(optimizer)

    def train_stage2(self, net, lr_glob, refiner_prefixes, phead_prefixes):
        net.train()
        trainable_prefixes = tuple(refiner_prefixes) + tuple(phead_prefixes)
        for name, param in net.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)

        # requires_grad=False alone does not freeze BN running stats or
        # stochastic operations in frozen feature-extractor Dropout.
        # The CIFAR ConvNet's top-level "dropout" follows trainable fc1
        # and belongs to the refiner pathway: it MUST stay in train mode.
        # Other frozen modules (including MNIST "conv2_drop") stay in eval.
        for module_name, module in net.named_children():
            has_trainable_descendant = any(
                prefix == module_name or prefix.startswith(module_name + ".")
                for prefix in trainable_prefixes
            )
            is_refiner_dropout = (
                module_name == "dropout"
                and isinstance(module, nn.modules.dropout._DropoutNd)
            )
            if not has_trainable_descendant and not is_refiner_dropout:
                module.eval()

        trainable = [param for param in net.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(
            trainable,
            lr=lr_glob,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        epoch_losses = []

        for _ in tqdm(range(self.args.local_ep), desc="Local Epochs", leave=False):
            batch_losses = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.long().to(self.args.device)
                optimizer.zero_grad()
                outputs = net(images)
                p_logits = outputs[-1] if isinstance(outputs, tuple) else outputs
                loss = self.loss_func(p_logits, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        return copy.deepcopy(net.state_dict()), sum(epoch_losses) / len(epoch_losses), self._current_lr(optimizer)
