import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Fin_CNNMnist(nn.Module):
    def __init__(self, args):
        super(Fin_CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
        self.fc3 = nn.Linear(50, args.num_classes)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        y_g = self.fc2(x)
        y_p = self.fc3(x)
        return y_g, y_p


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, args.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Fin_CNNCifar(nn.Module):
    def __init__(self, args):
        super(Fin_CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, args.num_classes)
        self.fc3 = nn.Linear(64, args.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        y_g = self.fc2(x)
        y_p = self.fc3(x)
        return y_g, y_p

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)


        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))


        out = self.bn2(self.conv2(out))


        out += self.shortcut(x)


        out = F.relu(out)

        return out


class ResNet20(nn.Module):
    def __init__(self, args):
        super(ResNet20, self).__init__()

        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)


        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)


        self.fc = nn.Linear(64, args.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_planes, out_channels, stride))
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        x = F.avg_pool2d(x, 8)


        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Fin_ResNet20(nn.Module):
    def __init__(self, args):
        super(Fin_ResNet20, self).__init__()

        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)


        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)


        self.fc1 = nn.Linear(64, args.num_classes)
        self.fc2 = nn.Linear(64, args.num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []

        layers.append(block(self.in_planes, out_channels, stride))
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        x = F.avg_pool2d(x, 8)


        x = x.view(x.size(0), -1)
        y_g = self.fc1(x)
        y_p = self.fc2(x)

        return y_g, y_p


class CNNRW_FE(nn.Module):
    def __init__(self, in_channels=3):
        super(CNNRW_FE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        y = self.pool(F.relu(self.bn1(self.conv1(x))))
        y = self.pool(F.relu(self.bn2(self.conv2(y))))
        y = y.view(x.size(0), -1)
        return y

class CNNRW_RC(nn.Module):
    def __init__(self, dim_hidden=10816, dim_out=512):
        super(CNNRW_RC, self).__init__()
        self.fc1 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        y = self.fc1(x)
        return y

class CNNRW_CH(nn.Module):
    def __init__(self, num_classes=10, dim_out=512):
        super(CNNRW_CH, self).__init__()
        self.fc2 = nn.Linear(dim_out, num_classes)

    def forward(self, x):
        y = self.fc2(x)
        return y

class CNNRW(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dim_hidden=10816, dim_out=512):
        super(CNNRW, self).__init__()
        self.fe = CNNRW_FE(in_channels)
        self.rc = CNNRW_RC(dim_hidden, dim_out)
        self.ch = CNNRW_CH(num_classes, dim_out)

    def forward(self, x):
        fg = self.fe(x)
        fc = self.rc(fg)
        y = self.ch(fc)
        return y


class Fin_CNNRW(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dim_hidden=10816, dim_out=512):
        super(Fin_CNNRW, self).__init__()
        self.fe = CNNRW_FE(in_channels)
        self.rc = CNNRW_RC(dim_hidden, dim_out)
        self.ch_g = CNNRW_CH(num_classes, dim_out)
        self.ch_p = CNNRW_CH(num_classes, dim_out)

    def forward(self, x):
        fg = self.fe(x)
        fc = self.rc(fg)
        y_g = self.ch_g(fc)
        y_p = self.ch_p(fc)
        return y_g, y_p


class fastText_FE(nn.Module):
    def __init__(self, dim_hidden = 32, padding_idx = 0, vocab_size = 98635):
        super(fastText_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_hidden, padding_idx)

    def forward(self, x):
        y = self.embedding(x)
        return y

class fastText_RC(nn.Module):
    def __init__(self, dim_hidden = 32):
        super(fastText_RC, self).__init__()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)


    def forward(self, x):
        y = self.fc1(x.mean(1))


        return y

class fastText_CH(nn.Module):
    def __init__(self, dim_hidden = 32, num_classes=4):
        super(fastText_CH, self).__init__()
        self.fc2 = nn.Linear(dim_hidden, num_classes)

    def forward(self, x):
        y = self.fc2(x)
        return y

class fastText(nn.Module):
    def __init__(self, dim_hidden = 32, padding_idx = 0, vocab_size = 98635, num_classes = 4):
        super(fastText, self).__init__()
        self.fe = fastText_FE(dim_hidden, padding_idx, vocab_size)
        self.rc = fastText_RC(dim_hidden)
        self.ch = fastText_CH(dim_hidden, num_classes)

    def forward(self, x):
        fg = self.fe(x)
        fc = self.rc(fg)
        y = self.ch(fc)
        return y


class Fin_fastText(nn.Module):
    def __init__(self, dim_hidden = 32, padding_idx = 0, vocab_size = 98635, num_classes = 4):
        super(Fin_fastText, self).__init__()
        self.fe = fastText_FE(dim_hidden, padding_idx, vocab_size)
        self.rc = fastText_RC(dim_hidden)
        self.ch_g = fastText_CH(dim_hidden, num_classes)
        self.ch_p = fastText_CH(dim_hidden, num_classes)

    def forward(self, x):
        fg = self.fe(x)
        fc = self.rc(fg)
        y_g = self.ch_g(fc)
        y_p = self.ch_p(fc)
        return y_g, y_p

