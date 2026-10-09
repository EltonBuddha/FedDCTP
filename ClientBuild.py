import os
import pickle
import struct
import collections
import numpy as np
from torchvision import transforms
from scipy import special as sp
class DataProcessor:
    def __init__(self, args=None):
        self.args = args
        self.divide_data_73 = True if args is None else bool(args.divide_data_73)

        self.train_feature = None
        self.train_label = None
        self.test_feature = None
        self.test_label = None

        self.global_train_feature = None
        self.global_train_label = None
        self.global_test_feature = None
        self.global_test_label = None

        self.local_train_feature = None
        self.local_train_label = None
        self.local_train_index = None

        self.local_test_feature = None
        self.local_test_label = None
        self.local_test_index = None

        self.size_class = None
        self.size_device = None
        self.size_feature = None

        self.train_transform = None
        self.test_transform = None

        self.type = 'train'
        self.data_source = None

        if args is None:
            output_dir = "FedDCTP_Management_Information"
        else:
            output_dir = (
                f"{args.dataset_name}_{args.model}_alpha{args.alpha}_"
                f"client_{args.num_users}_Management_Information"
            )
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __len__(self):
        if self.type == 'train':
            return len(self.global_train_label)
        else:
            return len(self.global_test_label)

    def __getitem__(self, idx):
        if self.type == 'train':
            feature, label = self.global_train_feature[idx], self.global_train_label[idx]
        else:
            feature, label = self.global_test_feature[idx], self.global_test_label[idx]
        if self.data_source == "cifar10":
            # CIFAR pixels are 0..255; ToPILImage expects uint8, not float32.
            feature = feature.reshape(32, 32, 3).astype(np.uint8)
        elif self.data_source == "cifar100":
            feature = feature.reshape(32, 32, 3).astype(np.uint8)
        elif self.data_source == "mnist":
            feature = feature.reshape(28, 28, 1).astype(np.uint8)
        elif self.data_source == "FMNIST":
            feature = feature.reshape(28, 28, 1).astype(np.uint8)
        if self.type == 'train':
            img = self.train_transform(feature)
        else:
            img = self.test_transform(feature)
        return img, label


    def get_input(self, name):
        self.__init__(self.args)
        self.data_source = name
        if name == 'cifar10':
            if self.divide_data_73:

                dimension_size = 3072
                num_classes = 10
                num_train_per_class = 800


                self.train_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.train_label = np.array([], dtype=np.int64)
                self.test_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.test_label = np.array([], dtype=np.int64)


                for i in range(1, 6):
                    with open('./data/cifar10/data_batch_{}'.format(i), 'rb') as fo:
                        dic = pickle.load(fo, encoding='bytes')
                    self.train_feature = np.vstack((self.train_feature, dic[b'data']))
                    self.train_label = np.hstack((self.train_label, np.array(dic[b'labels'], dtype=np.int64)))


                with open('./data/cifar10/test_batch', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                original_test_feature = dic[b'data']
                original_test_label = np.array(dic[b'labels'], dtype=np.int64)


                self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)
                original_test_feature = original_test_feature.reshape(len(original_test_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                original_test_feature = original_test_feature.reshape(len(original_test_feature), -1)


                train_features_per_class = []
                train_labels_per_class = []
                test_features_per_class = []
                test_labels_per_class = []


                for class_id in range(num_classes):

                    class_indices = np.where(self.train_label == class_id)[0]


                    test_class_indices = class_indices[:num_train_per_class]
                    train_class_indices = class_indices[num_train_per_class:]


                    test_features_per_class.append(self.train_feature[test_class_indices])
                    test_labels_per_class.append(self.train_label[test_class_indices])


                    train_features_per_class.append(self.train_feature[train_class_indices])
                    train_labels_per_class.append(self.train_label[train_class_indices])


                self.train_feature = np.vstack(train_features_per_class)
                print(len(self.train_feature))
                self.train_label = np.hstack(train_labels_per_class)
                print(len(self.train_label))
                self.test_feature = np.vstack(test_features_per_class + [original_test_feature])
                print(len(self.test_feature))
                self.test_label = np.hstack(test_labels_per_class + [original_test_label])
                print(len(self.test_label))


                self.train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                self.test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

            else:
                dimension_size = 3072
                self.train_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.train_label = np.array([], dtype=np.int64)
                for i in range(1, 6):
                    with open('./data/cifar10/data_batch_{}'.format(i), 'rb') as fo:
                        dic = pickle.load(fo, encoding='bytes')
                    self.train_feature = np.vstack((self.train_feature, dic[b'data']))
                    self.train_label = np.hstack((self.train_label, np.array(dic[b'labels'], dtype=np.int64)))
                self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)
                with open('./data/cifar10/test_batch', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.test_feature = dic[b'data']
                self.test_label = np.array(dic[b'labels'], dtype=np.int64)
                self.test_feature = self.test_feature.reshape(len(self.test_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.test_feature = self.test_feature.reshape(len(self.test_feature), -1)
                self.train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                self.test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        elif name == 'cifar100':

            if self.divide_data_73:
                dimension_size = 3072
                num_classes = 100
                num_train_per_class = 80


                self.train_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.train_label = np.array([], dtype=np.int64)
                self.test_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.test_label = np.array([], dtype=np.int64)


                with open('./data/cifar100/train', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.train_feature = dic[b'data']
                self.train_label = np.array(dic[b'fine_labels'], dtype=np.int64)


                with open('./data/cifar100/test', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                original_test_feature = dic[b'data']
                original_test_label = np.array(dic[b'fine_labels'], dtype=np.int64)


                self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)
                original_test_feature = original_test_feature.reshape(len(original_test_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                original_test_feature = original_test_feature.reshape(len(original_test_feature), -1)

                train_features_per_class = []
                train_labels_per_class = []
                test_features_per_class = []
                test_labels_per_class = []


                for class_id in range(num_classes):

                    class_indices = np.where(self.train_label == class_id)[0]


                    test_class_indices = class_indices[:num_train_per_class]
                    train_class_indices = class_indices[num_train_per_class:]


                    test_features_per_class.append(self.train_feature[test_class_indices])
                    test_labels_per_class.append(self.train_label[test_class_indices])


                    train_features_per_class.append(self.train_feature[train_class_indices])
                    train_labels_per_class.append(self.train_label[train_class_indices])


                self.train_feature = np.vstack(train_features_per_class)
                print(len(self.train_feature))
                self.train_label = np.hstack(train_labels_per_class)
                print(len(self.train_label))
                self.test_feature = np.vstack(test_features_per_class + [original_test_feature])
                print(len(self.test_feature))
                self.test_label = np.hstack(test_labels_per_class + [original_test_label])
                print(len(self.test_label))


                self.train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                self.test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])


            else:
                dimension_size = 3072
                self.train_feature = np.empty((0, dimension_size), dtype=np.int64)
                self.train_label = np.array([], dtype=np.int64)

                with open('./data/cifar100/train', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.train_feature = dic[b'data']
                self.train_label = np.array(dic[b'fine_labels'], dtype=np.int64)

                self.train_feature = self.train_feature.reshape(len(self.train_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.train_feature = self.train_feature.reshape(len(self.train_feature), -1)

                with open('./data/cifar100/test', 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                self.test_feature = dic[b'data']
                self.test_label = np.array(dic[b'fine_labels'], dtype=np.int64)

                self.test_feature = self.test_feature.reshape(len(self.test_feature), 3, 32, 32).transpose(0, 2, 3, 1)
                self.test_feature = self.test_feature.reshape(len(self.test_feature), -1)

                self.train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                self.test_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])

        elif name == 'mnist':
            def load_mnist(path, kind='train'):
                labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
                images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))
                with open(labels_path, 'rb') as lbpath:
                    magic, n = struct.unpack('>II', lbpath.read(8))
                    labels = np.fromfile(lbpath, dtype=np.uint8)

                with open(images_path, 'rb') as imgpath:
                    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

                return images, labels

            self.train_feature, self.train_label = load_mnist('./data/mnist', 'train')
            self.test_feature, self.test_label = load_mnist('./data/mnist', 't10k')
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        elif name == 'FMNIST':
            def load_mnist(path, kind='train'):
                labels_path = os.path.join(path, '{}-labels-idx1-ubyte'.format(kind))
                images_path = os.path.join(path, '{}-images-idx3-ubyte'.format(kind))
                with open(labels_path, 'rb') as lbpath:
                    magic, n = struct.unpack('>II', lbpath.read(8))
                    labels = np.fromfile(lbpath, dtype=np.uint8)

                with open(images_path, 'rb') as imgpath:
                    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
                    images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

                return images, labels

            self.train_feature, self.train_label = load_mnist('./data/FMNIST', 'train')
            self.test_feature, self.test_label = load_mnist('./data/FMNIST', 't10k')
            self.train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        else:
            raise ValueError(
                "The compact loader bundles only MNIST/FMNIST/CIFAR-10/CIFAR-100. "
                "Dataset-specific pipelines for OfficeCaltech10, DomainNet, and AGNews "
                "used in the paper are not included in this reference package."
            )

        self.size_class = len(set(self.train_label) | set(self.test_label))
        self.size_feature = self.train_feature.shape[1]

        self.train_feature = self.train_feature.astype(int)
        self.train_label = self.train_label.astype(int)
        self.test_feature = self.test_feature.astype(int)
        self.test_label = self.test_label.astype(int)


    @staticmethod
    def get_size_difference(arr):
        for i, a in enumerate(arr):
            print('the {}th device size: {}'.format(i, len(a)))


    def get_local_difference(self, arr):
        n = len(arr)
        res = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                res[i][j] = self.get_kl_divergence(arr[i], arr[j])
        return res

    def get_global_difference(self, arr):
        c1 = collections.Counter(arr).values()
        return [0] * (self.size_class - len(c1)) + sorted(c1)

    @staticmethod
    def get_kl_divergence(input1, input2):
        c1, c2 = collections.Counter(input1), collections.Counter(input2)
        keys = sorted(set(c1.keys()).union(c2.keys()))
        p = np.array([c1[k] for k in keys], dtype=np.float64)
        q = np.array([c2[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        q = q / q.sum()
        eps = 1e-12
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(sp.rel_entr(p, q)))

    @staticmethod
    def get_js_divergence(input1, input2):
        c1, c2 = collections.Counter(input1), collections.Counter(input2)
        keys = sorted(set(c1.keys()).union(c2.keys()))
        p = np.array([c1[k] for k in keys], dtype=np.float64)
        q = np.array([c2[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        eps = 1e-12
        p = np.maximum(p, eps)
        q = np.maximum(q, eps)
        m = np.maximum(m, eps)
        return float(0.5 * np.sum(sp.rel_entr(p, m)) + 0.5 * np.sum(sp.rel_entr(q, m)))


    def gen_local_imbalance(self, num_device, device_size, alpha):


        self.size_device = num_device
        self.local_train_feature = []
        self.local_train_label = []

        self.local_train_index = []


        feature_by_class = []
        for i in range(self.size_class):
            need_idx = np.where(self.train_label == i)[0]
            feature_by_class.append(self.train_feature[need_idx])

        remain_size = int(device_size * alpha)
        sample_size = device_size - remain_size

        sample_feature_pool = np.array([], dtype=np.int64)
        sample_label_pool = np.array([], dtype=np.int64)


        for i in range(self.size_class):
            need_idx = np.arange(len(feature_by_class[i]))
            np.random.shuffle(need_idx)
            step = -1
            for j in range(i, self.size_device, self.size_class):
                step += 1
                select_idx = need_idx[step*remain_size:(step+1)*remain_size]
                self.local_train_feature.append(feature_by_class[i][select_idx])
                self.local_train_label.append(np.repeat(i, remain_size))


            select_idx = need_idx[(step + 1) * remain_size:]
            if sample_feature_pool.size:
                sample_feature_pool = np.vstack([sample_feature_pool, feature_by_class[i][select_idx]])
            else:
                sample_feature_pool = feature_by_class[i][select_idx]
            sample_label_pool = np.hstack([sample_label_pool, np.repeat(i, len(select_idx))])


        need_idx = np.arange(len(sample_feature_pool))
        np.random.shuffle(need_idx)
        step = -1
        for i in range(self.size_device):
            step += 1
            select_idx = need_idx[step*sample_size:(step+1)*sample_size]
            if self.local_train_feature[i].size:
                self.local_train_feature[i] = np.vstack([self.local_train_feature[i], sample_feature_pool[select_idx]])
            else:
                self.local_train_feature[i] = sample_feature_pool[select_idx]
            self.local_train_label[i] = np.hstack([self.local_train_label[i], sample_label_pool[select_idx]])
        self.refresh_global_data()

    def gen_local_imbalance_hsu(self, num_device, alpha):
        """Dirichlet label-skew partition with matched train/test distributions.

        For each class, one Dirichlet draw allocates both the training and test
        samples across clients.  This keeps every client's P-Test distribution
        aligned with its local training distribution, as required by the paper's
        evaluation protocol.
        """
        self.size_device = num_device
        self.local_train_feature = [
            np.empty((0, self.size_feature), dtype=self.train_feature.dtype)
            for _ in range(num_device)
        ]
        self.local_train_label = [np.array([], dtype=np.int64) for _ in range(num_device)]
        self.local_test_feature = [
            np.empty((0, self.size_feature), dtype=self.test_feature.dtype)
            for _ in range(num_device)
        ]
        self.local_test_label = [np.array([], dtype=np.int64) for _ in range(num_device)]

        train_class_counts = np.zeros((num_device, self.size_class), dtype=int)
        test_class_counts = np.zeros((num_device, self.size_class), dtype=int)

        for class_id in range(self.size_class):
            train_idx = np.where(self.train_label == class_id)[0]
            test_idx = np.where(self.test_label == class_id)[0]
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)

            q = np.random.dirichlet([alpha] * num_device)
            train_counts = (q * len(train_idx)).astype(int)
            test_counts = (q * len(test_idx)).astype(int)
            train_counts[np.argmax(q)] += len(train_idx) - int(train_counts.sum())
            test_counts[np.argmax(q)] += len(test_idx) - int(test_counts.sum())

            train_pos = 0
            test_pos = 0
            for client_id in range(num_device):
                n_train = int(train_counts[client_id])
                n_test = int(test_counts[client_id])
                client_train_idx = train_idx[train_pos:train_pos + n_train]
                client_test_idx = test_idx[test_pos:test_pos + n_test]
                train_pos += n_train
                test_pos += n_test

                if n_train:
                    self.local_train_feature[client_id] = np.vstack([
                        self.local_train_feature[client_id],
                        self.train_feature[client_train_idx],
                    ])
                    self.local_train_label[client_id] = np.hstack([
                        self.local_train_label[client_id],
                        np.repeat(class_id, n_train),
                    ])
                if n_test:
                    self.local_test_feature[client_id] = np.vstack([
                        self.local_test_feature[client_id],
                        self.test_feature[client_test_idx],
                    ])
                    self.local_test_label[client_id] = np.hstack([
                        self.local_test_label[client_id],
                        np.repeat(class_id, n_test),
                    ])

                train_class_counts[client_id, class_id] = n_train
                test_class_counts[client_id, class_id] = n_test

        if any(len(labels) == 0 for labels in self.local_train_label):
            raise ValueError(
                "Dirichlet partition produced an empty training client. "
                "Use a different seed/alpha or fewer clients."
            )

        output_file = os.path.join(self.output_dir, "client_data_summary.txt")
        with open(output_file, "w") as f:
            f.write("Client data allocation summary:\n")
            for client_id in range(num_device):
                f.write(f"Client {client_id}:\n")
                f.write(f"  Train data: {len(self.local_train_feature[client_id])} samples\n")
                for class_id in range(self.size_class):
                    f.write(
                        f"    Class {class_id}: {train_class_counts[client_id, class_id]} samples\n"
                    )
                f.write(f"  Test data: {len(self.local_test_feature[client_id])} samples\n")
                for class_id in range(self.size_class):
                    f.write(
                        f"    Class {class_id}: {test_class_counts[client_id, class_id]} samples\n"
                    )

        self.refresh_global_data()

    def gen_size_imbalance(self, list_size):


        self.size_device = len(list_size)
        self.local_train_feature = []
        self.local_train_label = []

        self.local_train_index = []

        need_idx = np.arange(len(self.train_feature))
        np.random.shuffle(need_idx)
        cur_idx = 0
        for s in list_size:
            self.local_train_feature.append(self.train_feature[need_idx[cur_idx:cur_idx+s]])
            self.local_train_label.append(self.train_label[need_idx[cur_idx:cur_idx+s]])
            cur_idx += s
        self.refresh_global_data()

    def gen_global_imbalance(self, num_device, device_size, num_each_class):


        self.size_device = num_device
        self.local_train_feature = []
        self.local_train_label = []

        feature_by_class = []
        for i in range(self.size_class):
            need_idx = np.where(self.train_label == i)[0]
            feature_by_class.append(self.train_feature[need_idx])
        sample_feature_pool = np.array([], dtype=np.int64)
        sample_label_pool = np.array([], dtype=np.int64)

        for i in range(self.size_class):
            need_idx = np.arange(len(feature_by_class[i]))
            np.random.shuffle(need_idx)
            if sample_feature_pool.size:
                sample_feature_pool = np.vstack([sample_feature_pool,
                                                 feature_by_class[i][need_idx[:num_each_class[i]]]])
            else:
                sample_feature_pool = feature_by_class[i][need_idx[:num_each_class[i]]]
            sample_label_pool = np.hstack([sample_label_pool, np.repeat(i, num_each_class[i])])

        need_idx = np.arange(len(sample_feature_pool))
        np.random.shuffle(need_idx)
        step = -1
        for i in range(self.size_device):
            step += 1
            select_idx = need_idx[step*device_size:(step+1)*device_size]
            self.local_train_feature.append(sample_feature_pool[select_idx])
            self.local_train_label.append(sample_label_pool[select_idx])
        self.refresh_global_data()

    def refresh_global_data(self):


        self.global_train_feature = np.empty((0, self.size_feature), dtype=np.int64)
        self.global_train_label = np.array([], dtype=np.int64)
        self.local_train_index = []


        self.global_test_feature = np.empty((0, self.size_feature), dtype=np.int64)
        self.global_test_label = np.array([], dtype=np.int64)
        self.local_test_index = []

        idx_start_train = 0
        idx_start_test = 0

        for i in range(self.size_device):

            self.global_train_feature = np.vstack([self.global_train_feature, self.local_train_feature[i]])
            self.global_train_label = np.hstack([self.global_train_label, self.local_train_label[i]])
            self.local_train_index.append(np.arange(idx_start_train, idx_start_train + len(self.local_train_label[i])))
            idx_start_train += len(self.local_train_label[i])


            self.global_test_feature = np.vstack([self.global_test_feature, self.local_test_feature[i]])
            self.global_test_label = np.hstack([self.global_test_label, self.local_test_label[i]])
            self.local_test_index.append(np.arange(idx_start_test, idx_start_test + len(self.local_test_label[i])))
            idx_start_test += len(self.local_test_label[i])

