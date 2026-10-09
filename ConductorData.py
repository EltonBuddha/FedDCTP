import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans


class ConductorData:
    def __init__(self, dp, args):
        self.dp = dp
        self.num_fe = max(1, int(args.num_FE))
        self.num_rc = max(1, int(args.num_RC))
        self.threshold = float(args.threshold)
        self.epsilon = float(args.epsilon)
        self.seed = int(getattr(args, "seed", 1))
        if (not np.isfinite(self.epsilon) and not np.isposinf(self.epsilon)) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive or +inf (no privacy noise)")
        self.conductor = []
        self.output_dir = (
            f"{args.dataset_name}_{args.model}_alpha{args.alpha}_"
            f"client_{args.num_users}_Management_Information"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=np.float64)
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s <= 0:
            return np.ones_like(x) / len(x)
        return x / s

    def _label_distribution(self, labels):
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=self.dp.size_class)
        return self._normalize(counts)

    @staticmethod
    def _histogram_sensitivity(n_samples):
        """Appendix-C L1 sensitivity under fixed-size label replacement: 2/n_k."""
        if n_samples <= 0:
            raise ValueError("Appendix-C privacy requires a non-empty client dataset")
        return 2.0 / n_samples

    def perturb_label_distribution(self, labels):
        if len(labels) == 0:
            raise ValueError("Cannot privatize an empty client label histogram")
        p = self._label_distribution(labels)
        if np.isposinf(self.epsilon):
            return p
        sensitivity = self._histogram_sensitivity(len(labels))
        noise = np.random.laplace(
            loc=0.0,
            scale=sensitivity / self.epsilon,
            size=p.shape,
        )
        # Non-negative projection and renormalization are DP postprocessing.
        return self._normalize(p + noise)

    def _private_distributions(self):
        """Release each client's privatized histogram once and reuse it for FE/RC.

        The cache belongs to the DataProcessor rather than a conductor: FE and RC
        coordinators may be separate Python objects but must observe the same
        single DP release (otherwise the privacy budgets compose).
        """
        settings = (self.dp.size_device, self.dp.size_class, self.epsilon,
                    tuple(len(labels) for labels in self.dp.local_train_label))
        cache = getattr(self.dp, "_feddctp_private_histograms", None)
        if cache is None:
            private_p = np.vstack([
                self.perturb_label_distribution(self.dp.local_train_label[k])
                for k in range(self.dp.size_device)
            ])
            self.dp._feddctp_private_histograms = (settings, private_p)
        else:
            cached_settings, private_p = cache
            if cached_settings != settings:
                raise ValueError(
                    "FE and RC grouping must reuse the same privatized histograms "
                    "with identical epsilon and client sizes; use a new DataProcessor "
                    "for a different privacy setting."
                )
        return private_p.copy()

    @staticmethod
    def _kl(p, q):
        eps = 1e-12
        p = np.maximum(np.asarray(p, dtype=np.float64), eps)
        q = np.maximum(np.asarray(q, dtype=np.float64), eps)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def _js(p, q):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        return 0.5 * ConductorData._kl(p, m) + 0.5 * ConductorData._kl(q, m)

    def _save_groups(self, filename):
        with open(os.path.join(self.output_dir, filename), "w") as f:
            for c, clients in enumerate(self.conductor):
                f.write(f"conductor {c} manages clients: {sorted(clients)}\n")

    def assign_clients(self, balance=True):
        if not balance:
            self.conductor = [{k} for k in range(self.dp.size_device)]
            self._save_groups("Conductor_FE_clients.txt")
            return

        private_p = self._private_distributions()
        sizes = np.asarray([
            len(self.dp.local_train_label[k]) for k in range(self.dp.size_device)
        ], dtype=np.float64)
        global_p = np.average(private_p, axis=0, weights=sizes)
        c_num = min(self.num_fe, self.dp.size_device)
        base = self.dp.size_device // c_num
        extra = self.dp.size_device % c_num
        capacities = [base + int(c < extra) for c in range(c_num)]
        remaining = set(range(self.dp.size_device))
        groups = []

        for capacity in capacities:
            members = set()
            weighted_sum = np.zeros(private_p.shape[1], dtype=np.float64)
            total_size = 0.0
            for _ in range(capacity):
                best_client = None
                best_score = float("inf")
                for k in remaining:
                    candidate = (weighted_sum + sizes[k] * private_p[k]) / (total_size + sizes[k])
                    score = self._kl(candidate, global_p)
                    if score < best_score:
                        best_score = score
                        best_client = k
                members.add(best_client)
                remaining.remove(best_client)
                weighted_sum += sizes[best_client] * private_p[best_client]
                total_size += sizes[best_client]
            groups.append(members)

        self.conductor = groups
        self._save_groups("Conductor_FE_clients.txt")

    def assign_clients_ls(self, balance=True, max_iterations=10):
        if not balance:
            self.conductor = [{k} for k in range(self.dp.size_device)]
            self._save_groups("Conductor_FE_clients_large_scale.txt")
            return

        private_p = self._private_distributions()
        sizes = np.asarray([
            len(self.dp.local_train_label[k]) for k in range(self.dp.size_device)
        ], dtype=np.float64)
        global_p = np.average(private_p, axis=0, weights=sizes)
        c_num = min(self.num_fe, self.dp.size_device)

        # Appendix B.2.2 large-scale approximation: random C-way
        # initialization followed by a bounded number of local client swaps.
        order = np.random.permutation(self.dp.size_device)
        self.conductor = [set(x.tolist()) for x in np.array_split(order, c_num)]
        group_of = {}
        for c, group in enumerate(self.conductor):
            for k in group:
                group_of[k] = c

        weighted_sums = []
        total_sizes = []
        for group in self.conductor:
            ids = np.asarray(sorted(group), dtype=int)
            weighted_sums.append((private_p[ids] * sizes[ids, None]).sum(axis=0))
            total_sizes.append(float(sizes[ids].sum()))

        def kl_from(sum_vec, total):
            return self._kl(sum_vec / total, global_p)

        for _ in range(max_iterations):
            improved = False
            for i in np.random.permutation(self.dp.size_device):
                a = group_of[int(i)]
                for b in np.random.permutation(c_num):
                    b = int(b)
                    if b == a or not self.conductor[b]:
                        continue
                    # One local partner proposal per target group keeps each
                    # refinement pass linear in N*C (up to label dimension L).
                    candidates = tuple(self.conductor[b])
                    j = int(candidates[np.random.randint(len(candidates))])

                    before = kl_from(weighted_sums[a], total_sizes[a])
                    before += kl_from(weighted_sums[b], total_sizes[b])

                    sum_a = weighted_sums[a] - sizes[i] * private_p[i] + sizes[j] * private_p[j]
                    sum_b = weighted_sums[b] - sizes[j] * private_p[j] + sizes[i] * private_p[i]
                    total_a = total_sizes[a] - sizes[i] + sizes[j]
                    total_b = total_sizes[b] - sizes[j] + sizes[i]
                    after = kl_from(sum_a, total_a) + kl_from(sum_b, total_b)

                    if after < before:
                        self.conductor[a].remove(int(i))
                        self.conductor[a].add(j)
                        self.conductor[b].remove(j)
                        self.conductor[b].add(int(i))
                        group_of[int(i)], group_of[j] = b, a
                        weighted_sums[a], weighted_sums[b] = sum_a, sum_b
                        total_sizes[a], total_sizes[b] = total_a, total_b
                        improved = True
                        break
            if not improved:
                break

        self._save_groups("Conductor_FE_clients_large_scale.txt")

    def assign_clients_to(self, balance=True):
        if not balance:
            self.conductor = [{k} for k in range(self.dp.size_device)]
            self._save_groups("Conductor_RC_clients.txt")
            return

        private_p = self._private_distributions()
        n = self.dp.size_device
        distances = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = self._js(private_p[i], private_p[j])
                distances[j, i] = distances[i, j]

        condensed = squareform(distances)
        tree = linkage(condensed, method="average")
        labels = fcluster(tree, t=self.threshold, criterion="distance")
        unique = sorted(np.unique(labels))
        self.conductor = [set(np.where(labels == label)[0].tolist()) for label in unique]
        self._save_groups("Conductor_RC_clients.txt")

    def assign_clients_to_ls(self, balance=True, iterations=100):
        if not balance:
            self.conductor = [{k} for k in range(self.dp.size_device)]
            self._save_groups("Conductor_RC_clients_large_scale.txt")
            return

        private_p = self._private_distributions()
        m = min(self.num_rc, self.dp.size_device)
        labels = KMeans(
            n_clusters=m,
            n_init=5,
            max_iter=iterations,
            random_state=self.seed,
        ).fit_predict(private_p)
        self.conductor = [set(np.where(labels == c)[0].tolist()) for c in range(m)]
        self._save_groups("Conductor_RC_clients_large_scale.txt")
