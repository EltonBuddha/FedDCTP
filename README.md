# FedDCTP reference implementation

This repository is a **compact algorithmic reference framework**, not a turnkey reproduction package. It illustrates the network partition, conductor-based grouping, privacy mechanism, and two-stage training in the accompanying paper. Dataset downloads, complete per-dataset experiment recipes, environment lockfiles, evaluation/plotting pipelines, and five-seed orchestration are not bundled.

Algorithm 1 treats `T1`, `T2`, the number of FE conductors `C`, and the RC grouping criterion as experiment-specific inputs. Accordingly, this reference requires `--stage1_rounds`, `--stage2_rounds`, `--num_FE`, and `--threshold` explicitly instead of embedding a one-size-fits-all numerical choice. Other CLI defaults are conveniences for the default CIFAR/ResNet reference path and should be adjusted to the experiment being studied.

## Algorithm correspondence

- **Stage 1 / FE:** Clients are partitioned using KL divergence between a conductor's joint label distribution and the global label distribution. The small/medium-client routine uses a balanced greedy heuristic; the large-scale routine follows the paper's random `C`-way initialization plus bounded local client swaps that reduce the KL objective. Each round samples exactly `floor(rho1 * C)` FE **conductors**, then processes all clients in each selected conductor **sequentially** in a freshly shuffled order. The conductor applies sample-size-weighted incremental updates, and the server aggregates selected conductor models using their handled sample counts.
- **Stage 2 / RC:** Clients are grouped by Jensen-Shannon (JS) divergence. The standard routine computes pairwise JS divergence, applies average-linkage hierarchical clustering, and cuts the tree at `tau`; the large-scale approximation applies KMeans directly to the privatized label-distribution vectors. Each round samples exactly `floor(rho2 * N)` individual clients globally. Active clients train the current group-shared **refiner** and their own **P-Head**. The conductor aggregates only refiner updates using participating-client data sizes; P-Heads remain private. The final group refiner is broadcast to every client in that group.
- **Initialization and freezing:** Stage 2 inherits and freezes the Stage-1 feature extractor and G-Head. Each RC conductor receives a separately reinitialized group refiner, and each client receives a separately reinitialized private P-Head. Frozen parameter-owning modules and their BatchNorm running statistics remain unchanged during Stage 2. Parameter-free Dropout that lies after a trainable refiner remains in training mode.
- **Learning rate:** Within a communication round, every participating client uses the same `lr * lr_decay ** round_index`. This reference uses a single communication-round counter across the two stages.
- **Non-IID image partition:** For MNIST/FMNIST/CIFAR, a per-class Dirichlet draw is used to allocate both training and test samples across clients, so each client's local P-Test follows the same label-skew mechanism as its training set. The CIFAR loader retains the reference 70/30 split used by the original experiment code; the hidden compatibility switch `--divide_data_73` selects the official CIFAR split instead.
- **Image preprocessing:** Grayscale MNIST/FMNIST and RGB CIFAR arrays carrying 0..255 pixel values are converted to `uint8` before `ToPILImage()`/`ToTensor()`, avoiding accidental intensity rescaling.

## Network partition

For the bundled image backbones, the code follows the paper's block placement:

- MNIST/FMNIST ConvNet: convolutional stack = feature extractor, first fully connected layer = refiner, final classifier = head.
- CIFAR ConvNet: three convolutional layers = feature extractor, first fully connected layer = refiner, classifier = head.
- ResNet20: the last three residual blocks (`layer3`) are the refiner; earlier residual blocks form the feature extractor; the final linear layer is the head.

Stage 1 trains the complete model. Stage 2 creates a two-head form: the frozen Stage-1 classifier becomes the G-Head, while a fresh P-Head is optimized locally.

## How many RC conductors are used?

The paper's Algorithm 1 denotes the number of RC conductors by `M`. In the **standard/small-scale** routine `ConductorData.assign_clients_to()`, the realized `M` is determined by the JS threshold `tau` and the client distributions: the hierarchical tree is cut at `--threshold`, and the effective number of groups is `len(rc.conductor)`. This is the path used to represent the paper's JS-threshold analysis, where changing `tau` changes the number of RC groups.

The **large-scale approximation** `assign_clients_to_ls()` uses KMeans on the privatized histograms, and `--num_RC` sets its target number of clusters. Thus `--num_RC` is intentionally not used by the standard hierarchical routine.

## Label-histogram privacy (Appendix C)

The implementation uses the manuscript's record-level **replacement adjacency** directly. For client `k`, the local sample count `n_k` is fixed and publicly known, and neighboring datasets of equal size differ only in the label of one record. Therefore the normalized label histogram has

`Delta_k = 2 / n_k`, giving Laplace scale `b_k = 2 / (n_k * epsilon)`.

There is no alternate sensitivity mode in the paper-aligned path. Each client produces **one** privatized normalized histogram. Negative perturbed entries are clipped to zero and the vector is renormalized; these operations are deterministic post-processing. The same cached privatized histogram is reused by both FE (KL) and RC (JS) grouping, so the second grouping does not trigger a second release. Group-level and global label distributions are formed from these privatized client histograms using publicly known client sample-size weights.

`--epsilon inf` selects the noise-free setting and does not provide a differential-privacy guarantee. The sensitivity depends on `n_k`, not on the number of classes.

## Scope

The lightweight `Main.py` wires the paper's MNIST/FMNIST and CIFAR image families. OfficeCaltech10, DomainNet, and AGNews require their dataset-specific domain/text preprocessing and experiment recipes; those pipelines are deliberately not represented as runnable branches in this compact package. The corresponding model components can remain useful as architectural references, but this repository should not be described as a six-dataset turnkey reproduction suite.

A configured invocation has the form

```bash
python Main.py \
  --stage1_rounds <T1> \
  --stage2_rounds <T2> \
  --num_FE <C> \
  --threshold <tau> \
  --epsilon 5
```

The single `--seed` value seeds Python, NumPy, PyTorch, CUDA (when available), the Dirichlet partition, Laplace perturbation, client/conductor sampling, and the large-scale KMeans initializer. Exact bitwise GPU determinism is not claimed.
