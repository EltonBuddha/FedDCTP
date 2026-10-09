# FedDCTP reference implementation

This repository is a **compact algorithmic reference framework**. It illustrates the network partition, conductor-based grouping, privacy mechanism, and two-stage training in the accompanying paper.

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
