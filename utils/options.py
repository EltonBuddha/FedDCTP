import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="Compact FedDCTP algorithmic reference implementation"
    )
    # Algorithm 1 treats T1, T2, C and the RC grouping threshold as
    # experiment-specific inputs.  Keep them explicit instead of baking one
    # paper configuration into a generic reference entry point.
    parser.add_argument("--stage1_rounds", type=int, required=True,
                        help="Number of Stage-1 communication rounds (T1)")
    parser.add_argument("--stage2_rounds", type=int, required=True,
                        help="Number of Stage-2 communication rounds (T2)")
    parser.add_argument("--num_FE", type=int, required=True,
                        help="Number of FE conductors C")
    parser.add_argument("--threshold", type=float, required=True,
                        help="JS-distance cut threshold tau for standard RC hierarchical clustering")

    parser.add_argument("--num_users", type=int, default=20)
    parser.add_argument("--rho1", type=float, default=1.0)
    parser.add_argument("--rho2", type=float, default=1.0)
    parser.add_argument("--local_ep", type=int, default=3)
    parser.add_argument("--local_bs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.99,
                        help="Multiplicative LR decay per communication round, across both stages")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--model", type=str, default="res", choices=["res", "cnn"])
    parser.add_argument("--dataset_name", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "FMNIST", "mnist"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes; if omitted, inferred from the loaded dataset")
    parser.add_argument("--num_channels", type=int, default=None,
                        help="Input channels; if omitted, inferred for MNIST/FMNIST/CIFAR")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)

    # Used only by the large-scale RC approximation (KMeans).  The standard
    # hierarchical RC routine obtains the realized M by cutting at tau.
    parser.add_argument("--num_RC", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=5.0,
                        help="Privacy budget for one cached label-histogram release per client; inf disables noise")

    # The original experimental code used the 70/30 CIFAR split adopted by
    # the referenced pFL setup.  Passing this hidden compatibility switch
    # selects the official CIFAR train/test files without that redistribution.
    parser.add_argument("--divide_data_73", action="store_false", help=argparse.SUPPRESS)
    return parser.parse_args()
