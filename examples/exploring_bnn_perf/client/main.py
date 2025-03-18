"""
This script depends on:
* pytorch
* numpy (1.2.x)
* sklearn
* matplotlib
* rtree (for analysis)
* scikit-learn (for high-dimensional regression example)
"""

import numpy as np
import torch
import argparse
import os

import torch.optim as optim

from torch import Tensor
from torch.nn import Module

# Stores the BNN architecture and training logic
from network import *
from data import *
from sembas_api import *

THRESHOLD = 0.5


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "BNN Exploration",
        "python fut [mode]",
        description="If no mode is provided, mode defaults to --explore.",
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--full",
        "-f",
        action="store_true",
        default=False,
        help="Generates a new model, trains, saves, and explores the solution space.",
    )

    mode.add_argument(
        "--train",
        "-t",
        action="store_true",
        default=False,
        help="Only generates a new model, trains, and saves it.",
    )

    mode.add_argument(
        "--experiment",
        default="simple",
        help=(
            "Selects which experiment to run:\n"
            "simple - A simple 2D regression problem for visualization.\n"
            "complex - An 8 dimensional regression problem.\n"
        ),
    )

    mode.add_argument(
        "--explore",
        "-x",
        action="store_true",
        default=False,
        help=(
            "Explores an existing model under --model-path. If no model exists "
            "under this path, the model will be generated prior to exploration."
        ),
    ),

    parser.add_argument(
        "--graphics",
        "-g",
        action="store_true",
        default=False,
        help=(
            "Displays loss history for training and the samples taken during "
            "exploration."
        ),
    ),

    parser.add_argument(
        "--model-path",
        "-p",
        type=str,
        default=".models/bnn_expl/",
        help="Where to store and load the BNN model.",
    )

    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="bnn.model",
        help="The name of the bnn model file.",
    )

    parser.add_argument(
        "--num-networks",
        "-n",
        type=int,
        default=1000,
        help=(
            "The number of networks to explore. "
            "Doesn't do anything for mode --train.",
        ),
    )
    parser.add_argument(
        "--threshold",
        "-l",
        type=float,
        default=0.5,
        help=(
            "The maximum error for considering an output to be 'valid'. Used to "
            "classified inputs for finding the region of validity "
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    if get_args().graphics:
        print("Enabled graphics")
        import graphics as graphics
        import matplotlib.pyplot as plt
    else:
        graphics = None
        import graphics as graphics2
        import matplotlib.pyplot as plt


def train_and_save(dataset: FutData, path: str, model_name: str, width=50, lr=0.01):
    "Trains a BNN using @dataset, and saves it under '@path/@model_name'"
    bnn = BayesianNN(dataset.inputs.shape[1], dataset.targets.shape[1], width)

    optimizer = optim.Adam(bnn.parameters(), lr=lr)

    test_history, train_history = train_bnn(bnn, optimizer, dataset, epochs=2)
    os.makedirs(path, exist_ok=True)
    torch.save(bnn.state_dict(), f"{path}/{model_name}")
    return test_history, train_history


def load_bnn(path: str):
    "If a BNN exists under @path, will load and return the BNN model."
    state = torch.load(path, weights_only=True)
    model = BayesianNN()
    model.load_state_dict(state)
    return model


def classify_validity(network: Module, dataset: FutData, x: Tensor):
    """
    Given a network, classifiers a sample as valid/invalid.

    In real-world circumstances, you would often need a measure of validity that is
    unsupervised, meaning you can't measure the exact "error" from ground truth.

    ***
    In the case of this toy example, we do have ground truth, so we are using
    it directly.
    ***

    An example of a solution for classifying an `AV testing scenario` would be
    collision occurance, time-to-collision thresholds, traffic violations, etc.

    Unsupervised solutions for an `image classifier` might require an adversarial
    neural network designed to catch fraudulant examples, such as those used in GANs.

    `Regression` is more difficult/complicated, but you might find measures that
    characterize confidence as a measure of validity, or K-nearest-neighbor based on
    a test set.
    """

    model_x = dataset.transform_request(x)
    true_x = dataset.input_scaler.inverse_transform(model_x)
    return (
        np.power(
            network(model_x).squeeze().detach().numpy()
            - dataset.target_scaler.transform(simple_2d_regr(*true_x.T).reshape(1, -1)),
            2.0,
        )
        < THRESHOLD
    )


def load_and_explore(args: argparse.Namespace, dataset: FutData, sample_classifier):
    """
    This program will load the BNN, sample a network from the BNN, and explore the
    network's input space. The function will iteratively sample the BNN until
    args.num_networks number of networks has been explored.

    Before sampling the next network, the remote classifier connection from the
    main.rs program will be dropped and restarted to indicate that the client must
    sample the next network. This is a slow process, and is a matter of future work
    (e.g. adding a 'next' signal or similar).
    """

    bnn = load_bnn(f"{args.model_path}/{args.model_name}")
    NDIM = 2

    os.makedirs(f"{args.model_path}/ensemble", exist_ok=True)

    for i in range(args.num_networks):
        network = bnn.sample_network()
        client = setup_socket(NDIM)

        samples = []

        session_ended = False
        while not session_ended:
            try:
                p = receive_request(client, NDIM)
                cls = sample_classifier(network, dataset, p.reshape(1, -1)).squeeze()
                samples.append((p, cls))
                send_response(client, cls)
            except struct.error as e:
                client.close()
                session_ended = True

        if graphics is not None:
            print(f"Took {len(samples)} samples.")
            print("Displaying sample graph. Close to continue...")
            fig, (axl, axr) = plt.subplots(ncols=2)
            fig.tight_layout()
            graphics.sample_graph(samples, ax=axl)
            graphics.brute_force_search(network, dataset, classify_validity, ax=axr)
            plt.show()

        torch.save(
            network.state_dict(), f"{args.model_path}/ensemble/network_{i}.model"
        )


def get_mode(args: argparse.Namespace) -> str:
    "Converts the mode arg flags to string"
    if args.full:
        return "full"
    elif args.train:
        return "train"
    elif args.explore:
        return "explore"
    return "explore"


def main(dataset_size: int = 2**10):
    global THRESHOLD
    import os

    args = get_args()

    if args.experiment == "simple":
        dataset = Simple2DRegrData(dataset_size)
    else:
        dataset = HighDimensionalRegrData()

    THRESHOLD = args.threshold

    pre_trained_exists = os.path.isfile(f"{args.model_path}/{args.model_name}")
    mode = get_mode(args)

    do_train = mode in (
        "train",
        "full",
    )
    do_explore = mode in ("explore", "full")

    if pre_trained_exists and mode == "explore":
        print(f"Pre-trained model exists.")
    elif mode == "explore":
        print(f"Pre-trained model does not exist.")
        do_train = True

    if do_train:
        print("Generating and training model...")
        test_history, train_history = train_and_save(
            dataset,
            args.model_path,
            args.model_name,
            lr=0.01 if args.experiment == "simple" else 0.05,
        )

        if graphics is not None:
            print("Displaying loss graph. Close to continue...")
            graphics.loss_graph(test_history, train_history)
            plt.show()

        print("Model training complete.")

    if do_explore:
        print("Beginning exploration.")
        load_and_explore(args, dataset, classify_validity)
        print("Exploration complete.")


def test():
    dataset = HighDimensionalRegrData()
    bnn = BayesianNN(dataset.inputs.shape[1], dataset.targets.shape[1], 126)

    optimizer = optim.Adam(bnn.parameters(), lr=0.05)

    test_history, train_history = train_bnn(
        bnn, optimizer, dataset, kl_weight=1e-5, epochs=10
    )

    plt.plot(np.arange(len(test_history)), test_history)
    plt.show()

    net = bnn.sample_network()

    yhat = net(dataset.inputs).detach()
    y = dataset.targets

    loss: Tensor = (yhat - y) ** 2

    print("Removed", len(loss[loss > 10000]), "entries")
    print("Below threshold", len(loss[loss < 50]) / len(loss), "entries")
    print("Median loss", loss.median())
    print("Below threshold", len(loss[loss <= 1]) / len(loss), r"% of entries")
    loss = loss[loss <= 10000]

    print(loss.shape)
    # print(dataset.inputs.shape, yhat.shape, y.shape)
    print(loss.min(), loss.max(), loss.mean())

    plt.plot(np.arange(len(loss)), loss)
    plt.show()


if __name__ == "__main__":
    test()
