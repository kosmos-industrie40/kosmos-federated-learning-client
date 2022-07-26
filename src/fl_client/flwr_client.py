"""
Flower client implementation providing all necessary functionalities to participate
in the federated learning process
"""
import time
import argparse

import flwr as fl
from fl_client.util.client_loader import load_client


# pylint: disable= too-many-arguments
def start_client(
    client_id: int,
    client_name: str,
    n_client_epochs: int = 5,
    learning_rate: float = None,
    flwr_server_address: str = "localhost:8080",
    usecase_name=None,
    usecase_params: dict = None,
):
    """
    Starts a client with specified data to participate in federated training
    :param client_id: A unique client id
    :param client_name: Name of the client that should be loaded.
    :param n_client_epochs: The number of epochs the client training in a federated learning round
    :param learning_rate: The learning rate applied in optimizing the model
    :param flwr_server_address: The address under which the server is hosted
    :param usecase_name: Name of the usecase that was selected by the server
    :param usecase_params: Parameters used to instantiate the usecase
    """

    if usecase_params is None:
        usecase_params = {}

    print(f"CREATING NUMPY_CLIENT {client_id}")

    numpy_client = load_client(
        client_name,
        usecase_name=usecase_name,
        client_id=client_id,
        n_epochs=n_client_epochs,
        learning_rate=learning_rate,
        **usecase_params,
    )

    time.sleep(5)
    fl.client.start_numpy_client(flwr_server_address, client=numpy_client)


if __name__ == "__main__":

    # Starting a client with default parameters and preset data partitioning

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clientID",
        dest="client_id",
        help="Client ID used to reference client and load client data",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--address",
        dest="server_address",
        help="Server address and Port of the flower server instance to connect to.",
        required=False,
        default="localhost:8080",
        type=str,
    )

    parser.add_argument(
        "--delay",
        dest="startup_delay",
        help="Delay defines a time in seconds which the client will wait before start running",
        required=False,
        default=0,
        type=int,
    )

    args = parser.parse_args()

    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    default_client_bearing_dist = dict(
        {
            0: ["Bearing1_1", "Bearing1_2"],  # data for client_id = 0
            1: ["Bearing2_1", "Bearing2_2"],  # data for client_id = 1
            2: ["Bearing3_1", "Bearing3_2"],  # data for client_id = 2
        }
    )

    start_client(args.client_id, default_client_bearing_dist.get(args.client_id))
