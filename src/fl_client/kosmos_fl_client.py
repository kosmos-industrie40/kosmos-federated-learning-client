"""
This script is the implementation for the federated learning participators it connects to the \
KOSMoS server. If enough training data has been collected it connects to the server which will \
start a federated training as soon as enough participants have applied for training.
"""
import argparse
import time
import os

import socketio
from dynaconf import Dynaconf
from fl_client.flwr_client import start_client

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
CONFIG = Dynaconf(includes=[CONFIG_FILE])

if isinstance(CONFIG.DEBUG, bool):
    LOG_SOCKET_IO = CONFIG.DEBUG
elif CONFIG.DEBUG.lower() == "true":
    LOG_SOCKET_IO = True
else:
    LOG_SOCKET_IO = False
sio = socketio.Client(logger=LOG_SOCKET_IO, engineio_logger=LOG_SOCKET_IO)


@sio.event
def connect():
    """
    This default function is called if the connection to the curator has been established
    """
    print("connection established")


# default handler for all messages
@sio.event
def message(data):
    """
    This is the default message handler if no handler has been specified for a event.
    It should not be used in the production code but is useful for debugging.
    :param data:
    """
    print("message received with ", data)


@sio.event
def disconnect():
    """
    The default event handler called if the connection is closed.
    NOTE: Won't be called if the connection is interrupted unintended.
    """
    print("Client disconnected from server")


@sio.event
def start_train(data):
    """
    This message handler starts the flwr client with it's necessary federated learning
    information. Expects the following keys in the configuration file:
    * client_type
    * usecase
    * num_client_train_epochs
    * learning_rate

    :param data: Dictionary containing data emitted from the server. Uses the following keys:
        * client_id
        * flwr_server_address
        * usecase_name
        * usecase_params, optional defaults to {}
    """

    client_id = data["client_id"]
    flwr_server_address = data["flwr_server_address"]
    usecase_name = data["usecase_name"]
    broadcast_params = data.get("usecase_params", {})
    client_name = CONFIG["client_type"]
    config_usecase_params = CONFIG.as_dict()["USECASE"][usecase_name].get(
        int(client_id), {}
    )

    print(f"STARTING TRAINING WITH CLIENT ID {client_id}")

    start_client(
        client_id=client_id,
        client_name=client_name,
        n_client_epochs=CONFIG["num_client_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        flwr_server_address=flwr_server_address,
        usecase_name=usecase_name,
        usecase_params={**config_usecase_params, **broadcast_params},
    )


if __name__ == "__main__":

    # Starts the client which registers for training at the server. The server emits the selected
    # usecase and runs the federated learning process using flower if enough client are available
    # for training.
    # The configuration for the training process is provided by the server. The clients can be
    # forced to  check their data for certain conditions further to attend in the training process.
    # The client will connect to localhost:6000 if there is no other address and port specified in
    # the python arguments or the environment variables. The environment variables will be used
    # before the parsed argument which will be used before default settings. The environment
    # variables must be named according to the argument parser destination.

    PATH = None
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "--delay",
        dest="startup_delay",
        help="Delays the startup of the kosmos client to be executed after the server",
        required=False,
        default="0",
        type=int,
    )

    args = PARSER.parse_args()

    if args.startup_delay > 0:
        time.sleep(args.startup_delay)

    while True:

        try:
            print(f"Trying to connect to: {CONFIG.socketio_address}")
            sio.connect(CONFIG.socketio_address)

            # Event used to check necessary criteria before connecting to the server.
            # Can be expanded, i.e. to check whether enough data has been collected
            sio.emit(event="client_criteria", data={"criteria_are_met": True})
            sio.wait()
            break

        except socketio.exceptions.ConnectionError as error:
            print(
                "The connection to the KOSMoS federated learning server could not be established."
                " The establishment of the connection will be reattempted in 3 seconds."
            )
            print(error)
            time.sleep(3)
