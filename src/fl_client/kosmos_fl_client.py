"""
This script is the implementation for the federated learning participators it connects to the
kosmos. If enough training data has been collected it connects to the server which will start a
federated training as soon as enough participants have applied for training.
"""
import argparse
import time

import socketio
from dynaconf import Dynaconf
from fl_client.flwr_client import start_client

CONFIG = Dynaconf(includes=["./config.yaml"])

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
    It won't be used in the production code but is useful for debugging.
    :param data:
    """
    print("message received with ", data)


@sio.event
def disconnect():
    """
    The default event handler called if the connection is closed.
    NOTE: It won't be called if the connection is interrupted unintended.
    """
    print("Client disconnected from server")


@sio.event
def start_train(data):
    """
    This message handler starts the flwr client with it's necessary federated learning
    information.
    :param data: A dict with the client ID
    """

    client_id = data.get("client_id")
    # mlflow_run_id = data.get("mlflow_run_id")
    # mlflow_server_address = data.get("mlflow_server_address")

    print(f"STARTING TRAINING WITH CLIENT ID {client_id}")

    start_client(
        client_id=client_id,
        train_bearings=CONFIG.get("train_bearings_per_clients").get(int(client_id)),
        n_client_epochs=CONFIG.get("num_client_train_epochs"),
        learning_rate=CONFIG.get("learning_rate"),
        flwr_server_address=CONFIG.get("flwr_server_address"),
        # mlflow_server_address=mlflow_server_address,
        # mlflow_run=mlflow_run_id,
    )


if __name__ == "__main__":

    # Starts the bearing use case client which registers for training at the server. The server runs
    # the federated learning process using flower if enough client are available for training.
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
            print(f"flwr server: {CONFIG.flwr_server_address}")
            sio.connect(CONFIG.socketio_address)
            sio.emit(event="client_criteria", data={"criteria_is_met": True})
            sio.wait()
            break

        except socketio.exceptions.ConnectionError as error:
            print(
                "The connection to the komos federated learning server couldn't be established."
                " A reconnect will be attempted in 3 seconds."
            )
            print(error)
            time.sleep(3)
