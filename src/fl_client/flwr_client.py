"""
Flower client implementation providing all necessary functionalities to participate
in the federated learning process
"""
import time
from typing import List, Dict
import argparse
import copy

import pandas as pd
import flwr as fl
import numpy as np

from fl_models.DegradationModel import DegradationModel
from fl_models.CNNSpectraFeatures import CNNSpectraFeatures
from fl_models.util.dataloader import read_feature_dfs_as_dict, df_dict_to_df_dataframe
from fl_models.util.metrics import rmse, correlation_coefficient

from fl_client.util.helper_methods import pop_labels


class NumpyClient(fl.client.NumPyClient):
    """
    Flower numpy client implementation redefining the fit and evaluate function which will be
    triggert by the server.
    """

    # pylint: disable= too-many-instance-attributes
    def __init__(
        self,
        client_id: int,
        n_epochs: int,
        client_bearings,
        learning_rate: float = None,
    ):

        self.client_id: int = client_id
        self.training_data_dict: Dict[str, pd.DataFrame] = None
        self.n_epochs: int = n_epochs
        self.client_bearings = client_bearings
        self.degradation_model: DegradationModel = CNNSpectraFeatures(
            name="CNN_" + str(client_id)
        )
        self.metric_functions = [rmse, correlation_coefficient]

        self.current_train_rnd = 0
        self.current_eval_rnd = 0
        self.load_data(client_bearings)
        self.learning_rate = learning_rate

    def get_parameters(self) -> List[np.ndarray]:
        """
        Processes the local model parameter
        :return: Model paramters as numpy array
        """
        return self.degradation_model.get_parameters()

    def fit(self, parameters, config):
        """
        This function trains a model with the local training data
        :param parameters: Model parameter to be trained with
        :param config: An optional Dict containing training configurations
        :return: Tupel containing the new model parameter,
        number of training data and Dict with optional Values
        (model_paramter, number of training data and Dict[trainings_loss]
        """
        self.degradation_model.set_parameters(new_weights=parameters)
        copied_training_data = copy.deepcopy(self.training_data_dict)
        tmp_training_data, tmp_training_labels = df_dict_to_df_dataframe(
            copied_training_data
        )
        history = self.degradation_model.train(
            training_data=tmp_training_data,
            training_labels=tmp_training_labels,
            epochs=self.n_epochs,
            validation_data=None,
            validation_labels=None,
            learning_rate=self.learning_rate,
        )

        self.current_train_rnd += 1
        return (
            self.degradation_model.get_parameters(),
            len(self.training_data_dict),
            {"loss": history.history.get("loss")[-1]},
        )

    def evaluate(self, parameters, config):
        """
        This function evaluates the provided model parameters on the local training data
        :param parameters: Model parameters to evaluate on
        :param config: An optional Dict containing evaluation configurations
        :return: Tupel containing three values representing training success,
        model aggregation weight and model metrics
        (RMSE, number of trainingdata and Dict[RMSE, PCC]
        """

        self.degradation_model.set_parameters(parameters)
        copied_training_data = copy.deepcopy(self.training_data_dict)
        copied_training_labels = pop_labels(copied_training_data)
        results = self.degradation_model.compute_metrics(
            df_dict=copied_training_data,
            labels=copied_training_labels,
            metrics_list=self.metric_functions,
        )

        denominator = len(results)
        metrics_names = ["RMSE", "PCC"]
        mean = dict.fromkeys(metrics_names, 0)
        for metric in metrics_names:
            metric_sum = 0

            for _, bearing_results in results.items():
                metric_sum += bearing_results[metric]

            mean[metric] = metric_sum / denominator

        # mlflow.log_metric(
        #     "RMSE_" + str(self.client_id),
        #     mean.get(metrics_names[0]),
        #     self.current_eval_rnd,
        # )
        # mlflow.log_metric(
        #     "PCC_" + str(self.client_id),
        #     mean.get(metrics_names[1]),
        #     self.current_eval_rnd,
        # )

        self.current_eval_rnd += 1

        return (
            mean[metrics_names[0]],  # RMSE
            len(self.training_data_dict),
            {"RMSE": mean[metrics_names[0]], "PCC": mean[metrics_names[1]]},
        )

    def load_data(self, bearing_names_list: List[str]):
        """
        This function load a subset of the bearing data for the purpose of training
        :param bearing_names_list: A list containing bearing names
        """
        self.training_data_dict: Dict[str, pd.DataFrame] = read_feature_dfs_as_dict(
            bearing_names=bearing_names_list
        )


# pylint: disable= too-many-arguments
def start_client(
    client_id: int,
    train_bearings: List[str],
    n_client_epochs: int = 5,
    learning_rate: float = None,
    flwr_server_address: str = "localhost:8080",
    # mlflow_server_address: str = "./mlruns",
    # mlflow_run=None,
):
    """
    Starts a client with specified data to participate in federated training
    :param client_id: A unique client id
    :param train_bearings: A list of bearing names to use for training
    :param n_client_epochs: The number of epochs the client traing in a federated learning round
    :param learning_rate: The learning rate applied in optimizing the model
    :param flwr_server_address: The address under which the server is hosted
    :param mlflow_server_address: The mlflow tracking URI either a folder path or a network address
    :param mlflow_run: mlflow run ID the client will save the logging to.
            All clients and the server should log to the same run
    """

    # mlflow.set_tracking_uri(mlflow_server_address)
    # if mlflow_run is not None:
    #     mlflow.start_run(run_id=mlflow_run, nested=True)
    # mlflow.log_param("train_bearings_" + str(client_id), train_bearings)

    # Start client
    numpy_client = NumpyClient(
        client_id=client_id,
        n_epochs=n_client_epochs,
        client_bearings=train_bearings,
        learning_rate=learning_rate,
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
