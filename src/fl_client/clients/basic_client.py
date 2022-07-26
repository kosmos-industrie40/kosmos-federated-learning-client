"""
Contains a default client that should work with most basic use cases
"""

from typing import List
from fl_models.abstract.abstract_usecase import FederatedLearningUsecase

import flwr as fl
import numpy as np

from fl_models.util.metrics import rmse, correlation_coefficient
from fl_models.util.dynamic_loader import load_usecase


class BasicClient(fl.client.NumPyClient):
    """
    Flower numpy client implementation using the evaluation function defined by the server's \
    usecase.
    """

    # pylint: disable= too-many-instance-attributes
    def __init__(
        self,
        client_id: int,
        n_epochs: int,
        usecase_name: str = None,
        learning_rate: float = None,
        **kwargs
    ):
        """
        Initializes a BasicClient.

        :param client_id: ID of the current client for logging purposes
        :param n_epochs: Number of training epochs.
        :param usecase_name: Name of the usecase selected by the server.
        :param learning_rate: (Optional) Learning rate used for training. If not given uses the \
            model's default
        :param kwargs: Arguments needed to initialize the usecase
        """
        assert usecase_name is not None, "Client is missing server usecase name!"

        self.client_id: int = client_id
        self.n_epochs: int = n_epochs

        self.usecase: FederatedLearningUsecase = load_usecase(
            usecase_name,
            model_name=usecase_name + "_client_" + str(client_id),
            log_mlflow=False,
            learning_rate=learning_rate,
            **kwargs
        )

        self.metric_functions = [rmse, correlation_coefficient]

        self.current_train_rnd = 0

    def get_parameters(self) -> List[np.ndarray]:
        """
        Processes the local model parameter
        :return: Model parameters as numpy array
        """
        return self.usecase.get_model().get_weights()

    def fit(self, parameters, config):
        """
        This function trains a model with the local training data
        :param parameters: Model parameter to be trained with
        :param config: An optional Dict containing training configurations
        :return: Tuple containing the new model parameter, \
            number of training data and Dict with optional Values \
            (model_parameter, number of training data and Dict[trainings_loss]
        """

        self.usecase.get_model().set_weights(new_weights=parameters)

        history = None

        history = self.usecase.get_model().train(
            training_data=self.usecase.get_data(flat=True),
            training_labels=self.usecase.get_labels(flat=True),
            epochs=self.n_epochs,
            validation_data=None,
        )

        assert (
            history is not None
        ), "No training samples were generated! Could not fit model!"

        self.current_train_rnd += 1
        return (
            self.usecase.get_model().get_weights(),
            self.usecase.get_number_of_samples(),
            {"loss": history.history.get("loss")[-1]},
        )

    def evaluate(self, parameters, config):
        """
        This function evaluates the usecase model parameters on the local training data
        :param parameters: Model parameters to evaluate on
        :param config: An optional Dict containing evaluation configurations
        :return: Tuple containing three values representing training success, \
            model aggregation weight and model metrics \
            (RMSE, number of training data and Dict["<metric_name>": <metric_value>]
        """
        eval_result = self.usecase.eval_fn(parameters)

        return eval_result[0], self.usecase.get_number_of_samples(), eval_result[1]


def load_class():
    """Getter for Dynamic Loading of this class

    Returns:
        Subclass of fl.client.NumPyClient
    """
    return BasicClient
