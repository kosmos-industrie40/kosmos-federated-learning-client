"""
Dynamically loads Client using DynamicLoader
"""
import os
from fl_models.util.dynamic_loader import DynamicLoader

import flwr as fl

_CLIENT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clients")
_CLIENT_LOADER = None


def load_client(client_name: str, *args, usecase_name: str = None, **kwargs):
    """
    Returns a DynamicLoader that loads the Client classes.
    """
    # pylint: disable=global-statement
    # Only load files when needed
    global _CLIENT_LOADER

    if _CLIENT_LOADER is None:
        # Init
        _CLIENT_LOADER = DynamicLoader(_CLIENT_FOLDER, fl.client.NumPyClient)

    return _CLIENT_LOADER.get(client_name)(usecase_name=usecase_name, *args, **kwargs)
