=================================
KOSMoS Federated Learning Client
=================================


DESCRIPTION
===========
This repository implements a federated learning client. The client is part of the KOSMoS Federated Learning Framework 
which consists of two additional components: the `KOSMoS Federated Learning Server <https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_ and the `KOSMoS Federated Learning Resources <https://github.com/kosmos-industrie40/kosmos-federated-learning-resources>`_ project.
This project is able to run with any arbitrary data set but by default is executed with the bearing data set.
For further information on design principals take a look at the `blogpost <https://www.inovex.de/de/blog/federated-learning-implementation-into-kosmos-part-3/>`_ describing the whole project.


USE CASE
========
The general goal is to collect machine data from machine operators at the KOSMoS Edge and then collaboratively train a model to predict the remaining useful lifetime. This Federated Bearing use case implements this approach with the following restrictions:

- The data used for training isn't collected by the machine operator but the bearing data set is manually distributed to the collaborating clients
- The current project can be deployed with the docker container provided in this project but isn't deployed in the current KOSMoS project
- The connection between clients and host is not encrypted as of now. This can be enabled in the Wrapper (websocket) and flower (grpc) implementation quite easily.

Open Points are:

- The optional but useful security features Differential Privacy and Secure Multiparty Computation are not implemented yet.

BUILD & RUN
===========

Server
******

Before starting the clients, the `KOSMoS Federated Learning Server <https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_ must be started.

Docker
******

All containers run within the same network. In case the network has not been created yet run:

.. code-block::

    docker network create fl_network


To start the KOSMoS Federated Learning Clients for a federated learning session, docker
containers can be used. Adjust the :code:`config.yaml` file and run several clients using the
following commands:

.. code-block::

    docker build --rm -t kosmos_fl_client:latest -f Dockerfiles/kosmos_fl_client.Dockerfile .

Run the clients either with the docker-compose given in the `server repository
<https://github.com/kosmos-industrie40/kosmos-federated-learning-server>`_ or run the clients with the following commands:

During and after training you can connect to the mlflow ui by visiting `http://localhost:5000
<http://localhost:5000>`_.

.. code-block::

    # Exchange <index> with the current client index
    docker run -d --network fl_network --name kosmos_fl_client_<index> kosmos_fl_client:latest

During and after training you can connect to the mlflow ui by visiting `http://localhost:5000
<http://localhost:5000>`_.

Without Docker
**************

Furthermore, to execute the client locally the following steps must be taken. This project was
developed using python version 3.8. The behavior with other versions is undefined. There are known issues with tensorflow 2.5 and 2.6.

1. Install all necessary python packages (install in a virtual environment if necessary):

.. code-block::

    pip install -r requirements.txt

2. Install the client

.. code-block::

    python setup.py install

3. Run the KOSMoS Federated Learning Client using the following command:

.. code-block::

    cd src/fl_client/
    python kosmos_fl_client.py

Troubleshooting
****************

- Client cannot connect to server: 
   - Assure that the ``socketio_address`` and the ``flwr_server_address`` match the server settings. 
   - Make sure that :code:`docker --network` is set to the same alias on both server and client.
   - Use docker aliases to access socketio and flower server:

   .. code-block::

       # Exchange <index> with the current client index
       docker run -d --network fl_network --name kosmos_fl_client_<index> -e DYNACONF_flwr_server_address=kosmos_fl_server:50052 -e DYNACONF_socketio_address=http://kosmos_fl_server:6000 --entrypoint /app/venv/bin/python kosmos_fl_client:latest kosmos_fl_client.py --delay=5

- No Progress bar is shown when loading the bearing data: Add :code:`-tty` argument to :code:`docker run`

CONFIG.YAML FILE
================

The upbringing of a federated learning client with flower is based on the :code:`config.yaml` file featuring the following parameters:

.. list-table:: Configuration Details
   :widths: 25 25 25 50
   :header-rows: 1

   * - Name
     - Values
     - Default
     - Description
   * - num_clients
     - 1 or higher
     - 3
     - Number of participating clients must match the number of train bearing sets
   * - flwr_server_address
     - ip_address:port
     - [::1]:50052
     - Address and Port of the central flower server. Uses IPv6.
   * - socketio_address
     - ip_address:port
     - 0.0.0.0:6000
     - Address and Port of the central socketio server.
   * - mlflow_server_address
     - url:port or a local directory
     - http://localhost:5000
     - Address to the MLFlow server
   * - train_bearings_per_clients:
     - Key:[List] of Bearings Used for training
     - 0 : ["Bearing1_1", "Bearing1_2"]
       1: ["Bearing2_1", "Bearing2_2"]
       2: ["Bearing3_1", "Bearing3_2"]
     - These bearings will be associated with the client ID
   * - learning_rate
     - float
     - null
     - The learning rate used for the local training
   * - DEBUG
     - True or False
     - False
     - Shows additional debugging messages

Note that the bearings available for training and testing are chosen distinctively from the list of all available bearing.
Because of the nature of federated learning, a bearing should only be used either in test or as a
client train data.

Developer Guide
===============

A guide on how to create new use cases and add new models can be found `here <https://github.com/kosmos-industrie40/kosmos-federated-learning-resources/blob/release/HOWTO.rst>`_.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
