=================================
kosmos federated learning server
=================================


DESCRIPTION
===========
This is the server implementation of the kosmos federated learning framework. The whole project consists of two additional components `kosmos federated learning client <https://github.com/kosmos-industrie40/kosmos-federated-learning-client>`_ and `kosmos federated learning resource <https://github.com/kosmos-industrie40/kosmos-federated-learning-resources>`_ project. This project is able to run with any arbitrary data set but by default is executed with the bearing data set. For further information on design principals take a look at the `blogpost <https://www.inovex.de/de/blog/federated-learning-part-3/>`_ describing the whole project.


USE CASE
========
The general goal is to collect machine data from machine operators at the KOSMoS Edge and then collaboratively train a model for remaining useful lifetime prediction. This Federated Bearing use case implements this approach with the following restrictions:

- The data used for training isn't collected by the machine operator but the bearing data set manually distributed to the collaborating clients
- The current project can be deployed with the docker container provided in this project but isn't deployed in the current KOSMOS project
- The connection between clients and host isn't encrypted by now. This can be enabled in the Wrapper (websocket) and flower (grpc) implementation quite easy.

Open Points are:

- The optional but useful security features of Differential Privacy and Secure Multiparty Computation are not implemented yet.

BUILD & RUN LOCALLY WITH DOCKER (RECOMMENDED)
=============================================

To start the kosmos federated learning server for a federated learning session docker containers can be used. First build the kosmos server and the mlflow server containers:

.. code-block::

    docker build --rm -t kosmos_fl_server:latest -f Dockerfiles/kosmos_fl_server.Dockerfile --build-arg GIT_USER=<gitlab username> --build-arg GIT_TOKEN=<gitlab token> .

    docker build --rm -t mlflow:latest -f Dockerfiles/mlflow.Dockerfile - < Dockerfiles/mlflow.Dockerfile #- < Dockerfiles/mlflow.Dockerfile avoid that context is copied to container


After building both containers they can be executed using docker by the following commands:

.. code-block::

    docker run -d --network host --name mlflow  mlflow:latest

    docker run -d --network host --name kosmos_fl_server --entrypoint /app/venv/bin/python kosmos_fl_server:latest kosmos_fl_server.py

While and after training you can connect to the mlflow ui by visiting  `http://localhost:5000 <http://localhost:5000>`_ or the remote server address.


CONFIG.YAML FILE
================

The upbringing of a federated learning server with flower is based on the :code:`config.yaml` file featuring the following parameters:

.. list-table:: Title
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
     - localhost:8080
     - Address and Port of the central flower server
   * - mlflow_server_address
     - url:port or a local directory
     - http://localhost:5000
     - Address to the MLFlow server
   * - mlflow_experiment_name
     - string
     - "Default"
     - The MLFlow experiment the federated training process will be logged to
   * - tags
     - List of one or more tag:value pairs
     - experiment_name: "default_config"
     - These tags will be associated with the MLRun and are important for filtering and comparing multiple federated learning runs
   * - n_federated_train_epoch
     - int >=1
     - 5
     - The number of federated learning iterations with all clients
   * - test_bearing
     - List of bearing names
     - ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7", "Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7", "Bearing3_3",]
     - The names of the bearings used for testing at the central flower server

Note that the bearings available for training and testing are chosen distinctivly from the list of all available bearing.
Because of the nature of federated learning, a bearing should only be used either in test or as a
client train data.

SETUP FOR ALL LOCAL EXCECUTION POSSIBILITIES
============================================

Further to execute the fl_server locally the following steps must be taken. This Project was
developed using python version 3.8 please make sure you are using this version.

1. Install all necessary python packages (create a virtual environment previously if necessary):

.. code-block::

    pip install -r requirements.txt

2. Install the fl_server

.. code-block::

    python setup.py install

3. Further to access the MLFlow logged training process access MLFlow UI server `http://localhost:5000 <http://localhost:5000>`_ after running. MLFlow must be up an running:

.. code-block::

    mlflow ui

4. After executing the mlflow server you can run the kosmos federated learninig server by the following command. Afterward start the client from the `federated learning clients repository <https://github.com/kosmos-industrie40/kosmos-federated-learning-client>`_ :

.. code-block::

    python src/fl_server/kosmos_fl_server.py
    mlflow ui



