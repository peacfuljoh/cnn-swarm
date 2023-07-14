# Docker experiments

This repository shows how to use Docker to set up a horizontally scalable ML service for training and serving 
ConvNet-based image classification using PyTorch.

This service has two components:
- controller: a Flask app that responds to `train` and `predict` requests.
- trainer: one or more Flask apps that run training jobs and respond to `train` requests.

These are run as stand-alone containers or more conveniently orchestrated as a swarm. 
There can be any number of trainer apps but only one controller app (at the moment).

A notable aspect of this system is that container/service orchestration is handled with Docker while responding to
external `train` and `predict` requests is handled by the Flask apps themselves. Docker is indifferent to the content
of the containers. It just maintains a desired state.


## Stand-alone containers

To run the system with one controller and one trainer, we can set up a docker network for them
to communicate over, then build images from the `Dockerfile_controller` and `Dockerfile_train` dockerfiles and spin up 
individual containers. The `curl` CLI tool or the `requests` Python module can then be used to issue requests
to the controller app. In this implementation, port 48515 is used to expose the controller to outside the docker network.

An easier way to orchestrate this is to use Docker compose and build from `docker-compose.yml`.

### Network

Create a network over which Docker containers can communicate:

`docker network create --subnet 30.0.0.0/24 MLNetwork`

The `subnet` option specifies that we want the network to assign containers attached to it 
IP addresses from 30.0.0.1 through 30.0.0.254 (0 and 255 are reserved).

### Controller

Build the controller image from the dockerfile (`-f`) in the current directory and tag it (`-t` option):

`docker image build -t ml_controller -f Dockerfile_controller .`

Run the controller container: 

`docker run -it --rm
-v /home/nuc/docker_exps_home_dir:/home 
-p 48515:48515 
--name ml_controller 
--network MLNetwork 
-e FLASK_CONTROLLER_HOST=0.0.0.0 
-e FLASK_CONTROLLER_PORT=48515 
-e FLASK_TRAIN_PORT=48516 
-e SUBNET_TRAIN=30.0.0 
-e HOME_DIR=/home 
ml_controller`

The options are as follows:
- `-it`: run the container and open a terminal into it for interaction, otherwise can specify `--detach` to initiate container in the background (usually what you want if not directly debugging)
- `--rm`: removes the container when it shuts down
- `-v`: specifies a directory binding so that the container has indirect access to a location on the host (alternatively could use a temporary volume)
- `-p`: port binding so that container is network-accessible from host/outside docker network
- `--name`: name given to container
- `--network`: docker network over which containers will talk to each other
- `-e`: env vars inside container (used here for Flask args, etc.)

### Trainer

Build and run as with the controller, except with different env vars (see `docker-compose.yml`).


## Docker compose

Instead of building and running containers individually, we can use Docker compose to automate everything 
(see https://docs.docker.com/compose/gettingstarted/).

Specify all options in `docker-compose.yml` in the repo root including all CLI args.

Run `docker compose -p "docker-exps" up --build` from the repo root.

Use `docker compose build` to rebuild images whose Dockerfiles have changed.


## Swarm

We want to horizontally scale the system so that train jobs can be distributed among multiple trainer apps.
We also want the system to be able to maintain and report on its state as well as handle situations where there
is only a controller and no trainers or only trainers and no controller. These may get spun up or 
restarted at any time. 

Each `train` container/task runs the trainer app that has its own message queue for maintaining its overall status. This
is necessary because individual train jobs within the container run in separate threads. The controller can access any train task's 
`status` route to get info about its ongoing jobs.

A docker swarm simplifies horizontal scaling by allowing us to create a `ml_train` service
with any number of replicas. Each one will be assigned a new IP address that the `ml_controller` service
can scan for to know what trainer containers are available at any time.

See https://docs.docker.com/engine/swarm/.

### Swarm init

Initialize swarm on current node (which is a manager): 

`docker swarm init`. 

This will print instructions for adding other nodes to the swarm as workers.

### Overlay network

Create an overlay network for containers to communicate over:

`docker network create --driver overlay --subnet 30.0.0.0/24 MLNetwork`

Swarms use overlay networks while stand-alone containers use bridge networks.
See https://docs.docker.com/engine/swarm/networking/.

### Start services

It is possible to start a service from a docker-compose.yml file, in which case we call it a stack: 
`docker stack deploy --compose-file docker-service-controller.yml s`. However, we can't scale the controller and 
trainer tasks separately that way. Instead, we start the trainer directly with the desired number of replicas. 
We can also scale up and down the number of replicas after-the-fact as desired.

To start the trainer service:

`docker service create --mount type=bind,source=/home/nuc/docker_exps_home_dir,target=/home --name ml_train --network MLNetwork -e FLASK_TRAIN_HOST=0.0.0.0 -e FLASK_TRAIN_PORT=48516 -e HOME_DIR=/home --replicas 4 docker-exps-ml_train`

To start the controller service: 

`docker service create --mount type=bind,source=/home/nuc/docker_exps_home_dir,target=/home --name ml_controller --network MLNetwork -e FLASK_CONTROLLER_HOST=0.0.0.0 -e FLASK_CONTROLLER_PORT=48515 -e HOME_DIR=/home -e FLASK_TRAIN_PORT=48516 -e SUBNET_TRAIN=30.0.0 -p 48515:48515 docker-exps-ml_controller`


### Inspect swarm with an observer

If we want a stand-alone "observer" container (e.g. for debugging networking), we can use an ubuntu image that 
launches and then just sits there:

`docker service create --name test --network MLNetwork ubuntu:22.04 tail -f /dev/null`

This allows us to run `ifconfig`, `ping`, and `curl` commands from within the container in a bash terminal:

`docker exec -it <CONTAINER_NAME> bash`

`apt update && apt install net-tools && apt-get install -y iputils-ping && apt-get install -y curl`

Run `docker container ls` to see container names.

We can access trainer apps via `curl http://<TRAINER_IP>:48516/status` and the controller app
via `curl http://<CONTROLLER_IP>:48515/status`. All of the IP addresses within the overlay network are 
displayed by `docker network inspect MLNetwork`. 
The controller's `status` endpoint provides info about the train jobs as well as all trainer IP addresses.


## Other

Stop all ML services:

`docker service rm ml_controller && docker service rm ml_train`

Build all ML images:

`docker image build -t docker-exps-ml_controller -f Dockerfile_controller . && docker image build -t docker-exps-ml_train -f Dockerfile_train .`

Check ml_controller logs:

`docker service logs ml_controller`

Maintain an up-to-date list of all containers running inside swarm services and their statuses:

`watch "docker node ps"`


# Conclusion

This serves as a MWE of an ML pipeline (from training to serving) implemented with Docker.

Learnings:
- Docker CLI commands: `build`, `run`, `compose`, `network`, `service`, `image`, `container`
- Dockerfiles: prioritizing lightweight image via least-changed/slowest ops first, appropriate base image, multi-stage build
- Networking: creating a bridge or overlay network, connecting containers to them, how IP addresses are assigned, port binding
- Compose: combining Dockerfiles into a single compose .yml file for one-line launch of container stacks with networking
- Swarm: scaling a one-container app to a swarm service with replicas automatically maintained across multiple nodes
- Flask: implementing a network of web apps for managing and running ML jobs

TODO:
- MLOps automation and metadata storage, e.g. to avoid loss of progress when a trainer node goes down while running a job
- More routes on the controller and trainer apps for increased control of jobs and status info
- Multiple controller apps for redundancy and scaling (controller currently handles all `predict` requests), perhaps as a global service (i.e. one on each node)

