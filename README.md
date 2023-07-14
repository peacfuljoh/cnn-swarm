# docker-exps
Experiments with Docker images

# CLI commands

## Network

Create a network for Docker containers to communicate with each
other through: 

`docker network create MLNetwork`

Running a container in the following form allows the host to access the container on port 8080
and other containers on the network to access it on port 80.

`docker run --network MyNetwork --name Container1 -p 8080:80 Image1`

## Individual containers

### Controller

Build the controller image: 

`docker image build -t ml_controller -f Dockerfile_controller .`

Run the controller container: 

`docker run -it --rm --detach -v /home/nuc/docker_exps_home_dir:/home -p 48515:48515 --name ml_controller --network MLNetwork -e FLASK_CONTROLLER_HOST=0.0.0.0 -e FLASK_CONTROLLER_PORT=48515 -e URL_TRAIN_INTERNAL=http://127.0.0.1:48516 -e HOME_DIR=/home ml_controller`

### Trainer

Build the trainer image: 

`docker image build -t ml_train -f Dockerfile_train .`

Run the trainer container: 

`docker run -it --rm --detach -v /home/nuc/docker_exps_home_dir:/home -p 48516:48516 --name ml_train --network MLNetwork -e FLASK_TRAIN_HOST=0.0.0.0 -e FLASK_TRAIN_PORT=48516 ml_train`


## Docker compose

To run the controller and trainer containers using Docker compose, simply use the `docker-compose.yml` file in the repo root.
Run `docker compose -p "docker-exps" up --build` from the repo root.

Use `docker compose build` to rebuild images whose Dockerfiles have changed.


## Swarm commands

Initialize swarm on current node: `docker swarm init`

Create an overlay network (with desired subnet) for service replicas to communicate over: 

`docker network create --driver overlay --subnet 30.0.0.0/24 MLNetwork`

Start ml_controller service from .yml file as stack: `docker stack deploy --compose-file docker-service-controller.yml s`

Start ml_train service directly: 
`docker service create --mount type=bind,source=/home/nuc/docker_exps_home_dir,target=/home --name ml_train --network MLNetwork -e FLASK_TRAIN_HOST=0.0.0.0 -e FLASK_TRAIN_PORT=48516 -e HOME_DIR=/home --replicas 4 docker-exps-ml_train`

Start ml_controller service directly: 
`docker service create --mount type=bind,source=/home/nuc/docker_exps_home_dir,target=/home --name ml_controller --network MLNetwork -e FLASK_CONTROLLER_HOST=0.0.0.0 -e FLASK_CONTROLLER_PORT=48515 -e HOME_DIR=/home -e FLASK_TRAIN_PORT=48516 -e SUBNET_TRAIN=30.0.0 -p 48515:48515 docker-exps-ml_controller`


## Start 1-container service to inspect swarm

`docker service create --name test --network MLNetwork --replicas 1 ubuntu:22.04 tail -f /dev/null`

To use `ifconfig`, `ping`, and `curl`: `docker exec -it <CONTAINER_NAME> bash`, then 
`apt update && apt install net-tools && apt-get install -y iputils-ping && apt-get install curl`

Run `docker container ls` to get container names.


## Other

Stop all ML services:
`docker service rm ml_controller && docker service rm ml_train`

Build all ML images:
`docker image build -t docker-exps-ml_controller -f Dockerfile_controller . && docker image build -t docker-exps-ml_train -f Dockerfile_train .`

Check ml_controller logs:
`docker service logs ml_controller`
