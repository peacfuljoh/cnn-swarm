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
Run `docker compose -p "docker-exps" up` from the repo root.

Use `docker compose build` to rebuild images whose Dockerfiles have changed.

