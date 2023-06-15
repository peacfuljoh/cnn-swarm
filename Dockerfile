### Commands to use this container:
###    docker image build -t docker_exps .
###    docker run -it --rm -v /home/nuc/docker_exps_home_dir:/home -p 48513:48513 docker_exps
###
###

# Download base image ubuntu 22.04
FROM ubuntu:22.04

# author
MAINTAINER Johannes Traa

# extra metadata
LABEL version="0.0.1"
LABEL description="Experiments with Docker"

# Set working dir
WORKDIR /code

# Copy python src files
COPY src/docker_exps/app src/docker_exps/app
COPY requirements.txt .
COPY Makefile .

# Install all packages
RUN apt-get update && \
    apt-get install -y python3 && \
    apt install -y python3-pip && \
	pip install -r requirements.txt

# Define env variables
ENV PYTHONPATH=/code
ENV HOME_DIR=/home
ENV FLASK_HOST="10.0.0.34"
ENV FLASK_PORT="48513"

# Run python process
ENTRYPOINT [ "python3" ]
CMD [ "src/docker_exps/app.py" ]