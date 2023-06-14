### Commands to use this container:
###    docker image build -t docker-exps .
###    docker run -it --rm -v /home/nuc/docker_exps_data:/data docker-exps
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
WORKDIR /app

# Copy python src files
COPY src src
COPY requirements.txt .
COPY Makefile .

# Install all packages
RUN apt-get update && \
    apt-get install -y python3 && \
    apt install -y python3-pip && \
	pip install -r requirements.txt

# Define env variables
ENV DATA_DIR=/data
ENV PYTHONPATH=/app/src

# Run python process
ENTRYPOINT [ "python3" ]
CMD [ "src/docker-exps/main_docker.py" ]