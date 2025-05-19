# Use the official Miniconda image
FROM continuumio/miniconda3:latest

# Miniconda builds off of Debian so we need to tell
# Debian that we're running in a non-interactive session
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary compile toolchains
RUN apt-get update && \
    apt-get -y install
        git \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy the environment.yml file into the Docker image
# to provide the base conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Install pip manually in case it is missing in the environment
RUN conda run -n env python -m ensurepip --upgrade
RUN conda run -n env python -m pip install --upgrade pip setuptools wheel


# Set the default shell to use bash and activate the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env"]
