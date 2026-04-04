FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and add repositories
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:nnstreamer/ppa \
    && apt-get update && apt-get install -y \
    build-essential \
    cmake \
    meson \
    ninja-build \
    pkg-config \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    gcc \
    g++ \
    libopenblas-dev \
    libiniparser-dev \
    libjsoncpp-dev \
    libcurl4-openssl-dev \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtest-dev \
    libunwind-dev \
    flatbuffers-compiler \
    libprotobuf-dev \
    protobuf-compiler \
    tensorflow2-lite-dev \
    nnstreamer-dev \
    ml-api-common-dev \
    ml-inference-api-dev \
    android-tools-adb \
    && rm -rf /var/lib/apt/lists/*

# Install protobuf 23.2
# WORKDIR /tmp
# RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v23.2/protoc-23.2-linux-x86_64.zip && \
#     unzip protoc-23.2-linux-x86_64.zip -d protoc-23.2 && \
#     cp protoc-23.2/bin/protoc /usr/local/bin/ && \
#     cp -r protoc-23.2/include/* /usr/local/include/ && \
#     ldconfig

# Set working directory to match the host path

# Set up environment variables to use protoc 23.2
ENV PATH="/usr/local/bin:${PATH}"

# Install TensorFlow and numpy for golden test generation
RUN pip3 install --quiet tensorflow numpy

# Run as root to avoid permission issues with mounted volumes
CMD ["/bin/bash"]
