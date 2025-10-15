# Custom Dockerfile for Python 3.14 with optimizations
# Use this if you want more control over the build

FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    uuid-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Download and build Python 3.14 from source with free-threading
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.14.0/Python-3.14.0.tgz && \
    tar xzf Python-3.14.0.tgz && \
    cd Python-3.14.0 && \
    ./configure \
        --enable-optimizations \
        --disable-gil \
        --with-lto \
        --enable-shared \
        LDFLAGS="-Wl,-rpath /usr/local/lib" && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-3.14.0*

# Create symlinks
RUN ln -s /usr/local/bin/python3.14 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.14 /usr/local/bin/pip

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHON_GIL=0
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "--version"]