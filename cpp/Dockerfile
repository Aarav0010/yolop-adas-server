FROM ubuntu:22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (Compiler, CMake, OpenCV, Wget)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime for Linux (CPU version for maximum compatibility)
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.17.0.tgz && \
    rm onnxruntime-linux-x64-1.17.0.tgz && \
    cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/include/ && \
    cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/lib/

# Set working directory for our app
WORKDIR /app

# Copy the unzipped source code into the container
COPY . /app/

# Build the C++ application
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Expose the web server port for the dashboard
EXPOSE 8080

# Set the entrypoint to run the app
WORKDIR /app/build
CMD ["./yolop_app"]
