FROM ubuntu:22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build tools and CPU OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime for aarch64 (Jetson Orin CPU)
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz && \
    tar -zxvf onnxruntime-linux-aarch64-1.16.3.tgz && \
    cp -r onnxruntime-linux-aarch64-1.16.3/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-aarch64-1.16.3/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-aarch64-1.16.3*

# Set up working directory
WORKDIR /app

# Copy the C++ source code, model, and frontend
COPY cpp/ ./cpp/
COPY yolop_static.onnx ./
COPY edited_ultimate_video.mp4 ./
COPY frontend/ ./frontend/

# Build the C++ application
WORKDIR /app/cpp
RUN mkdir build && cd build && \
    cmake .. && \
    make -j4

# Expose the server port
EXPOSE 8080

# Move back to root so the app can find the ONNX model, the video, and the frontend folder
WORKDIR /app

# Run the application when the container starts
ENTRYPOINT ["./cpp/build/yolop_app", "edited_ultimate_video.mp4", "8080"]
