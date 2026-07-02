FROM ubuntu:22.04

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build tools and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    unzip \
    tar \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract ONNX Runtime based on architecture
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz"; \
        ORT_DIR="onnxruntime-linux-x64-1.17.1"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-aarch64-1.17.1.tgz"; \
        ORT_DIR="onnxruntime-linux-aarch64-1.17.1"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget -q $ORT_URL -O onnxruntime.tgz && \
    tar -xzf onnxruntime.tgz && \
    mv $ORT_DIR /opt/onnxruntime && \
    rm onnxruntime.tgz

# Set up working directory
WORKDIR /app

# Copy the C++ source code, model, and frontend
COPY cpp/ ./cpp/
COPY yolop.onnx ./
COPY edited_ultimate_video.mp4 ./
COPY frontend/ ./frontend/

# Build the C++ application
WORKDIR /app/cpp
RUN mkdir build && cd build && \
    cmake -DONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include -DONNXRUNTIME_LIB_DIR=/opt/onnxruntime/lib .. && \
    make -j4

# Add ONNX Runtime library to the system's dynamic linker path
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH

# Expose the server port
EXPOSE 8080

# Move back to root so the app can find yolop.onnx, the video, and the frontend folder
WORKDIR /app

# Run the application when the container starts
ENTRYPOINT ["./cpp/build/yolop_app", "edited_ultimate_video.mp4", "8080"]
