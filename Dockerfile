FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# Avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary build tools
# Note: OpenCV with CUDA and GStreamer is already included in the l4t-ml base image.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

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
    cmake .. && \
    make -j4

# Expose the server port
EXPOSE 8080

# Move back to root so the app can find yolop.onnx, the video, and the frontend folder
WORKDIR /app

# Run the application when the container starts
ENTRYPOINT ["./cpp/build/yolop_app", "edited_ultimate_video.mp4", "8080"]
