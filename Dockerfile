# ── Base image: RunPod's official PyTorch image with CUDA ──
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model checkpoints
# Your repo should have a checkpoints/ folder with fold_1.pth … fold_5.pth
COPY checkpoints/ /app/checkpoints/

# Copy inference handler
COPY handler.py .

# Environment variables
ENV CHECKPOINT_DIR=/app/checkpoints
ENV PYTHONUNBUFFERED=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
