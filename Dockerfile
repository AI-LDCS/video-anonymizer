# The anonymizer worker: worker.py polls the field-capture app and, for each recording, cuts still images
# (sample_frames.py), hides faces and plates (anonymize.py) and tags GPS (tag_gps.py); or anonymizes the whole video.
#
# GPU (the normal way, Linux with an NVIDIA GPU, the driver and nvidia-container-toolkit on the host):
#   docker compose --profile gpu up -d --build
# CPU (slow; for a laptop or a machine without an NVIDIA GPU, including Docker Desktop on macOS):
#   docker compose --profile cpu up -d --build
#
# TORCH_INDEX picks the PyTorch build. Empty (the default) installs the standard Linux wheel from PyPI, which
# bundles CUDA, so no CUDA base image is needed: the host only needs a new enough NVIDIA driver. The gpu profile
# in docker-compose.yml sets the CUDA 13.0 index (the tested build); for an older driver use a matching index
# (for example https://download.pytorch.org/whl/cu126). The cpu profile sets the CPU-only index.
FROM python:3.12-slim
ARG TORCH_INDEX=
ENV PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
# libGL and glib are needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
# On arm64 (Apple Silicon Docker, ARM servers) newer OpenCV wheels (4.11 and later, 5.x) die with "Illegal instruction"
# on import inside the VM, so pin the last version that works there. x86 uses whatever requirements.txt resolves to.
RUN if [ "$(uname -m)" = "aarch64" ]; then echo "opencv-python==4.10.0.84" > /constraints.txt; else : > /constraints.txt; fi
# torch and torchvision must come from the same index, or torchvision's compiled ops do not match torch
RUN if [ -n "$TORCH_INDEX" ]; then pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"; \
    else pip install --no-cache-dir torch torchvision; fi \
 && pip install --no-cache-dir -c /constraints.txt -r requirements.txt
COPY anonymize.py worker.py sample_frames.py tag_gps.py ./
# the model is mounted (see docker-compose.yml), not baked into the image
CMD ["python", "worker.py"]
