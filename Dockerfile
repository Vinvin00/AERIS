# Public web demo backend (Render). Linux has no hash-locked requirements yet, so this
# resolves the pyproject.toml ranges plus the onnxruntime pin; the weights are SHA-256 checked.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir "numpy>=2,<3" "opencv-python-headless>=5,<6" "scipy==1.17.1" "onnxruntime==1.30.0"
RUN mkdir -p models/onnx && curl -fL -o models/onnx/yolox_tiny.onnx \
      https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx \
    && echo "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7  models/onnx/yolox_tiny.onnx" | sha256sum -c -
COPY python python
COPY configs configs
ENV PYTHONPATH=/app/python PYTHONUNBUFFERED=1
RUN useradd -m app && mkdir -p benchmarks/runs && chown -R app /app
USER app
CMD ["sh", "-c", "exec python -m aeris.web.server --hosted --no-open --port ${PORT:-10000}"]
