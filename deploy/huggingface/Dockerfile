# Hugging Face Space image: builds the AERIS web demo from the GitHub repo at AERIS_REF.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy>=2,<3" "opencv-python-headless>=5,<6" "scipy==1.17.1" "onnxruntime==1.30.0"
ARG AERIS_REF=deploy-hosted
WORKDIR /app
RUN git clone --depth 1 --branch ${AERIS_REF} https://github.com/Vinvin00/AERIS /tmp/src \
    && cp -r /tmp/src/python /tmp/src/configs . && rm -rf /tmp/src
RUN mkdir -p models/onnx && curl -fL -o models/onnx/yolox_tiny.onnx \
      https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx \
    && echo "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7  models/onnx/yolox_tiny.onnx" | sha256sum -c -
ENV PYTHONPATH=/app/python PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 app && mkdir -p benchmarks/runs && chown -R app /app
USER app
CMD ["python", "-m", "aeris.web.server", "--hosted", "--no-open", "--port", "10000"]
