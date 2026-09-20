FROM python:3.9-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        opencv-python-headless \
        onnxruntime \
        numpy \
        psutil \
        requests \
        smbus2

WORKDIR /app
COPY classification.py basil_mobilenet.onnx ./

CMD ["python3", "classification.py"]
