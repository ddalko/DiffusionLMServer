FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential zip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /tmp
COPY ../requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app