ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.01-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /workspace/dali-xnvme
