ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.01-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/karlowich/xnvme.git
WORKDIR /workspace/xnvme/
RUN git checkout for-ubuntu-focal
RUN ./toolbox/pkgs/ubuntu-focal.sh
RUN meson setup builddir && meson compile -C builddir && meson install -C builddir
WORKDIR /workspace/dali-xnvme
