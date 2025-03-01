ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.01-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Install basic tools
RUN apt-get update
RUN apt-get -y install pciutils kmod linux-modules-extra-$(uname -r)

# Install xNVMe
RUN git clone https://github.com/karlowich/xnvme.git
WORKDIR /workspace/xnvme/
RUN git checkout for-ubuntu-focal
RUN ./toolbox/pkgs/ubuntu-focal.sh
RUN meson setup builddir && meson compile -C builddir && meson install -C builddir

# Install python bindings
RUN pip install xnvme

WORKDIR /workspace/dali-xnvme
