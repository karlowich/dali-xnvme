ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.11-py3
FROM ${FROM_IMAGE_NAME}

ADD requirements.txt /workspace/
WORKDIR /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Install xallib (xfs*), BaM (cmake, linx-headers), and xNVMe bindings (libclang) dependencies
RUN echo deb http://archive.ubuntu.com/ubuntu/ focal-updates main restricted >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get -y install cmake linux-headers-$(uname -r) xfsprogs xfslibs-dev libclang-dev

# Install BaM
RUN git clone https://github.com/ZaidQureshi/bam.git
WORKDIR /workspace/bam/
RUN git submodule update --init --recursive
RUN mkdir -p build
WORKDIR /workspace/bam/build
RUN cmake ..
RUN make libnvm
RUN make install
WORKDIR /workspace/bam/build/module
RUN make
WORKDIR /workspace/

# Install xNVMe
RUN git clone https://github.com/karlowich/xnvme.git
WORKDIR /workspace/xnvme/
RUN git checkout bam
RUN git submodule update --init --recursive
# Install xNVMe dependencies
RUN ./toolbox/pkgs/ubuntu-jammy.sh
RUN meson setup builddir -Dwith-spdk=disabled && meson compile -C builddir && meson install -C builddir
WORKDIR /workspace/

# Install python bindings
WORKDIR /workspace/xnvme/python/bindings/
RUN pip install -r requirements.txt --user
RUN make build
RUN pip install dist/xnvme-*.tar.gz
WORKDIR /workspace/

# Install xallib
RUN git clone https://github.com/safl/xallib.git
WORKDIR /workspace/xallib/
RUN make
WORKDIR /workspace/


WORKDIR /workspace/dali-xnvme
