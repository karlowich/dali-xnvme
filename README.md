# Dali / xNVMe
This is a minimal example of using xNVMe as an `external_source` in a DALI pipeline

## Setup
- # Install dependency
  * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- # Build docker container
  * docker build . -t dali-xnvme
- # Run docker container
  * docker run --rm --runtime nvidia -it -p 8888:8888 -v .:/workspace/dali-xnvme -v {datadir}:/data --ipc=host dali-xnvme
## Jupyter
