# Dali / xNVMe
This is a minimal example of changing the data reader in a DALI pipeline to use an external source.
Specifically we are interested in replacing the `fn.readers.file()` data reader with `fn.external_source()`.
The goal is a comparison of 4 different DALI pipelines:

- [x] Using `fn.readers.file()`
- [x] Using an external source with `np.fromfile()`
- [x] Using an external source with xNVMe file API and `torch.frombuffer()`
- [ ] Using an external source with xNVMe block API, a file -> block mapping, and `torch.frombuffer()`

## More about the DALI file reader
`fn.readers.file()` expects a directory with the following structure:
```
  <file_root>/a/img0.jpg
  <file_root>/a/img1.jpg
  <file_root>/b/img2.jpg
  <file_root>/b/img3.jpg
  <file_root>/c/img4.jpg
  <file_root>/c/img5.jpg
```
From this the output will be `(file, label)`:
```
  (<file_root>/a/img0.jpg, 0) 
  (<file_root>/a/img1.jpg, 0)
  (<file_root>/b/img2.jpg, 1)
  (<file_root>/b/img3.jpg, 1)
  (<file_root>/c/img4.jpg, 2)
  (<file_root>/c/img5.jpg, 2)
```
As such, the labels are derived from the sorting of subdirectories.
This is behaviour we need to mimic with the external source for the implementation to work in a plug-n-play manner.

## Setup
- Install dependency
  * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Build docker container
  * `docker build . -t dali-xnvme`
- Run docker container
  * The dataset should reside in a directory, `{datadir}`, with a `train` subdirectory.
  * The `-p` allows the jupyter instance to be opened in a browser on a host machine
  * `docker run --rm --privileged --runtime nvidia -it -p 8888:8888 -v .:/workspace/dali-xnvme -v {datadir}:/data --ipc=host dali-xnvme`
## Jupyter
- run `jupyter-lab`
- open `dali-example.ipynb`
