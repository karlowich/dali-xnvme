import os
import sys
import time

import json
import argparse
import torch
from random import shuffle
import numpy as np 
import cupy as cp
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from ctypes import POINTER, c_uint32, c_uint8, c_void_p, byref, pointer, cast
import xnvme.ctypes_bindings as xnvme
from xnvme.ctypes_bindings.api import char_pointer_cast

IMAGENET_MAX_SIZE = 211025920 # max size of images in the imagenet dataset

@pipeline_def
def dali_pipe(datadir):
    traindir = os.path.join(datadir, "train")
    jpegs, labels = fn.readers.file(file_root=traindir,
            shard_id=0,
            num_shards=1,
            random_shuffle=True,
            pad_last_batch=True,
            name="FILE"
        )

    return jpegs.gpu(), labels


# Based on https://github.com/NVIDIA/DALI/blob/main/docs/examples/frameworks/pytorch/pytorch-external_input.ipynb
class FileInputIterator(object):
    # This is essentially fn.readers.file reimplemented as an external source
    def __init__(self, datadir, batch_size):
        self.images_dir = os.path.join(datadir, "train")
        self.batch_size = batch_size
        dirs = {os.path.join(self.images_dir, d): i for i, d in enumerate(sorted(os.listdir(self.images_dir)))}
        self.files = [(os.path.join(root, file), dirs[root]) for root, _, files in os.walk(self.images_dir) for file in files]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg, label = self.files[self.i % self.n]

            batch.append(
                np.fromfile(jpeg, dtype=np.uint8)
            )
            labels.append(
                torch.tensor([label], dtype=torch.int32)
            )
            self.i += 1
        return (batch, labels)

    def __len__(self):
        return self.n

    next = __next__
    

@pipeline_def
def file_pipe(datadir):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=FileInputIterator(datadir=datadir, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32]
    )
    
    return jpegs.gpu(), labels


class XNVMEFileInputIterator(object):
    # This is implemented using the xNVMe file API
    def __init__(self, datadir, batch_size):
        self.images_dir = os.path.join(datadir, "train")
        self.batch_size = batch_size
        dirs = {os.path.join(self.images_dir, d): i for i, d in enumerate(sorted(os.listdir(self.images_dir)))}
        self.files = [(os.path.join(root, file), dirs[root]) for root, _, files in os.walk(self.images_dir) for file in files]
        opts = xnvme.xnvme_opts()
        xnvme.xnvme_opts_set_defaults(byref(opts))
        # Make tmp file to link buffers to
        file_path = "/tmp/xnvme_file"
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write("tmp")
        self.dev = xnvme.xnvme_file_open(char_pointer_cast(file_path), byref(opts))
        size = IMAGENET_MAX_SIZE
        self.buffers = (c_void_p * self.batch_size)()
        self.views = []
        for i in range(self.batch_size):
            buf = xnvme.xnvme_buf_alloc(self.dev, size)
            view = np.ctypeslib.as_array(
                    cast(buf, POINTER(c_uint8)),
                    shape=(size,),
            )
            self.buffers[i] = cast(buf, c_void_p)
            self.views.append(view)

        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        
        opts = xnvme.xnvme_opts()
        xnvme.xnvme_opts_set_defaults(byref(opts))
        for i in range(self.batch_size):
            jpeg, label = self.files[self.i % self.n]
                
            file = xnvme.xnvme_file_open(char_pointer_cast(jpeg), byref(opts))
            size = xnvme.xnvme_dev_get_geo(file).contents.tbytes
            ctx = xnvme.xnvme_file_get_cmd_ctx(file)
            err = xnvme.xnvme_file_pread(byref(ctx), self.buffers[i], size, 0)
            
            batch.append(
                self.views[i][:size]    
            )
           
            labels.append(
                torch.tensor([label], dtype=torch.int32)
            )
            xnvme.xnvme_file_close(file)
            
            self.i += 1
        
        return (batch, labels)

    def __len__(self):
        return self.n

    def __del__(self):
        for i in range(self.batch_size):
            xnvme.xnvme_buf_free(self.dev, self.buffers[i])
        xnvme.xnvme_file_close(self.dev)

    next = __next__
    
@pipeline_def
def xnvme_file_pipe(datadir):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=XNVMEFileInputIterator(datadir=datadir, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32],
    )
    return jpegs.gpu(), labels


class XNVMECPUInputIterator(object):
    def __init__(self, dev, bmap, batch_size):
        with open(bmap) as f:
            d = json.load(f)
            dirs = sorted(list(set([k.split("/")[1] for k in d.keys()])))
            for k in d.keys():
                d[k]["label"] = dirs.index(k.split("/")[1])
            self.files = list(d.values())
        
        self.dev = dev
        self.batch_size = batch_size
        self.nsid = xnvme.xnvme_dev_get_nsid(self.dev)
        geo = xnvme.xnvme_dev_get_geo(self.dev)
        self.nlb = 7
        self.nbytes = geo.contents.nbytes
        size = IMAGENET_MAX_SIZE 
        self.qd = 63
        self.buffers = (c_void_p * self.batch_size)()
        self.views = []
        for i in range(self.batch_size):
            buf = xnvme.xnvme_buf_alloc(dev, size)
            view = np.ctypeslib.as_array(
                    cast(buf, POINTER(c_uint8)),
                    shape=(size,),
            )
            self.buffers[i] = cast(buf, c_void_p)
            self.views.append(view)
        self.queue = POINTER(xnvme.xnvme_queue)()
        err = xnvme.xnvme_queue_init(self.dev, self.qd, 0, byref(self.queue))
        if err != 0:
            print("Failed to init queue:", err)
            raise RuntimeError
        
        self.n = len(self.files)


    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        slbas = (c_uint32 * self.batch_size)()
        elbas = (c_uint32 * self.batch_size)()

        for i in range(self.batch_size):
            file = self.files[self.i % self.n]
            labels.append(
                torch.tensor([file["label"]], dtype=torch.int32)
            )
            slbas[i] = c_uint32(file["startblock"])
            elbas[i] = c_uint32(file["endblock"])
            self.i += 1
        
        err = xnvme.xnvme_io_range_submit(self.queue, xnvme.XNVME_SPEC_NVM_OPC_READ, slbas, elbas, self.nlb, self.nbytes, self.buffers, self.batch_size)
        if err:
            print("Err: ", err)

        batch = [self.views[i][:((elbas[i] - slbas[i])+ 1) * self.nbytes] for i in range(self.batch_size)]
        return (batch, labels)

    def __len__(self):
        return self.n
    
    def __del__(self):
        for i in range(self.batch_size):
            xnvme.xnvme_buf_free(self.dev, self.buffers[i])
        xnvme.xnvme_queue_term(self.queue)

    next = __next__

   
@pipeline_def
def xnvme_gds_cpu_pipe(dev, bmap):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=XNVMECPUInputIterator(dev=dev, bmap=bmap, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32]
    )

    return jpegs.gpu(), labels


class XNVMEGPUInputIterator(object):
    def __init__(self, dev, bmap, batch_size):
        with open(bmap) as f:
            d = json.load(f)
            dirs = sorted(list(set([k.split("/")[1] for k in d.keys()])))
            for k in d.keys():
                d[k]["label"] = dirs.index(k.split("/")[1])
            self.files = list(d.values())
        
        self.dev = dev
        self.batch_size = batch_size
        self.nsid = xnvme.xnvme_dev_get_nsid(self.dev)
        geo = xnvme.xnvme_dev_get_geo(self.dev)
        self.nlb = 7
        self.nbytes = geo.contents.nbytes
        size = IMAGENET_MAX_SIZE
        self.qd = 63
        self.buffers = (c_void_p * self.batch_size)()
        self.views = []
        for i in range(self.batch_size):
            buf = xnvme.xnvme_buf_alloc(dev, size)
            self.buffers[i] = cast(buf, c_void_p)
            view = cp.ndarray(shape=(size,), dtype=np.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buf, size, dev), 0))
            self.views.append(view)
        self.queue = POINTER(xnvme.xnvme_queue)()
        err = xnvme.xnvme_queue_init(self.dev, self.qd, 0, byref(self.queue))
        if err != 0:
            print("Failed to init queue:", err)
            raise RuntimeError
        
        self.n = len(self.files)


    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        slbas = (c_uint32 * self.batch_size)()
        elbas = (c_uint32 * self.batch_size)()

        for i in range(self.batch_size):
            file = self.files[self.i % self.n]
            labels.append(
                torch.tensor([file["label"]], dtype=torch.int32)
            )
            slbas[i] = c_uint32(file["startblock"])
            elbas[i] = c_uint32(file["endblock"])
            self.i += 1
        
        err = xnvme.xnvme_io_range_submit(self.queue, xnvme.XNVME_SPEC_NVM_OPC_READ, slbas, elbas, self.nlb, self.nbytes, self.buffers, self.batch_size)
        if err:
            print("Err: ", err)

        batch = [self.views[i][:((elbas[i] - slbas[i])+ 1) * self.nbytes] for i in range(self.batch_size)]
        return (batch, labels)

    def __len__(self):
        return self.n
    
    def __del__(self):
        for i in range(self.batch_size):
            xnvme.xnvme_buf_free(self.dev, self.buffers[i])
        xnvme.xnvme_queue_term(self.queue)

    next = __next__


@pipeline_def
def xnvme_gds_gpu_pipe(dev, bmap):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=XNVMEGPUInputIterator(dev=dev, bmap=bmap, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32]
    )

    return jpegs, labels


class XNVMEBAMInputIterator(object):
    def __init__(self, dev, bmap, batch_size):
        with open(bmap) as f:
            d = json.load(f)
            dirs = sorted(list(set([k.split("/")[1] for k in d.keys()])))
            for k in d.keys():
                d[k]["label"] = dirs.index(k.split("/")[1])
            self.files = list(d.values())
        
        self.dev = dev
        self.batch_size = batch_size
        self.nsid = xnvme.xnvme_dev_get_nsid(self.dev)
        geo = xnvme.xnvme_dev_get_geo(self.dev)
        self.nlb = 7
        self.nbytes = geo.contents.nbytes
        size = IMAGENET_MAX_SIZE
        buffers = []
        self.views = []
        for i in range(self.batch_size):
            buf = xnvme.xnvme_buf_alloc(dev, size)
            buffers.append(buf)
            view = cp.ndarray(shape=(size,), dtype=np.uint8, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buf, size, dev), 0))
            self.views.append(view)
            
        self.buffers = cp.asarray(buffers)
        self.buffers_ref = cast(self.buffers.data.ptr, POINTER(c_void_p))
        
        self.n = len(self.files)


    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        labels = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        slbas = cp.ndarray(shape=(self.batch_size,), dtype=cp.uint32)
        elbas = cp.ndarray(shape=(self.batch_size,), dtype=cp.uint32)
        for i in range(self.batch_size):
            file = self.files[self.i % self.n]
            labels.append(
                torch.tensor([file["label"]], dtype=torch.int32)
            )
            slbas[i] = file["startblock"]
            elbas[i] = file["endblock"]
            
            self.i += 1
        y = 64
        total_blocks = sum([(elbas[i]-slbas[i]) + 1 for i in range(self.batch_size)])
        x = int(total_blocks) // 64
        
        err = xnvme.xnvme_kernels_range_submit(x, y, self.dev, xnvme.XNVME_SPEC_NVM_OPC_READ, cast(slbas.data.ptr, POINTER(c_uint32)), cast(elbas.data.ptr, POINTER(c_uint32)), self.nlb, self.nbytes, self.buffers_ref, self.batch_size)
        if err:
            print("Err: ", err)

        batch = [self.views[i][:((elbas[i] - slbas[i])+ 1) * self.nbytes] for i in range(self.batch_size)]
        return (batch, labels)

    def __len__(self):
        return self.n
    
    def __del__(self):
        for i in range(self.batch_size):
            xnvme.xnvme_buf_free(self.dev, int(self.buffers[i]))

    next = __next__


@pipeline_def
def xnvme_bam_pipe(dev, bmap):
    pipe = Pipeline.current()
    batch_size = pipe.max_batch_size

    jpegs, labels = fn.external_source(
        source=XNVMEBAMInputIterator(dev=dev, bmap=bmap, batch_size=batch_size), num_outputs=2, dtype=[types.UINT8, types.INT32]
    )

    return jpegs, labels


def dev_open(uri, be, mem):
    opts = xnvme.xnvme_opts()
    xnvme.xnvme_opts_set_defaults(byref(opts))
    opts.be = char_pointer_cast(be)
    if mem:
        opts.mem = char_pointer_cast(mem)
    dev = xnvme.xnvme_dev_open(char_pointer_cast(uri), byref(opts))
    if not dev:
        print("xNVMe failed to open device")
    xnvme.xnvme_dev_derive_geo(dev)
    return dev


def dev_close(dev):
    xnvme.xnvme_dev_close(dev)


def setup():
    """Parse command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Benchmark loading data with DALI and xNVMe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batchsize", required=True, help='The number of samples to files per batch', type=int)
    parser.add_argument("--batches", required=False, help='The number of batches to load, default loads the entire dataset', type=int)
    subparsers = parser.add_subparsers(dest='dataloader', required=True)
    # Subcommand 'add'
    parser_dali = subparsers.add_parser('dali', help='Use DALI file reader for loading data')
    parser_dali.add_argument(
        "--datadir", help="The directory from which to load files", required=True
    )

    parser_python = subparsers.add_parser('python-file', help='Use Python file reader for loading data')
    parser_python.add_argument(
        "--datadir", help="The directory from which to load files", required=True
    )

    parser_xnvme_file = subparsers.add_parser('xnvme-file', help='Use xNVMe file API for loading data')
    parser_xnvme_file.add_argument(
        "--datadir", help="The directory from which to load files", required=True
    )

    parser_xnvme_gds = subparsers.add_parser('xnvme-gds', help='Use xNVMe GDS-like backend for loading data')
    parser_xnvme_gds.add_argument(
        "--uri", help="The uri of the device to read from", required=True
    )
    parser_xnvme_gds.add_argument(
        "--mem", choices=["cpu", "gpu"], help="The desired xNVMe memory interface", required=True
    )
    parser_xnvme_gds.add_argument(
        "--bmap", help="The path to the bmap JSON file used for xNVMe", required=True
    )
    parser_xnvme_bam = subparsers.add_parser('xnvme-bam', help='Use xNVMe BaM-like backend for loading data')
    parser_xnvme_bam.add_argument(
        "--uri", help="The uri of the device to read from", required=True
    )
    parser_xnvme_bam.add_argument(
        "--bmap", help="The path to the bmap JSON file used for xNVMe", required=True
    )
    args = parser.parse_args()

    return args


def main():    

    args = setup()
    pipe = None
    if args.dataloader == "dali":
        print(f"dataloader: {args.dataloader}, datadir: {args.datadir}")
        pipe = dali_pipe(datadir=args.datadir, batch_size=args.batchsize, num_threads=1, device_id=1)
        
    elif args.dataloader == "python-file":
        print(f"dataloader: {args.dataloader}, datadir: {args.datadir}")
        pipe = file_pipe(datadir=args.datadir, batch_size=args.batchsize, num_threads=1, device_id=1)

    elif args.dataloader == "xnvme-file":
        print(f"dataloader: {args.dataloader}, datadir: {args.datadir}")
        pipe = xnvme_file_pipe(datadir=args.datadir, batch_size=args.batchsize, num_threads=1, device_id=1)

    elif args.dataloader == "xnvme-gds":
        print(f"dataloader: {args.dataloader}, uri: {args.uri}, mem: {args.mem}, bmap: {args.bmap}")
        dev = dev_open(args.uri, "gds", args.mem)
        if args.mem == "cpu":
            pipe = xnvme_gds_cpu_pipe(dev=dev, bmap=args.bmap, batch_size=args.batchsize, num_threads=1, device_id=0)
        if args.mem == "gpu":
            pipe = xnvme_gds_gpu_pipe(dev=dev, bmap=args.bmap, batch_size=args.batchsize, num_threads=1, device_id=0)
        
    elif args.dataloader == "xnvme-bam":
        print(f"dataloader: {args.dataloader}, uri: {args.uri}, bmap: {args.bmap}")
        dev = dev_open(args.uri, "bam", None)
        pipe = xnvme_bam_pipe(dev=dev, bmap=args.bmap, batch_size=args.batchsize, num_threads=1, device_id=0)

    print(f"batches: {args.batches}, batchsize: {args.batchsize}")

    pipe.build()

    batches = args.batches

    if (not batches):
        # Either run forever and wait for exception (GDS, BaM) or set limit (DALI)
        batches = sys.maxsize
        if args.dataloader == "dali":
            limit = pipe.epoch_size("FILE")
            batches = limit // args.batchsize
            if (limit % args.batchsize != 0):
                # if there is a remainder do one more iteration
                batches += 1

    start = time.time()
    for _ in range(batches):
        try:
            jpegs, _ = pipe.run()
        except StopIteration:
            break

    end = time.time()
    print(end - start)
    exit(0)

if __name__ == "__main__":
    sys.exit(main())    
