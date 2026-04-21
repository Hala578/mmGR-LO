"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

try:
    from mpi4py import MPI
except ImportError:
    class _DummyComm:
        rank = 0
        size = 1

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, value, root=0):
            return value

        def gather(self, value, root=0):
            return [value]

    class _DummyMPI:
        COMM_WORLD = _DummyComm()

    MPI = _DummyMPI()

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    comm = MPI.COMM_WORLD
    if comm.Get_size() <= 1:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{comm.Get_rank() % GPUS_PER_NODE}"
    backend = "gloo"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_size() <= 1:
        with bf.BlobFile(path, "rb") as f:
            return th.load(io.BytesIO(f.read()), **kwargs)

    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if comm.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        comm.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            comm.bcast(data[i : i + chunk_size])
    else:
        num_chunks = comm.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += comm.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p.detach(), 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
