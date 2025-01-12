# test_xla.py
print(">>> Starting test_xla.py")
import torch_xla

print(">>> Importing xr, xm, xmp...")
from torch_xla import runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os

def _mp_fn(index):
    xm.master_print(f"Hello from process {index}. Whoohoo the XLA is working!")

if __name__ == "__main__":
    # Check if TPU is available
    try:
        print("Trying to get TPU devices...")
        print("TPU devices:", xm.get_xla_supported_devices())
    except ImportError as e:
        print("Failed to import torch_xla:", e)

    # Check environment variables
    print("PJRT_DEVICE:", os.getenv("PJRT_DEVICE"))
    print("TPU_IP:", os.getenv("TPU_IP"))

    try:
        device = xm.xla_device()
        print(f"TPU device: {device}")
        xm.master_print("TPU initialized successfully!")

        #xmp.spawn(_mp_fn, nprocs=8)
        torch_xla.launch(_mp_fn, nprocs=8)
    except Exception as e:
        print(f"Error: {e}")
