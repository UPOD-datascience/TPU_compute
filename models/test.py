# test_xla.py
print(">>> Importing libraries...")
import os
#os.environ.pop('TPU_PROCESS_ADDRESSES')
#os.environ.pop('CLOUD_TPU_TASK_ID')

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
    # Check environment variables
    print("PJRT_DEVICE:", os.getenv("PJRT_DEVICE"))
    print("TPU_IP:", os.getenv("TPU_IP"))

    try:
        #device = xm.xla_device()
        #print(f"TPU device: {device}")
        xmp.spawn(_mp_fn)
        xm.master_print("TPU initialized successfully!")
        #torch_xla.launch(_mp_fn)
    except Exception as e:
        print(f"Error: {e}")
