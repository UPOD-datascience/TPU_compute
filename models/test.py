# test_xla.py
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def _mp_fn(index):
    xm.master_print(f"Hello from process {index}. Whoohoo the XLA is working!")

if __name__ == "__main__":
    xm.master_print("Hello from test.py!")
    xmp.spawn(_mp_fn, nprocs=8)
