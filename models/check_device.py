import cupy as cp
print(cp.cuda.Device(0).mem_info)