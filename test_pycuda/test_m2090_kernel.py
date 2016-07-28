import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

a = np.random.randn(4,4)
a = a.astype(np.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
  __global__ void doublify(float *a) {
}
""")


func = mod.get_function("doublify")
func(a_gpu, block=(1024,1,1), grid=(1,1,1))
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print(a_doubled)
print(a)
