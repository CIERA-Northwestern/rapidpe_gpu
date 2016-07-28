import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

mod = SourceModule('''
__global__ void add(double *a, double *b, double *c) {	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a[tid] + b[tid];
}
''')

N = 10
a = np.arange(N).astype(np.float64)
b = np.arange(N).astype(np.float64)
c = np.zeros(N).astype(np.float64)

print(a)
print("\n")
print(b)
print("\n")
print(c)
print("\n")

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(c_gpu, c)

cu_add = mod.get_function("add") 
cu_add(a_gpu, b_gpu, c_gpu, block=(1,1,1), grid=(N,1))

cuda.memcpy_dtoh(c, c_gpu)
print(c)
