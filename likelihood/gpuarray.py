import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule


mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.complex128))

doublify = mod.get_function("doublify")
doublify(a_gpu, block=(4,4,1))

a_doubled = a_gpu.get()
print(a_doubled)
print(a_gpu)
