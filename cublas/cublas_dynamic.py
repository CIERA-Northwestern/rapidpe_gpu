##### GPU ACCELERATED SPHERICAL HARMONICS

#_ Preliminaries
import numpy as np
import pycuda
import pycuda.driver as cuda
# SourceModule is the thing that allows us to
# write cuda c modules 
from pycuda.compiler import SourceModule
import pycuda.autoinit
import math

mod=SourceModule('''
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void call_zgemv_gpu(int m, int n, cuDoubleComplex* alpha_gpu, cuDoubleComplex *A_gpu, int ndim, cuDoubleComplex* vec_gpu, cuDoubleComplex* beta_gpu, cuDoubleComplex* result_gpu) {

        cublasHandle_t handle;
        cublasCreate(&handle);
        int tid = threadIdx.x + blockIdx.x*blockDim.x;

        cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_gpu, A_gpu, ndim, vec_gpu, 1, beta_gpu, &result_gpu[tid * ndim], 1);
        cudaDeviceSynchronize();

        cublasDestroy(handle);
}
''', options=['-rdc=true', '-lcublas_device', '-lcudadevrt'])

#_____________________
# Actual function
#_____________________

# Put the compiled c module in a python object
import_cublas = mod.get_function("call_zgemv_gpu")






