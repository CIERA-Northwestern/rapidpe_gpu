import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

ntimes = 2523 # How many time samples to marginalize over
nsamps = 20000 # How many samples we're maginalizing for, is also nrows
padwidth = 1024 - (ntimes % 1024) # width of rectangular padding 
ncols = ntimes + padwidth # How many columns for the 2D array

a = np.ones((3*nsamps, ncols))
# Insert some random numbers to pad out
a[:, ntimes:ncols] = np.random.randn()
a = a.astype(np.complex128)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)


ntimes_h   = np.array(ntimes, ndmin=1).astype(np.int32)
nsamps_h   = np.array(nsamps, ndmin=1).astype(np.int32)
ncols_h    = np.array(ncols, ndmin=1).astype(np.int32)
padwidth_h = np.array(padwidth, ndmin=1).astype(np.int32)


ntimes_gpu = cuda.mem_alloc(ntimes_h.nbytes)
nsamps_gpu = cuda.mem_alloc(nsamps_h.nbytes)
ncols_gpu = cuda.mem_alloc(ncols_h.nbytes)
padwidth_gpu = cuda.mem_alloc(padwidth_h.nbytes)

cuda.memcpy_htod(ntimes_gpu, ntimes_h)
cuda.memcpy_htod(nsamps_gpu, nsamps_h)
cuda.memcpy_htod(ncols_gpu, ncols_h)
cuda.memcpy_htod(padwidth_gpu, padwidth_h)


### We should launch a block of threads whose dimensions are padwidth * nsamps
### And have each of them target an empty slot into which we write a zero 0.0.
mod = SourceModule("""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>


__device__ int get_padded_idx_2d_1d(int ntimes, int ncols) {
	return threadIdx.x + blockIdx.y*ncols + ntimes; 
}

__global__ void pad_with_zeros(cuDoubleComplex *arr_to_pad, int *ntimes, int *nsamps, int *ncols, int *padwidth) {
	
	int gid = get_padded_idx_2d_1d(*ntimes, *ncols);
	arr_to_pad[gid] = make_cuDoubleComplex(0.0, 0.0);	
}

__global__ void insert_ylms(cuDoubleComplex *padded_ts_all, cuDoubleComplex *ylms_all, int *nmodes) {
	__shared__ cuDoubleComplex shr[nmodes]; // My Ylms will reside here	
	if (threadIdx.x < *nmodes) {
		int *my_mode_addr = &ylms_all[*nmodes * blockIdx.y]
		shr[threadIdx.x] = my_mode_addr[threadIdx.x];	
	}
	
	int gid = get_padded_idx_2d_1d(0, *ncols);	
	padded_ts_all[gid] *= shr[blockIdx.y % *nmodes]; 

} 


""")

_reduce_1d_array = mod.get_function("pad_with_zeros")

# We need a grid of blocks with shape (1, nsamps), because we never will
# need to pad with more zeros then to fill up the last block to prep for 
# reduction

_pad_with_zeros = mod.get_function("pad_with_zeros")
_pad_with_zeros(a_gpu, ntimes_gpu, nsamps_gpu, ncols_gpu, padwidth_gpu, grid=(1, 3*nsamps, 1), block=(padwidth, 1, 1))
cuda.memcpy_dtoh(a, a_gpu)

print(a)

# Now lets instantiate some Ylms



