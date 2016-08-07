import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule


mod = SourceModule("""
#include <math.h>
#include <cuComplex.h>
__device__ int get_global_idx_1d_1d() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ cuDoubleComplex cpx_outer_prod(cuDoubleComplex *CT, cuDoubleComplex *V) {
	
	cuDoubleComplex result = make_cuDoubleComplex(0.0, 0.0);
	
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			result = cuCadd(result, cuCmul(cuCmul(cuConj(V[i]), CT[i*3 + j]), V[j]));	
		}
	}
	return result;
}

__global__ void make_3x3_outer_prods(cuDoubleComplex *CT, cuDoubleComplex *all_V, cuDoubleComplex *out, int *nmodes, int *nsamps) {

	extern __shared__ cuDoubleComplex allshr[];

	int gid = get_global_idx_1d_1d();	
	int id  = threadIdx.x; 
			
	cuDoubleComplex *myshr = &allshr[*nmodes*id];

	for (int i = 0; i < *nmodes; i++) {
		myshr[i] = all_V[(*nsamps * i) + id];	
	}

	out[gid] = cpx_outer_prod(CT, myshr); 

}

""")

nmodes = np.array(3, ndmin=1).astype(np.int32)
nsamps = np.array(512, ndmin=1).astype(np.int32)

ylms = np.zeros((nmodes, nsamps)).astype(np.complex128)
ylms[0,:] = np.arange(nsamps)
ylms[1,:] = np.arange(nsamps)
ylms[2,:] = np.arange(nsamps)


#import pdb
#pdb.set_trace()
#ylms = ylms.transpose()


nmodes_gpu = gpuarray.to_gpu(nmodes)
nsamps_gpu = gpuarray.to_gpu(nsamps)
ylms_gpu   = gpuarray.to_gpu(ylms)

crossTerms = np.eye(3) * 2.0 
crossTerms_gpu = gpuarray.to_gpu(crossTerms)


out = np.zeros(nsamps).astype(np.complex128)
out_gpu = gpuarray.to_gpu(out)

_make_outer_prods = mod.get_function("make_3x3_outer_prods")

grd = (1,1,1)
blk = (512, 1, 1)
_make_outer_prods(crossTerms_gpu, ylms_gpu, out_gpu, nmodes_gpu, nsamps_gpu, grid=grd, block=blk, shared=(512*3*8*2))
import pdb
pdb.set_trace()

