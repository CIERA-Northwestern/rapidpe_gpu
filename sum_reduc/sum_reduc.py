import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

length = 1024
print("Reducing %d ones... \n" % length)



a = np.ones(length)
a = a.astype(np.complex128)
a = a + 1.0j
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)


ans     = np.array(0, ndmin=1).astype(np.complex128)
ans_gpu = cuda.mem_alloc(ans.nbytes)
cuda.memcpy_htod(ans_gpu, ans)

mod = SourceModule("""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>
__global__ void nv_reduc(cuDoubleComplex *result, cuDoubleComplex *indat) {
	extern __shared__ cuDoubleComplex shr[]; 

	// local and lobal thread ID
	unsigned int lid = threadIdx.x; 
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x; 

	shr[lid] = indat[gid];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s /= 2) {
		if (lid < s) {
			//shr[lid] += shr[lid + s]; 
			shr[lid] = make_cuDoubleComplex(cuCreal(shr[lid]) + cuCreal(shr[lid+s]), cuCimag(shr[lid]) + cuCimag(shr[lid+s]));
		}
		__syncthreads();
	}	
	if (lid == 0) {
		*result = shr[0];
	}


}
""")



_reduce_1d_array = mod.get_function("nv_reduc")
_reduce_1d_array(ans_gpu, a_gpu, block=(1024, 1, 1), shared=16*length)

cuda.memcpy_dtoh(ans, ans_gpu)
print(ans)


# Now assume you have been given an n*m array (with m the "long" dimension) and you need to sum over all the columns 




