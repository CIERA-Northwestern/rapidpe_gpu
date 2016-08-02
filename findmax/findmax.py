### This is the gpu array class
import pycuda.gpuarray as gpuarray
###

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule


mod = SourceModule("""
__device__ int get_global_idx_2d_1d() {
	return gridDim.x*blockDim.x*blockIdx.y + blockDim.x*blockIdx.x + threadIdx.x;
}

__global__ void find_max_in_shrmem(double *myarr) {
	extern double __shared__ shr[];
	int linidx = get_global_idx_2d_1d();

	shr[threadIdx.x] = myarr[linidx];

	__syncthreads();

	for (unsigned int s=blockDim.x/2; s > 0; s /= 2) {
		if (threadIdx.x < s) {
			if (shr[threadIdx.x] <= shr[threadIdx.x + s]) {
				shr[threadIdx.x] = shr[threadIdx.x + s];
			}
		}	
		__syncthreads();
	}	
	
	if (threadIdx.x == 0) {
		myarr[blockIdx.x] = shr[0];	
	}	

}	
""")


a = np.random.uniform(0,1000.0, 1000.0).astype(np.float64)
print("NumPy finds max of %f \n" % np.max(a))
a_gpu = gpuarray.to_gpu(a)

findmax = mod.get_function("find_max_in_shrmem")
findmax(a_gpu, grid=(2,1,1), block=(512,1,1), shared=512*8)
findmax(a_gpu, grid=(1,1,1), block=(2,  1,1), shared=512*8)


a_doubled = a_gpu.get()
print("GPU finds max of %f \n" % a_doubled[0])
