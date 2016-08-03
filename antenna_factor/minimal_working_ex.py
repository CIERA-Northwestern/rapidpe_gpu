import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from math import pi

mydat = np.arange(12).astype(np.float64)
mydat_gpu = gpuarray.to_gpu(mydat)
myresult = np.array(0, ndmin=1).astype(np.float64)
myresult_gpu = gpuarray.to_gpu(myresult)

mod = SourceModule("""
__global__ void my_kernel(double *mydat, double *myresult) {
	extern __shared__ double shr[];
	int id = threadIdx.x;

	double *X = &shr[(id * 6)];
	double *Y = &shr[(id * 6) + 3];

 	X[0] = mydat[0];
	X[1] = mydat[1];	
	X[2] = mydat[2];	
	Y[0] = mydat[3];
	Y[1] = mydat[4];
	Y[2] = mydat[5];


	__syncthreads();	
	
	double result;

	for (int i = 0; i < 3; i++) {
		result += X[i] + Y[i];
	}
	myresult[0] += result;
}
""")

my_kernel = mod.get_function("my_kernel")
blk = (1,1,1)
grd = (1,1,1)

my_kernel(mydat_gpu, myresult_gpu, grid=grd, block=blk, shared=(8*6))
print(myresult_gpu.get())
