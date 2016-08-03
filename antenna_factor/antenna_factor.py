import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from math import pi


# FIXME - Detector tensor should definitely go into constant memory
mod = SourceModule("""
#include <math.h>
#include <cuComplex.h>
__device__ int get_global_idx_1d_1d() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void complex_antenna_factor(double *R, double *ra, double *dec, double *psi, double *tref, cuDoubleComplex *result) {
	
	extern __shared__ double shr[];	

	int gid = get_global_idx_1d_1d();	
	int id  = threadIdx.x;

	double *X = &shr[(id * 6)];
	double *Y = &shr[(id * 6) + 3];

	double gha = tref[gid] - ra[gid]; 
	X[0] = -cos(psi[gid]) * sin(gha) - sin(psi[gid]) * cos(gha) * sin(dec[gid]);  
	X[1] = -cos(psi[gid]) * cos(gha) + sin(psi[gid]) * sin(gha) * sin(dec[gid]);
	X[2] =  sin(psi[gid]) * cos(dec[gid]);
	Y[0] =  sin(psi[gid]) * sin(gha) - cos(psi[gid]) * cos(gha) * sin(dec[gid]); 
	Y[1] =  sin(psi[gid]) * cos(gha) + cos(psi[gid]) * sin(gha) * sin(dec[gid]);
	Y[2] =  cos(psi[gid]) * cos(dec[gid]);	

	__syncthreads();
	
	double Fp;
	double Fc;


	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Fp += X[i]*R[i*3 + j]*X[j] - Y[i]*R[i*3 + j]*Y[j]; 
			Fc += Y[i]*R[i*3 + j]*X[j] + X[i]*R[i*3 + j]*Y[j]; 
		}
	}
	result[gid] = make_cuDoubleComplex(Fp, Fc);
}
""")

nsamps = 512

RA = np.linspace(0, 2*pi, nsamps).astype(np.float64)
DEC = np.linspace(0, 2*pi, nsamps).astype(np.float64)
PSI = np.linspace(0, 2*pi, nsamps).astype(np.float64)

tref = np.array([24715.581890875823 for item in RA]).astype(np.float64)

RA_gpu = gpuarray.to_gpu(RA)
DEC_gpu = gpuarray.to_gpu(DEC)
PSI_gpu = gpuarray.to_gpu(PSI)
tref_gpu = gpuarray.to_gpu(tref)

R = np.array([-0.3926141 , -0.07761341, -0.24738905 
              -0.07761341,  0.31952408,  0.22799784
              -0.24738905,  0.22799784,  0.07309003]).astype(np.float64)

R_gpu = gpuarray.to_gpu(R)
result = np.zeros(nsamps).astype(np.complex128)
result_gpu = gpuarray.to_gpu(result)

complex_antenna_factor = mod.get_function("complex_antenna_factor")

grd = (1,1,1)
blk = (512,1,1)
complex_antenna_factor(R_gpu, RA_gpu, DEC_gpu, PSI_gpu, tref_gpu, result_gpu, grid=grd, block=blk, shared=(8*6*512)) 
import pdb
pdb.set_trace()
