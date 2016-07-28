import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import math

mod = SourceModule('''
#include <math.h>
#include <cuComplex.h>
__global__ void compute_sph_harmonics_l_eq_2(double *theta, double *phi, int *sel_modes, int *nmodes, int *nsamps, cuDoubleComplex *result) {	

	int nModes = (int)nmodes[0];	
	int nSamps = (int)nsamps[0];

	double _a2m2 = sqrt(5.0 / (64.0 * M_PI));
	double _a2m1 = sqrt(5.0 / (16.0 * M_PI));
	double _a20  = sqrt(15.0 /(32.0 * M_PI));
	double _a21  = sqrt(5.0 / (16.0 * M_PI));
	double _a22  = sqrt(5.0 / (64.0 * M_PI));
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int modeIdx = 0; modeIdx < nModes; modeIdx++) {
		int m = sel_modes[modeIdx];
		if (sel_modes[modeIdx] == -2) {
			double Re = _a2m2 * (1.0 - cos(theta[tid])) * (1.0 - cos(theta[tid])) * cos(m*phi[tid]);
			double Im = _a2m2 * (1.0 - cos(theta[tid])) * (1.0 - cos(theta[tid])) * sin(m*phi[tid]);
			cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			result[tid] = Ylm;
			tid += nSamps; 
		}
		if (sel_modes[modeIdx] == -1) {
			double Re = _a2m1 * sin(theta[tid]) * (1.0 - cos(theta[tid])) * cos(m*phi[tid]);
			double Im = _a2m1 * sin(theta[tid]) * (1.0 - cos(theta[tid])) * sin(m*phi[tid]);
			cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			result[tid] = Ylm;
			tid += nSamps;
		}
		if (sel_modes[modeIdx] ==  0) {
			double Re = _a20  * sin(theta[tid]) * sin(theta[tid]) * cos(m*phi[tid]);
			double Im = _a20  * sin(theta[tid]) * sin(theta[tid]) * sin(m*phi[tid]);
			cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			result[tid] = Ylm;
			tid += nSamps;
		}
		if (sel_modes[modeIdx] ==  1) {
			double Re = _a21  * sin(theta[tid]) * (1.0 + cos(theta[tid])) * cos(m*phi[tid]);
			double Im = _a21  * sin(theta[tid]) * (1.0 + cos(theta[tid])) * sin(m*phi[tid]);
			cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			result[tid] = Ylm;
			tid += nSamps;
		}
		if (sel_modes[modeIdx] ==  2) {
			double Re = _a22  * (1.0 + cos(theta[tid])) * (1.0 + cos(theta[tid])) * cos(m*phi[tid]);
			double Im = _a22  * (1.0 + cos(theta[tid])) * (1.0 + cos(theta[tid])) * sin(m*phi[tid]);
			cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
			result[tid] = Ylm;
			tid += nSamps;
		}
	} 
}
''')

N = 1024
theta = np.linspace(0,2*math.pi,N).astype(np.float64)
phi   = np.linspace(0,2*math.pi,N).astype(np.float64)

theta_m = theta

theta_gpu = cuda.mem_alloc(theta.nbytes)
phi_gpu   = cuda.mem_alloc(phi.nbytes)

cuda.memcpy_htod(theta_gpu, theta)
cuda.memcpy_htod(phi_gpu, phi)

selected_modes = [(2,-2), (2,0), (2,2)]

nsamps = np.int32(1024)
#nmodes = np.int32(len(

result_gpu = cuda.mem_alloc(theta_m.nbytes * len(selected_modes) * 2)	

modelist = np.array(sorted([mode[1] for mode in selected_modes])).astype(np.int32)
modelist_gpu = cuda.mem_alloc(modelist.nbytes)
cuda.memcpy_htod(modelist_gpu, modelist)

def get_spharms_l_eq_2(theta, phi, selected_Modes_gpu, rslt_gpu):
	modelist = np.array(sorted([mode[1] for mode in selected_modes])).astype(np.int32)


	modelist_gpu = cuda.mem_alloc(modelist.nbytes)

#	nsampslen = np.array(len(theta), ndmin=1).astype(np.int32)
	nmodeslen = np.array(len(modelist), ndmin=1).astype(np.int32)
	nsamps_gpu = cuda.mem_alloc(nsamps.nbytes)
	nmodes_gpu = cuda.mem_alloc(nmodeslen.nbytes) 	
	
	cuda.memcpy_htod(nsamps_gpu, nsamps)
	cuda.memcpy_htod(nmodes_gpu, nmodeslen)

#	cuda.memcpy_htod(theta_gpu, theta)
#	cuda.memcpy_htod(phi_gpu, phi)
	cuda.memcpy_htod(modelist_gpu, modelist)


	# Get and compile the cuda function 
	sph = mod.get_function("compute_sph_harmonics_l_eq_2")
	result_gpu = cuda.mem_alloc(theta_m.nbytes * len(modelist) * 2)	
	blk  = (1024,1,1)
	grd = (1,1,1) 
	sph(theta, phi, modelist_gpu, nmodes_gpu, nsamps_gpu, rslt_gpu, block=blk, grid=grd)	

#	cuda.memcpy_dtoh(result, result_gpu)
#	print(result[0:9])
#	print(len(result))
	return	

get_spharms_l_eq_2(theta_gpu, phi_gpu, modelist_gpu, result_gpu)



