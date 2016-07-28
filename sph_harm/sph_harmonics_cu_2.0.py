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

#_____________________
#Instantiate mock data
#_____________________

nsamps = np.array(1024, ndmin=1).astype(np.int32)
theta  = np.linspace(0, 2*math.pi, nsamps)
phi    = np.linspace(0, 2*math.pi, nsamps)

# There will be a different function for
# each l value. The m values will be handed
# down to the gpu as a sorted list
selected_modes = [(2, -2), (2, 0), (2, 2)]
mlist_sort = sorted([mode[1] for mode in selected_modes]) 

# This cast is required to interface with the c kernel
mlist_sort = np.array(mlist_sort).astype(np.int32)

# Alloate memory and copy down
mlist_gpu = cuda.mem_alloc(mlist_sort.nbytes)

theta_gpu = cuda.mem_alloc(theta.nbytes)
phi_gpu   = cuda.mem_alloc(phi.nbytes)

cuda.memcpy_htod(mlist_gpu, mlist_sort)
cuda.memcpy_htod(theta_gpu, theta)
cuda.memcpy_htod(phi_gpu, phi)

# We will need the number of modes as an int 
# on the device for a loop within the kernel

nmodes     = np.array(len(mlist_sort), ndmin=1).astype(np.int32)
nmodes_gpu = cuda.mem_alloc(nmodes.nbytes)
cuda.memcpy_htod(nmodes_gpu, nmodes)

# As well as the total number of samples

nsamps_gpu = cuda.mem_alloc(nsamps.nbytes)
cuda.memcpy_htod(nsamps_gpu, nsamps)

# The final block of memory will contain the
# resulting complex numbers. We need a total 
# of nsamps*nmodes*sizeof(double)*2 bytes for
# the necessary double prec. complex numbers

results     = np.zeros(nmodes*nsamps).astype(np.complex128)
results_gpu = cuda.mem_alloc(theta.nbytes*len(mlist_sort) * 2)
mod=SourceModule('''
#include<math.h>
#include<cuComplex.h>
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

#_____________________
# Actual function
#_____________________

# Put the compiled c module in a python object
_get_spharms_l_eq_2 = mod.get_function("compute_sph_harmonics_l_eq_2")

def compute_spharms_l_eq_2(spharm_getter, th_gpu, ph_gpu, selected_modes_gpu, nmds_gpu, nsmps_gpu, rslt_gpu):
	nblocks = int(nsamps[0] / 1024)
	grd = (nblocks, 1, 1) # use 1D grid		
	blk = (1024,    1, 1) # maximum linear #threads
	##### KERNEL LAUNCH
	spharm_getter(th_gpu, ph_gpu, selected_modes_gpu, nmds_gpu, nsmps_gpu, rslt_gpu, block=blk, grid=grd)
	##### 	
	return rslt_gpu 


##### TIME EXECUTION
start = cuda.Event()
end   = cuda.Event()	

start.record() 
results_gpu = compute_spharms_l_eq_2(_get_spharms_l_eq_2, theta_gpu, phi_gpu, mlist_gpu, nmodes_gpu, nsamps_gpu, results_gpu)
end.record()
end.synchronize()
secs = start.time_till(end)*1e-3

print("Generated %d SWSH in %fs Seconds \n" % (nsamps, secs))

cuda.memcpy_dtoh(results, results_gpu)
print(results[0:9])





