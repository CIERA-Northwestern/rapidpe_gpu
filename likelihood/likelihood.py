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
import pycuda.gpuarray as gpuarray

device = cuda.Device(0)

#_____________________
#Instantiate mock data
#_____________________


max_tpb = 512

nsamps = np.array(10240, ndmin=1).astype(np.int32)
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

spharms_l_eq_2     = np.zeros(nmodes*nsamps).astype(np.complex128)
spharms_l_eq_2_gpu = cuda.mem_alloc(theta.nbytes*len(mlist_sort) * 2)

# Length of the time series

ntimes = np.array(600, ndmin=1).astype(np.int32) 
ntimes_gpu = cuda.mem_alloc(ntimes.nbytes)
cuda.memcpy_htod(ntimes_gpu, ntimes)

# zero padwidth 

padwidth = max_tpb - (ntimes[0] % max_tpb)
padwidth = np.array(padwidth, ndmin=1).astype(np.int32)
padwidth_gpu = cuda.mem_alloc(padwidth.nbytes) 
cuda.memcpy_htod(padwidth_gpu, padwidth)

# FIXME - let's use an ncols variable for now

ncols = ntimes + padwidth
ncols = np.array(ncols, ndmin=1).astype(np.int32)
ncols_gpu = cuda.mem_alloc(ncols.nbytes)
cuda.memcpy_htod(ncols_gpu, ncols)


#FIXME FIXME FIXME Put the spharms in constant memory
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
__device__ int get_padded_idx_2d_1d(int ntimes, int ncols) {
        return threadIdx.x + blockIdx.y*ncols + ntimes; 
}


// ROW MAJOR FORM
__device__ int get_global_idx_2d_1d() {
	return threadIdx.x + gridDim.x*blockDim.x*blockIdx.y + blockIdx.x*blockDim.x;
}

__device__ int get_xidx_within_row() {
	return threadIdx.x + blockIdx.x*blockDim.x;
}


__global__ void pad_with_zeros(cuDoubleComplex *arr_to_pad, int *ntimes, int *nsamps, int *ncols, int *padwidth) {
        
        int gid = get_padded_idx_2d_1d(*ntimes, *ncols);
        arr_to_pad[gid] = make_cuDoubleComplex(0.0, 0.0);       
}

__global__ void insert_ylms(cuDoubleComplex *padded_ts_all, cuDoubleComplex *ylms_all, int *nmodes, int *ncols, int *nsamps) {
        extern __shared__ cuDoubleComplex shr[]; // My Ylms will reside here     
        if (threadIdx.x < *nmodes) {
		int offset = (blockIdx.y - (blockIdx.y % *nmodes)) / *nmodes;
		shr[threadIdx.x] = ylms_all[*nsamps*threadIdx.x + offset];
        }
       
	__syncthreads();
 
        int gid = get_global_idx_2d_1d() ;      
	
	cuDoubleComplex myins_val = cuCmul(padded_ts_all[gid], shr[blockIdx.y % *nmodes]); 

        padded_ts_all[gid] = myins_val; 

} 

__global__ void nv_reduc(cuDoubleComplex *indat, int *ncols, int *nmodes) {
        extern __shared__ cuDoubleComplex shr[]; 

        // local and lobal thread ID
        unsigned int linidx = get_xidx_within_row();

	cuDoubleComplex *myrow = &indat[*ncols*(*nmodes*blockIdx.y + *nmodes - 1) ];  	

        shr[threadIdx.x] = myrow[linidx];
        __syncthreads();

        for (unsigned int s=blockDim.x/2; s>0; s /= 2) {
                if (threadIdx.x < s) {
			shr[threadIdx.x] = cuCadd(shr[threadIdx.x], shr[threadIdx.x+s]);
                }
                __syncthreads();
        }       
        if (threadIdx.x == 0) {
                myrow[blockIdx.x] = shr[0];
        }


}
__global__ void expand_rhoTS(int *ncols, int *nmodes, int *ntimes, cuDoubleComplex *rhots, cuDoubleComplex *rhots_all) {
	
	int gid = get_global_idx_2d_1d();
	int linidx = get_xidx_within_row();
	int which_mode = blockIdx.y % *nmodes;

	if (linidx < *ntimes) {
		rhots_all[gid] = rhots[*ntimes*which_mode + linidx];
	}	
	
}

__global__ void accordion(cuDoubleComplex *contr_arr, int *nmodes, int *ntimes, int *ncols) {
	int linidx = get_xidx_within_row();	
	int gid = get_global_idx_2d_1d();
	
	for (int mode = 0; mode < *nmodes - 1; mode++) {
		cuDoubleComplex *myrow = &contr_arr[*ncols*(*nmodes * blockIdx.y + mode)];		
		cuDoubleComplex *nxrow = &contr_arr[*ncols*(*nmodes * blockIdx.y + mode + 1)];
		nxrow[linidx] = cuCadd(myrow[linidx], nxrow[linidx]);
	}			
} 


''')





#_____________________
# Retrieve Ylms
#_____________________

# Put the compiled c module in a python object
_get_spharms_l_eq_2 = mod.get_function("compute_sph_harmonics_l_eq_2")

def compute_spharms_l_eq_2(spharm_getter, th_gpu, ph_gpu, selected_modes_gpu, nmds_gpu, nsmps_gpu, rslt_gpu):
	nblocks = int(nsamps[0] / max_tpb)
	grd = (nblocks, 1, 1) # use 1D grid		
	blk = (max_tpb, 1, 1) # maximum linear #threads
	##### KERNEL LAUNCH
	
	#empty = mod.get_function("empty")
	#empty(grid=grd, block=blk)

	spharm_getter(th_gpu, ph_gpu, selected_modes_gpu, nmds_gpu, nsmps_gpu, rslt_gpu, block=blk, grid=grd)
	##### 	
	return rslt_gpu 


#spharms_l_eq_2_gpu = compute_spharms_l_eq_2(_get_spharms_l_eq_2, theta_gpu, phi_gpu, mlist_gpu, nmodes_gpu, nsamps_gpu, spharms_l_eq_2_gpu)

#_____________________
# FIXME -For Debugging 
#_____________________


#cuda.memcpy_dtoh(spharms_l_eq_2, spharms_l_eq_2_gpu)
print(spharms_l_eq_2[0:9])

'''
 As of yet we have an (nmodes x nsamps) array of double
 complex numbers in memory 

 ***---------------***
 
 ***YLMS REFERFENCE***
     (for 3 modes) 

 SAMPLES        MEMORY 

 1024           49.15 KB  
 10240          491.5 KB
 102400         4.915 MB
 1024000        49.15 MB
 10240000       491.5 MB
 102400000      4.915 GB

 ***---------------***

'''


#_____________________
# build likelihood 
#_____________________

# For Folding, this array must be
# nsamps * ncols. Must be careful 
# to arrange correct axis as ncols

pre_likelihood     = np.zeros((nsamps*nmodes, ncols)).astype(np.complex128)
pre_likelihood_gpu = gpuarray.to_gpu(pre_likelihood)

'''
 ***---------------***
 
 ***LKELY Reference***
     (for 3 modes)
     (ntimes=2500)

 Note - is absurdly 
 memory inefficient

 SAMPLES	MEMORY
 1024		122.9 MB 
 10240		1.229 GB
 102400 <- Not realistic
 1024000	////////
 10240000       ////////
 102400000      ////////

 This indicates we will
 need sample blocks for 
 about (2-3)e3 nsamples	
 -we can play with this
'''	

#pre_likelihood_gpu = cuda.mem_alloc(pre_likelihood.nbytes)
#cuda.memcpy_htod(pre_likelihood_gpu, pre_likelihood)

#pre_likelihood_gpu = gpuarray.to_gpu(pre_likelihood) 

#_____________________
# build rhoTS 
#_____________________


# This is honestly a negligable amount of total memory
rhoTS = np.zeros((nmodes, ntimes)).astype(np.complex128)
rhoTS[0,:] += np.ones(ntimes) 
rhoTS[1,:] += np.ones(ntimes) 
rhoTS[2,:] += np.ones(ntimes) 


print(rhoTS[:, 0:10])

rhoTS_gpu = cuda.mem_alloc(rhoTS.nbytes)
cuda.memcpy_htod(rhoTS_gpu, rhoTS)

_get_rhoTS_expander   = mod.get_function("expand_rhoTS")


# We might not need this thanks to GPUarrays
_get_pad_with_zeros = mod.get_function("pad_with_zeros")


# Above two functions naturally group together
# FIXME - should probably be combined into one
def expand_rhoTS(rhoTS_expander, zero_padder, ncols_gpu, nmodes_gpu, ntimes_gpu, rhoTS_gpu, expansion_arr):
	griddimx  = int(ncols[0] / max_tpb)	
	griddimy  = int(nsamps[0]*nmodes[0])

	grd = (griddimx, griddimy, 1) 
	blk = (max_tpb,  1,        1) 

	rhoTS_expander(ncols_gpu, nmodes_gpu, ntimes_gpu, rhoTS_gpu, expansion_arr, grid=grd, block=blk)

	# I don't think this is needed anymore
	#zero_padder(pre_likelihood_gpu, ntimes_gpu, nsamps_gpu, ncols_gpu, padwidth_gpu, grid=(1, griddimy, 1), block=blk)

	return expansion_arr


rhoTS_expansion_gpu = expand_rhoTS(_get_rhoTS_expander, _get_pad_with_zeros, ncols_gpu, nmodes_gpu, ntimes_gpu, rhoTS_gpu, pre_likelihood_gpu) 

#_____________________
# Multiply in Ylms 
#_____________________

'''
 The rows in the expanded rhoTS are harmonic-mode time series 
 from minimum m value to maximum

         <- times ->
         ___________>    _
(2,-2)  |...........      |
(2, 0)  |...........      |- sample 0
(2, 2)  |...........     _|
(2,-2)  |...........      |
(2, 0)  |...........      |- sample 1
(2, 2)  |...........     _|
   .    v	          v	
   .
   .


 However the Ylms array has one row per mode and one column 
 for each sample. The sample-mode Ylms need to be multiplied in
 to the sample-mode-timeseries before we contract th rhoTS to
 create the set of "Term 1's" for the whole sample set
'''

_multiply_in_ylms = mod.get_function("insert_ylms") 

def ylms_into_rhoTS(ylm_multiplier, expanded_rhoTS, all_ylms, nmodes_gpu, ncols_gpu, nsamps_gpu):
	griddimx = int(ncols[0] / max_tpb)
	griddimy = int(nsamps[0]*nmodes[0])

	# We want to cover the whole expanded rhoTS in threads again	
	grd = (griddimx, griddimy, 1)
	blk = (max_tpb,  1,        1)	
	ylm_multiplier(expanded_rhoTS, all_ylms, nmodes_gpu, ncols_gpu, nsamps_gpu, grid=grd, block=blk, shared=(int(nmodes[0] *16)))

	return expanded_rhoTS

#_____________________
# Expanded Ylm Test 
#_____________________

spharms_l_eq_2 = np.zeros((nmodes, nsamps)).astype(np.complex128)
for i in range(nmodes):
	spharms_l_eq_2[i,:] = np.arange(nsamps)


spharms_l_eq_2_gpu = gpuarray.to_gpu(spharms_l_eq_2)

all_rhots_ylm = ylms_into_rhoTS(_multiply_in_ylms, rhoTS_expansion_gpu, spharms_l_eq_2_gpu, nmodes_gpu, ncols_gpu, nsamps_gpu)


'''
 The rhoTS have been multiplied by their corresponding Ylms
 Now we need summing over the timeseries-ylms and producing
 the large (first likelihood term). It needs to be folded 
 up as follows:

         <- times ->
         ___________>   
(2,-2)  |........... 
          | | | | |  \     
          v v v v v   |
(2, 0)  |...........   -   SAMPLE 1
          | | | | |   |    
          v v v v v  /  
(2, 2)  |...........    


(2,-2)  |........... 
          | | | | |  \     
          v v v v v   |
(2, 0)  |...........   -   SAMPLE 2
          | | | | |   |  
          v v v v v  /
(2, 2)  |...........    
   .         .	          .	
   .         .            .
    

 We launch a row of blocks for every sample that is long 
 enough to assign a thread on each element of the matrix
 the threads then sum downwards into the row below a total
 of nmodes times per thread

 This ensures that there are no memory conflics or race conditions
 each thread only ever reads or writes to memory locations 
 that are completely unique to that thread.

'''
accordion = mod.get_function("accordion")
accordion(all_rhots_ylm, nmodes_gpu, ntimes_gpu, ncols_gpu, grid=((int(ncols[0] / max_tpb)), int(nsamps[0]), 1), block=(max_tpb, 1, 1))


'''
 To get the likelihoods for all the samples, we will need to fold 
 the block of time series up along the sample axis with reduction
'''

def reduce(reducer, rhoTS_expansion_gpu, ncols_gpu, nmodes_gpu): 	

	griddimx = int(ncols[0] / max_tpb)
	griddimy = int(nsamps[0])

	grd =  (griddimx, griddimy, 1)
	blk  = (max_tpb,  1,        1)		
	
	#FIXME - Doesn't support rhoTS longer then 1024**2 = 1048576 elements 
	reducer(rhoTS_expansion_gpu, ncols_gpu, nmodes_gpu, grid=grd, block=blk, shared=16*max_tpb)		
	# Run again to sum the block sums
	grd = (1, griddimy, 1)
	blk = (griddimx, 1, 1) # Need a thread for each row block from first call
	reducer(rhoTS_expansion_gpu, ncols_gpu, nmodes_gpu, grid=grd, block=blk, shared=16*griddimx)

	

_reducer = mod.get_function("nv_reduc")
reduce(_reducer, all_rhots_ylm, ncols_gpu, nmodes_gpu)

import pdb
pdb.set_trace()
x = 2


