#_ Preliminaries
import numpy as np
import pycuda
import pycuda.driver as cuda
# SourceModule is the thing that allows us to
# write cuda c modules 
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.cumath as cumath
import math
import pycuda.gpuarray as gpuarray
from pycuda.tools import dtype_to_ctype
from string import Template
# Cuda C
mod=SourceModule('''
#include<math.h>
#include<cuComplex.h>

__constant__ int nmodes[1];
__constant__ int nsamps[1];
__constant__ int ntimes[1];
__constant__ int nclmns[1];

__constant__ double det_tns[9];


__constant__ double CTU[25];
__constant__ double CTV[25];


__device__ int get_global_idx_1d_1d() {
        return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int get_xidx_within_row() {
        return threadIdx.x + blockIdx.x*blockDim.x;
}
__device__ int get_global_idx_2d_1d() {
        return threadIdx.x + gridDim.x*blockDim.x*blockIdx.y + blockIdx.x*blockDim.x;
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

__global__ void compute_sph_harmonics_l_eq_2(double *theta, double *phi, int *sel_modes, cuDoubleComplex *result) {   

        double _a2m2 = sqrt(5.0 / (64.0 * M_PI));
        double _a2m1 = sqrt(5.0 / (16.0 * M_PI));
        double _a20  = sqrt(15.0 /(32.0 * M_PI));
        double _a21  = sqrt(5.0 / (16.0 * M_PI));
        double _a22  = sqrt(5.0 / (64.0 * M_PI));
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (int modeIdx = 0; modeIdx < *nmodes; modeIdx++) {
               int m = sel_modes[modeIdx];
               if (sel_modes[modeIdx] == -2) {
                       double Re = _a2m2 * (1.0 - cos(theta[tid])) * (1.0 - cos(theta[tid])) * cos(m*phi[tid]);
                       double Im = _a2m2 * (1.0 - cos(theta[tid])) * (1.0 - cos(theta[tid])) * sin(m*phi[tid]);
                       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
                       result[tid] = Ylm;
                       tid += *nsamps; 
               }
        
               if (sel_modes[modeIdx] == -1) {
                       double Re = _a2m1 * sin(theta[tid]) * (1.0 - cos(theta[tid])) * cos(m*phi[tid]);
                       double Im = _a2m1 * sin(theta[tid]) * (1.0 - cos(theta[tid])) * sin(m*phi[tid]);
                       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
                       result[tid] = Ylm;
                       tid += *nsamps;
               }
               if (sel_modes[modeIdx] ==  0) {
                       double Re = _a20  * sin(theta[tid]) * sin(theta[tid]) * cos(m*phi[tid]);
                       double Im = _a20  * sin(theta[tid]) * sin(theta[tid]) * sin(m*phi[tid]);
                       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
                       result[tid] = Ylm;
                       tid += *nsamps;
               }
               if (sel_modes[modeIdx] ==  1) {
                       double Re = _a21  * sin(theta[tid]) * (1.0 + cos(theta[tid])) * cos(m*phi[tid]);
                       double Im = _a21  * sin(theta[tid]) * (1.0 + cos(theta[tid])) * sin(m*phi[tid]);
                       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
                       result[tid] = Ylm;
                       tid += *nsamps;
               }
               if (sel_modes[modeIdx] ==  2) {
                       double Re = _a22  * (1.0 + cos(theta[tid])) * (1.0 + cos(theta[tid])) * cos(m*phi[tid]);
                       double Im = _a22  * (1.0 + cos(theta[tid])) * (1.0 + cos(theta[tid])) * sin(m*phi[tid]);
                       cuDoubleComplex Ylm = make_cuDoubleComplex(Re, Im);
                       result[tid] = Ylm;
                       tid += *nsamps;
               }
       } 
}

__global__ void complex_antenna_factor(double *ra, double *dec, double *psi, double *tref, cuDoubleComplex *result) {
        
        extern __shared__ double Shr[]; 

        int gid = get_global_idx_1d_1d();       
        int id  = threadIdx.x;

        double *X = &Shr[(id * 6)];
        double *Y = &Shr[(id * 6) + 3];

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
                        Fp += X[i]*det_tns[i*3 + j]*X[j] - Y[i]*det_tns[i*3 + j]*Y[j]; 
                        Fc += Y[i]*det_tns[i*3 + j]*X[j] + X[i]*det_tns[i*3 + j]*Y[j]; 
                }
        }
        result[gid] = make_cuDoubleComplex(Fp, Fc);
}

__global__ void expand_rhoTS(cuDoubleComplex *rhots, cuDoubleComplex *rhots_all) {
        
        int gid = get_global_idx_2d_1d();
        int linidx = get_xidx_within_row();
        int which_mode = blockIdx.y % *nmodes;

        if (linidx < *ntimes) {
                rhots_all[gid] = rhots[*ntimes*which_mode + linidx];
        }       
        
}


__global__ void double_expand_rhoTS(double *rhots, double *rhots_all) {
        
        int gid = get_global_idx_2d_1d();
        int linidx = get_xidx_within_row();
        int which_mode = blockIdx.y % *nmodes;

        if (linidx < *ntimes) {
                rhots_all[gid] = rhots[*ntimes*which_mode + linidx];
        }       
        
}

__global__ void insert_ylms(cuDoubleComplex *padded_ts_all, cuDoubleComplex *ylms_all) {
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

__global__ void accordion(cuDoubleComplex *contr_arr) {
        int linidx = get_xidx_within_row();     
        int gid = get_global_idx_2d_1d();
        
	cuDoubleComplex *myrow;
	cuDoubleComplex *nxrow;

        for (int mode = 0; mode < *nmodes - 1; mode++) {
                myrow = &contr_arr[*nclmns*(*nmodes * blockIdx.y + mode)];              
                nxrow = &contr_arr[*nclmns*(*nmodes * blockIdx.y + mode + 1)];
                nxrow[linidx] = cuCadd(myrow[linidx], nxrow[linidx]);
        }                       
	myrow[linidx] = nxrow[linidx];
} 

__global__ void make_3x3_outer_prods(cuDoubleComplex *CT, cuDoubleComplex *all_V, cuDoubleComplex *out) {

        extern __shared__ cuDoubleComplex allshr[];

        int gid = get_global_idx_1d_1d();       
        int id  = threadIdx.x; 
                        
        cuDoubleComplex *myshr = &allshr[*nmodes*id];

        for (int i = 0; i < *nmodes; i++) {
                myshr[i] = all_V[(*nsamps * i) + id];   
        }

        out[gid] = cpx_outer_prod(CT, myshr); 

}


__global__ void bcast_vec_to_matrix(double *matrix, double *vector) { 
	__shared__ double myval[1]; 

	int gid = threadIdx.x + blockIdx.x*blockDim.x + *nmodes*gridDim.x*blockDim.x*blockIdx.y;
	int linidx = get_xidx_within_row();
	int vecidx = blockIdx.y;

	if (threadIdx.x == 0) {	
		myval[0] = vector[vecidx]; 
	}
	__syncthreads();

	if (linidx < *ntimes) {
		matrix[gid + (*nmodes-2)*gridDim.x*blockDim.x] += myval[0];  	
		matrix[gid + (*nmodes-1)*gridDim.x*blockDim.x] += myval[0];  	
	}  
}

__global__ void find_max_in_shrmem(double *all_rhots) {
        extern double __shared__ share[];

	int linidx = get_xidx_within_row();
	double *my_row = &all_rhots[*nclmns*(*nmodes*blockIdx.y + *nmodes - 2)]; 
	
        share[threadIdx.x] = my_row[linidx];

        __syncthreads();

        for (unsigned int s=blockDim.x/2; s > 0; s /= 2) {
                if (threadIdx.x < s) {
                        if (share[threadIdx.x] <= share[threadIdx.x + s]) {
                                share[threadIdx.x] = share[threadIdx.x + s];
                        }
                }       
                __syncthreads();
        }       

	if (threadIdx.x == 0) {
		my_row[blockIdx.x] = share[0];	
	} 
} 

__global__ void nv_reduc(double *indat) {
        extern __shared__ double shr[]; 

        // local and lobal thread ID
        unsigned int linidx = get_xidx_within_row();

        double *myrow = &indat[*ncols*(*nmodes*blockIdx.y + *nmodes - 1) ];    

        shr[threadIdx.x] = myrow[linidx];
        __syncthreads();

        for (unsigned int s=blockDim.x/2; s>0; s /= 2) {
                if (threadIdx.x < s) {
                        shr[threadIdx.x] = shr[threadIdx.x] + shr[threadIdx.x+s];
                }
                __syncthreads();
        }       
        if (threadIdx.x == 0) {
                myrow[blockIdx.x] = shr[0];
        }
}


''')


device = cuda.Device(0)

#_____________________
#     Constants 
#_____________________

max_tpb = 512             # Max threads per block
nsamps  = np.int32(1024) # Number of samples
ntimes  = np.int32(600)   # Length of rhoTS block

# Pad the times so they are an even multiple of max_tpb
# FIXME - This can waste a huge amount of memory 
nclmns  = np.int32( ntimes + max_tpb - (ntimes % max_tpb) ) 

selected_modes = [(2, -2), (2, 0), (2, 2)]
mlist_sort     = sorted([mode[1] for mode in selected_modes])
mlist_sort     = np.array(mlist_sort).astype(np.int32)

nmodes = np.int32(len(selected_modes)) # Number of modes

detector_tensor =np.array([[-0.3926141 , -0.07761341, -0.24738905], 
                           [-0.07761341,  0.31952408,  0.22799784],
                           [-0.24738905,  0.22799784,  0.07309003]]).astype(np.float64)
 

theta = np.linspace(0, 2*math.pi, nsamps).astype(np.float64)
phi   = np.linspace(0, 2*math.pi, nsamps).astype(np.float64)
psi   = np.linspace(0, 2*math.pi, nsamps).astype(np.float64)

tref = np.array([24715.581890875823 for item in theta]).astype(np.float64)

rhoTS = np.ones((nmodes, ntimes)).astype(np.complex128)
rhoTS[0,:] = np.arange(ntimes) + 1.0j*np.arange(ntimes)
rhoTS[1,:] = 2.0*np.arange(ntimes) + 1.0j*2.0*np.arange(ntimes)
rhoTS[2,:] = 3.0*np.arange(ntimes) + 1.0j*3.0*np.arange(ntimes)

#_____________________
#   Pass down data 
#_____________________


# **-- constants --**
nmodes_gpu = mod.get_global("nmodes")[0]
nsamps_gpu = mod.get_global("nsamps")[0]
ntimes_gpu = mod.get_global("ntimes")[0]
nclmns_gpu = mod.get_global("nclmns")[0]
detector_tensor_gpu = mod.get_global("det_tns")[0] 

cuda.memcpy_htod(nmodes_gpu, nmodes)
cuda.memcpy_htod(nsamps_gpu, nsamps)
cuda.memcpy_htod(ntimes_gpu, ntimes)
cuda.memcpy_htod(nclmns_gpu, nclmns)
cuda.memcpy_htod(detector_tensor_gpu, detector_tensor)



# **---- data -----**

selected_modes_gpu = gpuarray.to_gpu(mlist_sort)
theta_gpu = gpuarray.to_gpu(theta)
phi_gpu = gpuarray.to_gpu(phi)
psi_gpu = gpuarray.to_gpu(psi)

tref_gpu = gpuarray.to_gpu(tref)

rhoTS_gpu = gpuarray.to_gpu(rhoTS)
#_____________________
# Get Source Functions 
#_____________________

GPU_compute_sph_harmonics_l_eq_2 = mod.get_function("compute_sph_harmonics_l_eq_2")
GPU_complex_antenna_factor = mod.get_function("complex_antenna_factor")
GPU_expand_rhoTS = mod.get_function("expand_rhoTS")
GPU_insert_ylms = mod.get_function("insert_ylms")
GPU_accordion = mod.get_function("accordion")
GPU_make_3x3_outer_prods = mod.get_function("make_3x3_outer_prods")
GPU_bcast_vec_to_matrix = mod.get_function("bcast_vec_to_matrix")
GPU_find_max_in_shrmem = mod.get_function("find_max_in_shrmem")
GPU_nv_reduc = mod.get_function("nv_reduc")


######################################################################
# MAIN ROUTINE #######################################################
######################################################################



'''
Calculate the spherical harmonics 
'''
spharms_l_eq_2 = np.zeros(nsamps*nmodes).astype(np.complex128)
spharms_l_eq_2_gpu = gpuarray.to_gpu(spharms_l_eq_2) 
# One thread for each sample, 1D1D
nblocks = int(nsamps / max_tpb)
grd = (nblocks, 1, 1)
blk = (max_tpb, 1, 1) 

GPU_compute_sph_harmonics_l_eq_2(theta_gpu, phi_gpu, selected_modes_gpu, spharms_l_eq_2_gpu, grid=grd, block=blk)


'''
Generate the Fs and multiply them into the Ylms
'''
complex_antenna_factor = np.zeros(nsamps).astype(np.complex128)
caf_gpu = gpuarray.to_gpu(complex_antenna_factor)
# One thread for each sample, 1D1D

GPU_complex_antenna_factor(theta_gpu, phi_gpu, psi_gpu, tref_gpu, caf_gpu, grid=grd, block=blk, shared=(max_tpb*8*6))

# Multiply the F's in: sample-wise, same F for each Y
# Ylms are in row major order with rows corresponding 
# to modes

for i in range(0, nmodes):
	strt_mode = i*nsamps
	stop_mode = (i+1)*nsamps
	spharms_l_eq_2_gpu[strt_mode:stop_mode] *= caf_gpu
# Conjugate
spharms_l_eq_2_gpu = spharms_l_eq_2_gpu.conj()

'''
Build the likelihood function
'''

all_l_rhots = np.zeros((nsamps * nmodes, nclmns)).astype(np.complex128)
all_l_rhots_gpu = gpuarray.to_gpu(all_l_rhots)
# Blanket the array with threads

nblockx = int(nclmns / max_tpb)
nblocky = int(nsamps * nmodes)
grd = (nblockx, nblocky, 1)
blk = (max_tpb, 1,       1)
GPU_expand_rhoTS(rhoTS_gpu, all_l_rhots_gpu, grid=grd, block=blk)

'''
Multiply the F,Ylm products into the large timeseries block
'''

####################################################################################
# FOR UNIT TEST ONLY ###############################################################
####################################################################################

'''
 Ylms are stored in row major order ranging from lowest value of m to the highest

 |-----------(2,-2)-----------|----------(2,0)---------|-----------(2,2)---------| 

 [0+0j, 1+1j, 2+2j...1023+1023j, 1024+1024j...2047+2047j, 2048+2048j...3071+3017j]
 
 Assuming some strange case where they are the simple sequence of numbers above we 
 should expect the following results from multiplying them into the block of rhoTS

 all_l_rhots = | 0+0j, 1+1j, 2+2j...| <- this row should get multiplied by 0+0j 
               | 0+0j, 2+2j, 4+4j...| <- this row should get multiplied by 1024+1024j
               | 0+0j, 3+3j, 6+6j...| <- this row should get multiplied by 2048+2048j
               | 0+0j, 1+1j, 2+2j...| <- this row should get multiplied by 1+1j
               | 0+0j, 2+2j, 4+4j...| <- this row should get multiplied by 1025+1025j
               | 0+0j, 3+3j, 6+6j...| <- this row should get multiplied by 2049+2049j

 So we should end up with:

 all_l_rhots = | 0+0j, 0+0j,     0+0j     ...|
               | 0+0j, 0+4096j,  0+8192j  ...| 
               | 0+0j, 0+12288j, 0+24576j ...|
               | 0+0j, 0+2j,     0+4j     ...| 
               | 0+0j, 0+4100j,  0+8200j  ...|
               | 0+0j, 0+12294j, 0+24588j ...| 



''' 
##### UNCOMMENT FOR UNIT TEST
spharms_l_eq_2 = np.arange(nmodes*nsamps).astype(np.complex128) + np.arange(nmodes*nsamps).astype(np.complex128)*1.0j
spharms_l_eq_2_gpu = gpuarray.to_gpu(spharms_l_eq_2)


nblockx = int(nclmns / max_tpb)
nblocky = int(nsamps * nmodes)
# Blanket the array with threads 
grd = (nblockx, nblocky, 1)
blk = (max_tpb, 1,       1)

GPU_insert_ylms(all_l_rhots_gpu, spharms_l_eq_2_gpu, grid=grd, block=blk, shared=(int(nmodes*16)))
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

'''
Sum downwards to collect F-YLM-RHOTS row-wise sums
'''

####################################################################################
# FOR UNIT TEST ONLY ###############################################################
####################################################################################

'''
 We have the result as produced by the previous unit test, this function should
 sum downwards over nmodes rows before copying their destination row to the 2nd
 to last row as a final step. 

 At that point the information is all present in the final row so using the old
 rows as free memory shouldn't disturb anything. We should end up with:

 all_l_rhots = | 0+0j, 0+0j,     0+0j     ...|  
               | 0+0j, 0+16384j, 0+32768j ...|
               | 0+0j, 0+16384j, 0+32768j ...|
               | 0+0j, 0+2j,     0+4j     ...| 
               | 0+0j, 0+16396j, 0+32792j ...|
               | 0+0j, 0+16396j, 0+32792j ...|
'''


nblockx = int(nclmns / max_tpb)
nblocky = int(nsamps)
# One thread for each sample-time. One 1/nmodes of all gridpoints
# Each thread packages nmodes numbers into a sum that is the Lval 
grd = (nblockx, nblocky, 1)
blk = (max_tpb, 1,       1)

GPU_accordion(all_l_rhots_gpu, grid=grd, block=blk) 
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

'''
Generate and collect U and V terms
'''

'''
 Given that for the unit test the ylms are the linear array as described above, and the crossterms are all set
 to 2.0, we should have

 U_gpu = V_gpu = (0-0j)*2*(0+0j) + (1024-1024j)*2*(1024+1024j) + (2048-2048j)*2*(2048+2048j) = 20971520 
		 (1-1j)*2*(1+1j) + (1025-1025j)*2*(1025+1025j) + (2049-2049j)*2*(2049+2049j) = 20996108 
                 (2-2j)*2*(2+2j) + (1026-1026j)*2*(1026+1026j) + (2050-2050j)*2*(2050+2050j) = 21020720 
                 ...........................................................................
                 ...........................................................................
                 ...........................................................................

'''

U = np.zeros(nsamps).astype(np.complex128)
V = np.zeros(nsamps).astype(np.complex128)
U_gpu = gpuarray.to_gpu(U)
V_gpu = gpuarray.to_gpu(V)

# FIXME - these should exist dynamically within constant memory 

CTU = 2.0*np.eye(3).astype(np.complex128)
CTV = 2.0*np.eye(3).astype(np.complex128)
CTU_gpu = gpuarray.to_gpu(CTU)
CTV_gpu = gpuarray.to_gpu(CTV)

griddimx = int(nsamps / max_tpb)
# One thread per sample, each thread builds one U and V crossterm 

GPU_make_3x3_outer_prods(CTU_gpu, spharms_l_eq_2_gpu, U_gpu, grid=grd, block=blk, shared=int(16*nmodes*max_tpb))
GPU_make_3x3_outer_prods(CTV_gpu, spharms_l_eq_2_gpu, V_gpu, grid=grd, block=blk, shared=int(16*nmodes*max_tpb))
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

'''
 Given the U and V terms as calculated in the unit test above, we should be able to write down 
 the expected result of this calculation. There is no slicing, it should be simple broadcasts. 

 The only thing that we acutally need to modify to be something sensible is the antenna factor
 let's make it a linear array of complex numbers just like the spherical harmonics.

 then we should have:


 term_two = (0+0j)*(0-0j)*20971520 - RE((0+0j)*(0+0j)*20971520) = 0
 term_two = (1+1j)*(1-1j)*20996108 - RE((1+1j)*(1+1j)*20996108) = 41992216
 term_two = (2+2j)*(2-2j)*21020720 - RE((2+2j)*(2+2j)*21020720) = 168165760
 ..............................................................

'''

##### UNCOMMENT FOR UNIT TEST
caf = np.arange(nsamps).astype(np.complex128) + np.arange(nsamps).astype(np.complex128)*1.0j
caf_gpu = gpuarray.to_gpu(caf)

# This is the entire crossterm expression
term_two = (caf_gpu*caf_gpu.conj()*U_gpu - (caf_gpu*caf_gpu*V_gpu).real).real
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

# Take the real part
all_l_rhots_gpu = all_l_rhots_gpu.real 
'''
Subtract U and V terms from big block of rhoTS
'''


griddimx = int(nclmns / max_tpb) 
griddimy = int(nsamps) 
# One thread per sample-time
grd = (griddimx, griddimy, 1)
blk = (max_tpb,  1,        1)

GPU_bcast_vec_to_matrix(all_l_rhots_gpu, term_two, grid=grd, block=blk, shared=8)
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

'''
 Get the maxes of all the relevant rows - must do it twice because max is blockwise
'''

'''
 Since we have confirmed everything is working up to this point, let's reset the 
 expanded rhoTS to test to see if our maxfinding and summation functions work   
'''

##### UNCOMMENT FOR UNIT TEST
rhoTS = np.real(rhoTS).astype(np.float64) 
rhoTS[1,:] = rhoTS[2,:]
rhoTS_gpu = gpuarray.to_gpu(rhoTS)
all_l_rhots = np.real(all_l_rhots).astype(np.float64)
all_l_rhots_gpu = gpuarray.to_gpu(all_l_rhots)

GPU_double_expand_rhoTS = mod.get_function("double_expand_rhoTS")
nblockx = int(nclmns / max_tpb)
nblocky = int(nsamps * nmodes)
grd = (nblockx, nblocky, 1)
blk = (max_tpb, 1,       1)

GPU_double_expand_rhoTS(rhoTS_gpu, all_l_rhots_gpu, grid=grd, block=blk)

griddimx = int(nclmns / max_tpb)
griddimy = int(nsamps)
# One thread per sample-time
grd = (griddimx, griddimy, 1)
blk = (max_tpb,  1,        1)

#####

GPU_find_max_in_shrmem(all_l_rhots_gpu, grid=grd, block=blk, shared=int(max_tpb*8))

griddimy = int(nsamps)
blokdimx = griddimx # Only need as many threads as we had blocks in x dimension

grd = (1, griddimy, 1) 
blk = (blokdimx, 1, 1)

GPU_find_max_in_shrmem(all_l_rhots_gpu, grid=grd, block=blk, shared=int(max_tpb*8))

# Indexes are not contiguous

griddimx = int(nclmns / max_tpb) 
griddimy = int(nsamps) 
# One thread per sample-time
grd = (griddimx, griddimy, 1)
blk = (max_tpb,  1,        1)

maxes = np.array(all_l_rhots_gpu[:,0][1::nmodes].get()).astype(np.float64)
maxes_gpu = gpuarray.to_gpu(maxes)

GPU_bcast_vec_to_matrix(all_l_rhots_gpu, -maxes_gpu, grid=grd, block=blk, shared=8)
# ***** THIS IS CORRECT AND WORKING UP THROUGH HERE AS OF AUGUST 10TH 2016 ***** 

''' 
 Marginalize over Time
'''

all_l_rhots_gpu = cumath.exp(all_l_rhots_gpu) # exponentiate 

GPU_nv_reduc(all_l_rhots_gpu) # sum over time 

lnL_gpu = maxes_gpu + cumath.log(all_l_rhots_gpu) # TIMES DELTA T FIXME




