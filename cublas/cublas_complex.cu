#include <stdlib.h>
#include <math.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
	}
}

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

//__device__ static cuDoubleComplex alpha_gpu;  
//__device__ static cuDoubleComplex beta_gpu; 


struct ZGEMVScalarParams {
    cuDoubleComplex alpha, beta;
};

void call_zgemv(cublasHandle_t handle, int m, int n, cuDoubleComplex alpha_gpu, cuDoubleComplex *A_gpu, int ndim, cuDoubleComplex* vec_gpu, cuDoubleComplex beta_gpu, cuDoubleComplex* result_gpu) {
        cublasZgemv(handle, CUBLAS_OP_N, m, n, &alpha_gpu, A_gpu, ndim, vec_gpu, 1, &beta_gpu, result_gpu, 1);
}

__global__ void call_zgemv_gpu(int m, int n, cuDoubleComplex* alpha_gpu, cuDoubleComplex *A_gpu, int ndim, cuDoubleComplex* vec_gpu, cuDoubleComplex* beta_gpu, cuDoubleComplex* result_gpu) {

	cublasHandle_t handle; 
	cublasCreate(&handle);
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid == 0) {
	        cublasZgemv(handle, CUBLAS_OP_N, m, n, alpha_gpu, A_gpu, ndim, vec_gpu, 1, beta_gpu, result_gpu, 1);
		cudaDeviceSynchronize();
	}
	cublasDestroy(handle);
}




void print_double_complex_matrix(cuDoubleComplex matrix[], int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f + %f j  \t", cuCreal(matrix[i*N + j]), cuCimag(matrix[i*N + j]));
		}
		printf("\n");
	}
}

void print_double_complex_vector(cuDoubleComplex vector[], int N) {
	for (int i = 0; i < N; i++) {
		printf("%f + %f j \t", cuCreal(vector[i]), cuCimag(vector[i]));
	}
	printf("\n");
}




int main(int argc, char* argv[]) {

        // Initial Machinery to select the GPU

        cudaDeviceProp prop; // Blank
        memset(&prop, 0, sizeof(cudaDeviceProp)); // Set struct to all 0

        int devcount;
        gpuErrchk(cudaGetDeviceCount(&devcount));
        printf("Devcount %d! \n", devcount);

        // Retrieve GPU properties - Look for a K20 or above    
        int dev;
        prop.multiProcessorCount = 13;
        cudaChooseDevice(&dev, &prop);
        cudaGetDeviceProperties(&prop, dev);


        printf(" *** DEVICE INFORMATION IS AS FOLLOWS *** \n");
        printDevProp(prop);
        printf(" *** END DEVICE INFORMATION  *** \n");





	
	// Instantiate mock data

	if (argc < 2) {
		printf("Please enter a value for ndim. \n");
		exit(1);
	}
	if (argc > 2) {
		printf("Please enter ONLY one value: ndim \n");
		exit(1);
	}

	int ndim = atoi(argv[1]);
	printf("Testing cuBLAS for a %d dimensional matrix... \n", ndim);


	cuDoubleComplex* A = (cuDoubleComplex *)malloc(ndim*ndim*sizeof(cuDoubleComplex)); 
	for (int i = 0; i < ndim; i++) {
		for (int j = 0; j < ndim; j++) {
			if (i == j) {
				A[ndim * i + j] = make_cuDoubleComplex(i, i); 	
			}
			else {
				A[ndim * i + j] = make_cuDoubleComplex(0, 0);
			}
		}
	}

	print_double_complex_matrix(A, ndim);

	printf("Which we multiply into the vector... \n");

	cuDoubleComplex* vec = (cuDoubleComplex *)malloc(ndim*sizeof(cuDoubleComplex));

	for (int i = 0; i < ndim; i++) {
		vec[i] = make_cuDoubleComplex(1.0, 1.0);
	}

	print_double_complex_vector(vec, ndim);

	// Allocate memory on the gpu

	printf("Copying memory down on to the cuda device...\n");

	cuDoubleComplex* A_gpu;
	cuDoubleComplex* vec_gpu;

	int nblas = 1;

	// CAN WE CALL CUBLAS ALL ON THE SAME MATRIX? TRY CALLING IT 10 TIMES	

	cuDoubleComplex* result = (cuDoubleComplex *)malloc(nblas*ndim * sizeof(cuDoubleComplex));
	cuDoubleComplex* result_gpu;

	if ( cudaSuccess != cudaMalloc((void**)&A_gpu, ndim*ndim*sizeof(cuDoubleComplex)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }
	if ( cudaSuccess != cudaMalloc((void**)&vec_gpu, ndim*sizeof(cuDoubleComplex)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }
	// WE NOW NEED 10X THE MEMORY ALLOCATED FOR RESULTS
	if ( cudaSuccess != cudaMalloc((void**)&result_gpu, nblas*ndim*sizeof(cuDoubleComplex)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }

	cudaMemcpy(A_gpu, A, ndim*ndim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(vec_gpu, vec, ndim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);


	// Initialize cublas

	int m = ndim;// Square Matrix
	int n = ndim;// Square Matrix
	dim3 grid = dim3(1, 1, 1);
	dim3 block = dim3(nblas, 1, 1);


//	cublasHandle_t handle; 
//	cublasCreate(&handle);

	ZGEMVScalarParams h_scalar_params = {make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)}; 
	ZGEMVScalarParams *d_scalar_params;
	

	if (cudaSuccess != cudaMalloc((void **)&d_scalar_params, sizeof(ZGEMVScalarParams)) ) {    
		printf("CudaMalloc Failed... \n");
		exit(1);
   	}

	if (cudaSuccess != cudaMemcpy(d_scalar_params, &h_scalar_params, sizeof(ZGEMVScalarParams), cudaMemcpyHostToDevice) ) {
        	printf("!!!! host to device memory copy error\n");
        	exit(1);
    	}



//      cublasZgemv(handle, CUBLAS_OP_N, m, n, &alpha_gpu, A_gpu, ndim, vec_gpu, 1, &beta_gpu, result_gpu, 1);

        call_zgemv_gpu<<<1,1>>>(m, n, &d_scalar_params->alpha, A_gpu, ndim, vec_gpu, &d_scalar_params->beta, result_gpu);

//	call_zgemv(handle, m, n, alpha_gpu, A_gpu, ndim, vec_gpu, beta_gpu, result_gpu);



	printf("Copying results up to host... \n");
	cudaMemcpy(result, result_gpu, nblas*ndim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	printf("Result is vector \n");
	print_double_complex_vector(result, ndim);
//	print_double_complex_vector(&result[ndim], ndim);
	cudaFree(result_gpu);
	cudaFree(A);
	cudaFree(vec);
	cudaFree(d_scalar_params);
	free(result);
	free(vec);
	free(A);

	
	return 0;
}
