#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuComplex.h>
#include <cublas_v2.h>

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



void print_double_matrix(double matrix[], int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%lf \t", matrix[i*N + j]);
		}
		printf("\n");
	}
}

void print_double_vector(double vector[], int N) {
	for (int i = 0; i < N; i++) {
		printf("%f \t", vector[i]);
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


	double* A = (double *)malloc(ndim*ndim*sizeof(double)); 
	for (int i = 0; i < ndim; i++) {
		for (int j = 0; j < ndim; j++) {
			if (i == j) {
				A[ndim * i + j] = (double)(i+1); 	
			}
			else {
				A[ndim * i + j] = (double)0;
			}
		}
	}

	print_double_matrix(A, ndim);

	printf("Which we multiply into the vector... \n");

	double* vec = (double *)malloc(ndim*sizeof(double));

	for (int i = 0; i < ndim; i++) {
		vec[i] = 1.0;
	}

	print_double_vector(vec, ndim);

	// Allocate memory on the gpu

	printf("Copying memory down on to the cuda device...\n");

	double* A_gpu;
	double* vec_gpu;
	double* result = (double *)malloc(ndim * sizeof(double));
	double* result_gpu;

	if ( cudaSuccess != cudaMalloc((void**)&A_gpu, ndim*ndim*sizeof(double)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }
	if ( cudaSuccess != cudaMalloc((void**)&vec_gpu, ndim*sizeof(double)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }
	if ( cudaSuccess != cudaMalloc((void**)&result_gpu, ndim*sizeof(double)) ) {
                printf("cudaMalloc Failed...\n");
                exit(1);
        }

	cudaMemcpy(A_gpu, A, ndim*ndim*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vec_gpu, vec, ndim*sizeof(double), cudaMemcpyHostToDevice);


	// Initialize cublas

	cuDoubleComplex alf = make_cuDoubleComplex(1.0, 0.0);
	cuDoubleComplex bet = make_cuDoubleComplex(0.0, 0.0);
	cuDoubleComplex *alpha = &alf;
	cuDoubleComplex *beta  = &bet;
	int m = ndim;// Square Matrix
	int n = ndim;// Square Matrix

	cublasHandle_t handle;
	cublasCreate(&handle);	

	cublasDgemv(handle, CUBLAS_OP_N, m, n, alpha, A_gpu, ndim, vec_gpu, 1, beta, result_gpu, 1); 	


	cublasDestroy(handle);

	printf("Copying results up to host... \n");
	cudaMemcpy(result, result_gpu, ndim*sizeof(double), cudaMemcpyDeviceToHost);
	printf("Result is vector \n");
	print_double_vector(result, ndim);

	cudaFree(result_gpu);
	cudaFree(A);
	cudaFree(vec);
	free(result);
	free(vec);
	free(A);
	return 0;
}
