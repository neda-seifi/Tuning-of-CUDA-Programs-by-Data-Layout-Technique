#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_profiler_api.h"

__global__ void CMM_CUDA_kernel(const int *dim, int *m, int *result, int n)
{
	int number_of_threads = n, i, min, j, k, index;
	index = threadIdx.x;
	i = index + 1;
	j = index + 1;
	m[(n + 1) * i + i] = 0;
	__syncthreads();

	while (number_of_threads >= 1){
		number_of_threads = number_of_threads - 1;
		if (index < number_of_threads){
			j = j + 1;
			min = INT_MAX;
			for (k = i; k <= j - 1; k++)
			if (m[(n + 1) * i + k] + m[(n + 1) * (k + 1) + j] + dim[i - 1] * dim[k] * dim[j] < min)
				min = m[(n + 1) * i + k] + m[(n + 1) * (k + 1) + j] + dim[i - 1] * dim[k] * dim[j];
			m[(n + 1) * i + j] = min;
		}
		__syncthreads();
	}
	if (index == 0) result[0] = m[(n + 1) * 1 + n];
}

__global__ void CMM_CUDA_with_Data_Layout_kernel(const int *dim, int *m, int *result, int n)
{
	int number_of_threads = n, i, min, j, k, index; int xprim, yprim, d;
	index = threadIdx.x;
	i = index + 1;
	j = index + 1;
	m[n + 1 + i] = 0;         //---- m[i,i]

	__syncthreads();

	while (number_of_threads >= 1){
		number_of_threads = number_of_threads - 1;
		if (index < number_of_threads){
			j = j + 1;
			min = INT_MAX;
			for (k = i; k <= j - 1; k++){
				d = k - i;  xprim = n + 1 + d*n - (d*(d - 1) / 2) + i;                      //---- m[i,k]
				d = j - k - 1;  yprim = n + 1 + d*n - (d*(d - 1) / 2) + k + 1;    //---- m[k+1,j]
				if (m[xprim] + m[yprim] + dim[i - 1] * dim[k] * dim[j] < min)
					min = m[xprim] + m[yprim] + dim[i - 1] * dim[k] * dim[j];
			}
			d = j - i;  xprim = n + 1 + d*n - (d*(d - 1) / 2) + i;  //---- m[i,j]
			m[xprim] = min;
			//printf("m[%d]=%d\n", xprim,m[xprim]);
		}
		__syncthreads();
	}
	if (index == 0) {
		xprim = (n + 1) * (n + 2) / 2;  //---- m[1,n]
		result[0] = m[xprim];
	}
}

int CMM_Serial(int*, int, float*);
int CMM_CUDA(int*, int, float*);
int CMM_CUDA_with_Data_Layout(int*, int, float*);

int main()
{
	FILE *out = fopen("output.txt", "w");
	int i, n, *dim;	cudaError_t cudaStatus;
	float time1 = 0, time2 = 0, time3 = 0;
	int result1 = 0, result2 = 0, result3 = 0;

	for (n = 10; n <= 1024; n++){

		dim = new int[n + 1];
		for (i = 0; i <= n; i++) dim[i] = (int)rand() / (RAND_MAX + 1) * (10 - 2) + 2;

		result1 = CMM_Serial(dim, n, &time1);
		result2 = CMM_CUDA(dim, n, &time2);
		result3 = CMM_CUDA_with_Data_Layout(dim, n, &time3);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess){ printf("for n=%d , cudaDeviceReset failed!", n);	break; }

		if (result1 != result2 || result1 != result3){ printf("for n=%d , results are wrong!\n", n); break; }

		printf("n = %d   ,time1 = %f   ,time2 = %f  ,time3 = %f \n", n, time1, time2, time3);
		fprintf(out, "%d %f %f %f\n", n, time1, time2, time3);


		delete[] dim;

	}
	fclose(out);
	return 0;
}

int CMM_Serial(int* dim, int n, float *time)
{
	float elapsed_time;
	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

	int i, j, k, diagonal, min, temp;
	int** m = new int*[n + 1];
	for (i = 0; i <= n; i++) m[i] = new int[n + 1];

	cudaEventRecord(start, 0);

	for (i = 0; i <= n; i++) m[i][i] = 0;
	for (diagonal = 1; diagonal < n; diagonal++){
		for (i = 1; i <= n - diagonal; i++){
			j = i + diagonal;
			min = INT_MAX;
			for (k = i; k < j; k++){
				temp = m[i][k] + m[k + 1][j] + dim[i - 1] * dim[k] * dim[j];
				if (temp < min) min = temp;
			}
			m[i][j] = min;
		}
	}
	temp = m[1][n];

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	*time = elapsed_time;


	delete[] m;

	return temp;
}

int CMM_CUDA(int* dim, int n, float *time)
{
	float elapsed_time;
	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

	int i, *dev_m, *dev_dim, *dev_result, result[1];

	cudaMalloc((void**)&dev_result, 1 * sizeof(int));
	cudaMalloc((void**)&dev_dim, (n + 1) * sizeof(int));
	cudaMalloc((void**)&dev_m, (n + 1) * (n + 1) * sizeof(int));
	cudaMemcpy(dev_dim, dim, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	for (i = 1; i <= 10; i++) CMM_CUDA_kernel << <1, n >> >(dev_dim, dev_m, dev_result, n);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	*time = elapsed_time / 10.0;


	cudaMemcpy(result, dev_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_result); cudaFree(dev_dim); cudaFree(dev_m);

	return result[0];
}

int CMM_CUDA_with_Data_Layout(int* dim, int n, float *time)
{
	float elapsed_time;
	cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

	int i, *dev_m, *dev_dim, *dev_result, result[1];

	cudaMalloc((void**)&dev_result, 1 * sizeof(int));
	cudaMalloc((void**)&dev_dim, (n + 1) * sizeof(int));
	cudaMalloc((void**)&dev_m, (n + 1) * (n + 1) * sizeof(int));
	cudaMemcpy(dev_dim, dim, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	for (i = 1; i <= 10; i++) CMM_CUDA_with_Data_Layout_kernel << <1, n >> >(dev_dim, dev_m, dev_result, n);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	*time = elapsed_time / 10.0;

	cudaMemcpy(result, dev_result, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_result); cudaFree(dev_dim); cudaFree(dev_m);

	return result[0];
}
