#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <thread>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "device_launch_parameters.h"

#define INF 1e4
#define BLOCK_SIZE 32
#define CPU_THREADS 20

// GPU Floyd Warshall
__global__ void FloydWarshall(float* d, int k, int n)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = by*BLOCK_SIZE + ty;
	int j = bx*BLOCK_SIZE + tx;

	int ij = n*i + j;

	if (i < n && j < n)
	{
		float dist = d[n*i + k] + d[n*k + j];
		d[ij] = dist*(dist < d[ij]) + d[ij] * (dist >= d[ij]);
	}
}

void FloydWarshallCPU(float* d, int n)
{
	for (int k = 0; k < n; ++k)
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				if (d[n*i + j] > d[n*i + k] + d[n*k + j])
					d[n*i + j] = d[n*i + k] + d[n*k + j];
}

void CPUMulti(float* d, int k, int s, int n)
{
	for (int c = s; c < n*n; c += CPU_THREADS)
	{
		int i = c/n;
		int j = c - i*n;
		if (d[n*i + j] > d[n*i + k] + d[n*k + j])
			d[n*i + j] = d[n*i + k] + d[n*k + j];
	}
}

void FloydWarshallCPUMulti(float* d, int n)
{
	for (int k = 0; k < n; ++k)
	{
		std::thread* t = new std::thread[CPU_THREADS];
		for (int i = 0; i < CPU_THREADS; ++i)
			t[i] = std::thread(CPUMulti, d, k, i, n);
		for (int i = 0; i < CPU_THREADS; ++i)
			t[i].join();
		delete[] t;
	}
}

void PrintMatrix(float* d, int n)
{
	for (int i = 0; i < n; ++i)
	{
		printf("\n");
		for (int j = 0; j < n; ++j)
			printf("%.2f ", d[i*n + j]);
	}
	printf("\n");
}

void RandomMatrix(float* d, int n)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
		{
			if (i != j)
			{
				int p = rand() % 100;
				if (p < 15)
					d[i*n + j] = INF;
				else
					d[i*n + j] = (float)(rand() % 1000) / 100;
			}
			else
				d[i*n + j] = 0;
		}
}

float ErrorNorm(float* a, float* b, int n)
{
	float e = 0;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			float d = abs(a[i*n + j] - b[i*n + j]);
			if (e < d)
				e = d;

		}
	}
	return e;
}

// Host code
int main()
{
	int n = 5000;
	size_t size = n*n * sizeof(float);
	float* h_W = (float*)malloc(size);

	int cpu_threads = std::thread::hardware_concurrency();

	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	RandomMatrix(h_W, n);
	//PrintMatrix(h_W, n);

	float* d_W;
	checkCudaErrors(cudaMalloc(&d_W, size));
	checkCudaErrors(cudaMemcpyAsync(d_W, h_W, size, cudaMemcpyHostToDevice, stream));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(n / threads.x + 1, n / threads.y + 1);

	//FloydWarshallCPUMulti(h_W, n);
	//printf("Gotov CPU Multi.\n");

	// Warmup
	for (int k = 0; k < n; ++k)
	{
		FloydWarshall<<<grid, threads, 0, stream>>>(d_W, k, n);
		cudaDeviceSynchronize();
	}
	checkCudaErrors(cudaMemcpyAsync(h_W, d_W, size, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));


	// GPU Time
	int num_iter = 1;
	checkCudaErrors(cudaEventRecord(start, stream));
	for (int i = 0; i < num_iter; ++i)
	{
		for (int k = 0; k < n; ++k)
		{
			FloydWarshall<<<grid, threads, 0, stream>>>(d_W, k, n);
			cudaDeviceSynchronize();
		}
	}
	checkCudaErrors(cudaEventRecord(stop, stream));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("Time elapsed (GPU): %.6f\n", msecTotal / 1000 / num_iter);
	
	// CPU Multi Time
	clock_t cpu_startTime, cpu_endTime;
	cpu_startTime = clock();
	for (int i = 0; i < num_iter; ++i)
		FloydWarshallCPUMulti(h_W, n);
	cpu_endTime = clock();
	printf("Time elapsed (CPU, multithread): %.6f\n", ((double)(cpu_endTime - cpu_startTime)) / CLOCKS_PER_SEC / num_iter);

	// CPU Single Time
	cpu_startTime = clock();
	for (int i = 0; i < num_iter; ++i)
		FloydWarshallCPU(h_W, n);
	cpu_endTime = clock();
	printf("Time elapsed (CPU, single thread): %.6f\n", ((double)(cpu_endTime - cpu_startTime)) / CLOCKS_PER_SEC / num_iter);


	// Free device memory
	cudaFree(d_W);

	// Free host memory
	free(h_W);
	//free(h_W_test);
}