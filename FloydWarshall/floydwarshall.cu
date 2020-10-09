#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <thread>

#include <cuda_runtime.h>
#include "errors.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

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

void CPUMulti(float* d, int k, int s, int n, int CPU_THREADS)
{
	for (int c = s; c < n*n; c += CPU_THREADS)
	{
		int i = c/n;
		int j = c - i*n;
		if (d[n*i + j] > d[n*i + k] + d[n*k + j])
			d[n*i + j] = d[n*i + k] + d[n*k + j];
	}
}

void FloydWarshallCPUMulti(float* d, int n, int CPU_THREADS)
{
	for (int k = 0; k < n; ++k)
	{
		std::thread* t = new std::thread[CPU_THREADS];
		for (int i = 0; i < CPU_THREADS; ++i)
			t[i] = std::thread(CPUMulti, d, k, i, n, CPU_THREADS);
		for (int i = 0; i < CPU_THREADS; ++i)
			t[i].join();
		delete[] t;
	}
}