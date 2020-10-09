#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "FloydWarshall\floydwarshall.cu"

#define INF 1e4
#define BLOCK_SIZE 32

class Tokenizer
{
	public:

	Tokenizer(std::string filename)
	{
		in.open(filename);		
	}

	std::string getToken()
	{
        return (ss >> token) ? token : "";
	}

	std::string getLine()
	{
		return line;
	}

    bool nextLine()
    {
        while(getline(in, line))
		{		
			if (!line.size()) continue;
			ss.clear();
			ss.str(line);
			return true;
		}
		return false;
    }

    private:

	std::string token, line;
	std::istringstream ss;
	std::ifstream in;
};

void PrintMatrix(float* d, int n)
{
	for (int i = 0; i < n; ++i)
	{
		printf("\n");
		for (int j = 0; j < n; ++j)
		{
		    float t = d[i*n + j];
		    if(t == INF)
		        printf("INF ");
		    else
			    printf("%.2f ", t);
		}
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

void RunExample(int n, int CPU_THREADS, float* mat, bool print)
{
	size_t size = n*n * sizeof(float);
	float* h_W = (float*)malloc(size);

	int cpu_threads = std::thread::hardware_concurrency();

	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	memcpy(h_W, mat, size);

	float* d_W;
	checkCudaErrors(cudaMalloc(&d_W, size));
	checkCudaErrors(cudaMemcpyAsync(d_W, h_W, size, cudaMemcpyHostToDevice, stream));

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(n / threads.x + 1, n / threads.y + 1);

	// Warmup
	for (int k = 0; k < n; ++k)
	{
		FloydWarshall<<<grid, threads, 0, stream>>>(d_W, k, n);
		cudaDeviceSynchronize();
	}
	checkCudaErrors(cudaMemcpyAsync(h_W, d_W, size, cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));


	// GPU Time
	checkCudaErrors(cudaEventRecord(start, stream));
	for (int k = 0; k < n; ++k)
	{
		FloydWarshall<<<grid, threads, 0, stream>>>(d_W, k, n);
		cudaDeviceSynchronize();
	}
	checkCudaErrors(cudaEventRecord(stop, stream));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("Time elapsed (GPU): %.6f\n", msecTotal / 1000);
	
	// CPU Multi Time
	clock_t cpu_startTime, cpu_endTime;
	cpu_startTime = clock();
	FloydWarshallCPUMulti(h_W, n, CPU_THREADS);
	cpu_endTime = clock();
	printf("Time elapsed (CPU, multithread): %.6f\n", ((double)(cpu_endTime - cpu_startTime)) / CLOCKS_PER_SEC);

	// CPU Single Time
	cpu_startTime = clock();
	FloydWarshallCPU(h_W, n);
	cpu_endTime = clock();
	printf("Time elapsed (CPU, single thread): %.6f\n", ((double)(cpu_endTime - cpu_startTime)) / CLOCKS_PER_SEC);

	if(print)
		PrintMatrix(h_W, n);
	
	// Free device memory
	cudaFree(d_W);

	// Free host memory
	free(h_W);
}

int main(int argc, char** argv)
{
	if(argc < 3)
	{
		printf("Not enough arguements...\n");
		return 0;
	}
	std::string argv1 = argv[1];
	std::string argv2 = argv[2];
	bool print = false;
	if(argv1 == "random")
	{
		std::string argv3 = argv[3];
		int n = std::stoi(argv2);
		int CPU_THREADS = std::stoi(argv3);
		if(argc < 4)
		{
			printf("Not enough arguements...\n");
			return 0;
		}
		if(argc >= 5)
		{
			std::string argv4 = argv[4];
			if(argv4 == "true")
				print = true;
			else if(argv4 == "false")
				print = false;
		}
		float* mat = (float*)malloc(n*n*sizeof(float));
		RandomMatrix(mat, n);
		RunExample(n, CPU_THREADS, mat, print);
	}
	else
	{
		Tokenizer T(argv1);
		std::vector<float> V;
		int n = 0;
		int CPU_THREADS = std::stoi(argv2);
		
		while(T.nextLine())
		{
			for(int i = 0; 1; ++i)
			{
				std::string s = T.getToken();
				if (s.size())
				{
					if(s != "INF")
						V.push_back(std::stof(s));
					else
						V.push_back(INF);
					continue;
				}
				else if (!n)
					n = i;
				break;
			}
		}
		if(argc >= 4)
		{
			std::string argv3 = argv[3];
			if(argv3 == "true")
				print = true;
			else if(argv3 == "false")
				print = false;
		}
		RunExample(n, CPU_THREADS, V.data(), print);
	}
	return 0;
}