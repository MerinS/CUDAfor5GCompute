#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h> 
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>

//Input size is hardcoded to 76 
# define INPUT_SIZE 76
# define OUTPUT_SIZE 64
# define N_SUBCARRIERS 4
# define DATA_SUBCARRIER_SEND 19
# define DATA_SUBCARRIER_RECV 16
# define PERCENT_GUARD 20
# define GUARD_SIZE 3
struct complex{
	double re;
    double im;
};

//CUDA Error checks
#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
bool check(T result, char const *const func, const char *const file, int const line)
{
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		return true;
	}
	else {
		return false;
	}
}


//c function to do cyclic prefix removal


//cuda function to do cyclic prefix removal
//idea - the entire data is comprised of signals that were brought in by N_SUBCARRIERS
//The size of data brought in by each of them = DATA_PER_SUBCARRIER of which
//floor(PERCENT_GUARD*DATA_PER_SUBCARRIER/100) will have guard information
__global__ void CyclicPrefixRem(complex * ip, complex* op) 
{ 
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	int modval = i % 19;
	if (modval >= GUARD_SIZE && modval <= DATA_SUBCARRIER_SEND -1) {
		int x1 = (i / DATA_SUBCARRIER_SEND);
		int x2 = (DATA_SUBCARRIER_RECV * x1);
		int x3 = x2 + modval - 3;
		op[x3].im = ip[i].im;
		op[x3].re = ip[i].re;
	}
}

//cuda function that does FFT calculation - 
//This is a simple, straightforward DFT implementation
__global__ void FFTCalcBlock(complex * ip, complex* op, int N)
{
	int k = threadIdx.x;
	int offset = N * blockIdx.x;
	double constant = -(2 * 3.141592654 * (double)k) / (double)N; 
	float cos_val, sin_val;
	complex dummy;
	for (int i = 0; i < N; i++)
	{
		cos_val = cos(i*constant);
		sin_val = sin(i*constant);
		dummy = ip[i + offset];
		op[k + offset].re += ((dummy.re*cos_val) - (dummy.im*sin_val));
		op[k + offset].im += ((dummy.re*sin_val) + (dummy.im*cos_val)); 
	}
}

int main()
{
	cudaEvent_t start1, stop1, start2, stop2;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	//The modulated signals from Matlab are fed in directly
	complex input[INPUT_SIZE] = { {0.059713, 0.094092}, {0.239277, -0.025888}, {-0.059713, -0.357549}, {0.000000, 0.250000}, {-0.232549, 0.127363}, {0.150888, -0.187500}, {0.069228, -0.065287}, {0.000000, -0.125000}, {-0.007936, -0.395869}, {-0.114277, 0.150888}, {0.007936, -0.194228}, {-0.375000, -0.125000}, {-0.069228, -0.075586}, {-0.025888, -0.187500}, {0.232549, -0.132936}, {0.125000, 0.250000}, {0.059713, 0.094092}, {0.239277, -0.025888}, {-0.059713, -0.357549}, {-0.067650, 0.109127}, {0.176777, 0.176777}, {0.086680, 0.340097}, {-0.125000, 0.125000}, {0.163320, -0.013456}, {0.000000, 0.000000}, {0.317650, -0.244426}, {-0.125000, 0.125000}, {0.067650, 0.244426}, {-0.176777, -0.176777}, {0.413320, 0.013456}, {0.125000, -0.125000}, {-0.163320, -0.340097}, {0.000000, 0.000000}, {0.182350, -0.109127}, {0.125000, -0.125000}, {-0.067650, 0.109127}, {0.176777, 0.176777}, {0.086680, 0.340097}, {0.376709, 0.095671}, {0.327665, -0.224112}, {0.125000, 0.251709}, {-0.187500, 0.062500}, {0.104261, 0.230970}, {0.025888, -0.025888}, {0.125000, -0.156038}, {0.062500, 0.062500}, {0.050068, -0.095671}, {-0.202665, -0.400888}, {0.125000, -0.074932}, {0.062500, 0.312500}, {-0.031038, -0.230970}, {-0.150888, 0.150888}, {0.125000, -0.020739}, {0.062500, 0.062500}, {0.376709, 0.095671}, {0.327665, -0.224112}, {0.125000, 0.251709}, {0.191996, -0.182481}, {-0.099112, -0.062500}, {-0.134645, 0.157617}, {-0.000000, 0.125000}, {0.211810, -0.038974}, {0.062500, 0.025888}, {0.048489, -0.080452}, {-0.125000, 0.125000}, {-0.066996, 0.307481}, {-0.275888, -0.062500}, {0.259645, 0.320937}, {-0.375000, 0.250000}, {-0.086810, 0.163974}, {0.062500, -0.150888}, {0.076511, -0.148101}, {0.250000, 0.250000}, {0.191996, -0.182481}, {-0.099112, -0.062500}, {-0.134645, 0.157617}};
	
	complex *input_array, *output_array;
	size_t size = sizeof(complex);

	//allocated memory in the host side
	input_array = (complex *)malloc(INPUT_SIZE * size);
	output_array = (complex *)malloc(OUTPUT_SIZE * size);

	//printf("size of float = %d\n", sizeof(float));

	//output to test if things are okay + put the data into the host memory
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		input_array[i] = input[i];
	}

	//init the device
	int device = -1;
	cudaDeviceProp deviceProp;
	//hardcoding to use device 0 for now
	device =  0;
	//checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

	//**************************START OF THE PREFIX REMOVAL BLOCK***************************************
	//No of threads and blocks - defined here
	//no of data points that each subcarrier carries in
	int blocksPerGrid = DATA_SUBCARRIER_SEND;
	//no of subcarriers
	int threadsPerBlock = N_SUBCARRIERS;
	//allocating memory for the operations
	//allocating memory for the arrays in GPU device
	complex *input_gpu;
	complex *output_gpu;
	int res = cudaMalloc((void**)&input_gpu, INPUT_SIZE * size);
	if (res != 0) {
		printf("Allocation failed");
	}
	res = cudaMalloc((void**)&output_gpu, OUTPUT_SIZE * size);
	if (res != 0) {
		printf("Allocation failed");
	}
	//transfering data to memory
	res = cudaMemcpy(input_gpu, input_array, INPUT_SIZE * size, cudaMemcpyHostToDevice);
	if (res != 0) {
		printf("MemCopy failed");
	}

	cudaEventRecord(start1);
	// Copy result from device memory to host memory
	CyclicPrefixRem<<<blocksPerGrid, threadsPerBlock>>> (input_gpu, output_gpu);
	cudaEventRecord(stop1);
	res = cudaMemcpy(output_array, output_gpu, OUTPUT_SIZE * size, cudaMemcpyDeviceToHost);
	if (res != 0) {
		printf("MemCopy failed");
	}
	cudaEventSynchronize(stop1);
	float milliseconds1 = 0;
	cudaEventElapsedTime(&milliseconds1, start1, stop1);
	printf("\nTIme for task 1 = %f", milliseconds1);
	cudaFree(input_gpu); //cudaFree(output_gpu);
	//**************************END OF THE PREFIX REMOVAL BLOCK***************************************
	

	////**************************START OF THE FFT BLOCK***************************************
	//On the modulator side, we had taken the IFFT of 4 blocks, with IFFT length = 16, so the
	//process is to take the FFT of 4 blocks, each of size = 16 
	complex *inFFT_GPU, *outFFT_GPU, *output_array_FFT;
	output_array_FFT = (complex *)malloc(OUTPUT_SIZE * size);
	//orign data size per carrier
	blocksPerGrid = N_SUBCARRIERS;
	//no of subcarriers
	threadsPerBlock = DATA_SUBCARRIER_RECV;
	// Free device memory cudaFree(d_A); 
	res = cudaMalloc((void**)&inFFT_GPU, OUTPUT_SIZE * size);
	if (res != 0) {
		printf("Allocation failed");
	}
	res = cudaMalloc((void**)&outFFT_GPU, OUTPUT_SIZE * size);
	if (res != 0) {
		printf("Allocation failed");
	}
	//transfering data to memory
	res = cudaMemcpy(inFFT_GPU, output_array, OUTPUT_SIZE * size, cudaMemcpyHostToDevice);
	if (res != 0) {
		printf("MemCopy failed");
	}
	// Copy result from device memory to host memory
	cudaEventRecord(start2);
	FFTCalcBlock << <blocksPerGrid, threadsPerBlock >> > (output_gpu, outFFT_GPU, DATA_SUBCARRIER_RECV);
	cudaEventRecord(stop2);
	res = cudaMemcpy(output_array_FFT, outFFT_GPU, OUTPUT_SIZE * size, cudaMemcpyDeviceToHost);
	if (res != 0) {
		printf("MemCopy failed");
	}
	cudaEventSynchronize(stop2);
	float milliseconds2 = 0;
	cudaEventElapsedTime(&milliseconds2, start2, stop2);
	printf("\nTime for task 2 = %f", milliseconds2);

	return 1;
}
