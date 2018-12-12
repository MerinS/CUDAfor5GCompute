#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//Input size is hardcoded to 76 
# define INPUT_SIZE 76
# define OUTPUT_SIZE 64
# define N_SUBCARRIERS 4
# define DATA_SUBCARRIER_SEND 19
# define DATA_SUBCARRIER_RECV 16
# define PERCENT_GUARD 20
# define GUARD_SIZE 3
#define BILLION 1E9
struct complex {
	double re;
	double im;
};

//c function to do cyclic prefix removal
void CyclicPrefixRem(complex * ip, complex* op)
{
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		int modval = i % 19;
		if (modval >= GUARD_SIZE && modval <= DATA_SUBCARRIER_SEND - 1) {
			int x1 = (i / DATA_SUBCARRIER_SEND);
			int x2 = (DATA_SUBCARRIER_RECV * x1);
			int x3 = x2 + modval - 3;
			op[x3].im = ip[i].im;
			op[x3].re = ip[i].re;
		}
	}
	return;
}

//This is a simple, straightforward DFT implementation
void FFTCalcBlock(complex * ip, complex* op, int N)
{
	for (int j = 0; j < OUTPUT_SIZE; j++)
	{
		int k = (j % DATA_SUBCARRIER_RECV);
		int offset = N * (j / DATA_SUBCARRIER_RECV);
		double constant = -2 * 3.141592654 * (double)k / (double)N;
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
	return;
}
int main()
{
	//The modulated signals from Matlab are fed in directly
	complex input[INPUT_SIZE] = { {0.059713, 0.094092}, {0.239277, -0.025888}, {-0.059713, -0.357549}, {0.000000, 0.250000}, {-0.232549, 0.127363}, {0.150888, -0.187500}, {0.069228, -0.065287}, {0.000000, -0.125000}, {-0.007936, -0.395869}, {-0.114277, 0.150888}, {0.007936, -0.194228}, {-0.375000, -0.125000}, {-0.069228, -0.075586}, {-0.025888, -0.187500}, {0.232549, -0.132936}, {0.125000, 0.250000}, {0.059713, 0.094092}, {0.239277, -0.025888}, {-0.059713, -0.357549}, {-0.067650, 0.109127}, {0.176777, 0.176777}, {0.086680, 0.340097}, {-0.125000, 0.125000}, {0.163320, -0.013456}, {0.000000, 0.000000}, {0.317650, -0.244426}, {-0.125000, 0.125000}, {0.067650, 0.244426}, {-0.176777, -0.176777}, {0.413320, 0.013456}, {0.125000, -0.125000}, {-0.163320, -0.340097}, {0.000000, 0.000000}, {0.182350, -0.109127}, {0.125000, -0.125000}, {-0.067650, 0.109127}, {0.176777, 0.176777}, {0.086680, 0.340097}, {0.376709, 0.095671}, {0.327665, -0.224112}, {0.125000, 0.251709}, {-0.187500, 0.062500}, {0.104261, 0.230970}, {0.025888, -0.025888}, {0.125000, -0.156038}, {0.062500, 0.062500}, {0.050068, -0.095671}, {-0.202665, -0.400888}, {0.125000, -0.074932}, {0.062500, 0.312500}, {-0.031038, -0.230970}, {-0.150888, 0.150888}, {0.125000, -0.020739}, {0.062500, 0.062500}, {0.376709, 0.095671}, {0.327665, -0.224112}, {0.125000, 0.251709}, {0.191996, -0.182481}, {-0.099112, -0.062500}, {-0.134645, 0.157617}, {-0.000000, 0.125000}, {0.211810, -0.038974}, {0.062500, 0.025888}, {0.048489, -0.080452}, {-0.125000, 0.125000}, {-0.066996, 0.307481}, {-0.275888, -0.062500}, {0.259645, 0.320937}, {-0.375000, 0.250000}, {-0.086810, 0.163974}, {0.062500, -0.150888}, {0.076511, -0.148101}, {0.250000, 0.250000}, {0.191996, -0.182481}, {-0.099112, -0.062500}, {-0.134645, 0.157617} };
	complex *input_array, *output_array;
	size_t size = sizeof(complex);

	//allocated memory
	input_array = (complex *)malloc(INPUT_SIZE * size);
	output_array = (complex *)malloc(OUTPUT_SIZE * size);
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		input_array[i] = input[i];
	}

	//**************************START OF THE PREFIX REMOVAL BLOCK***************************************
	struct timespec requestStart1, requestEnd1;
	timespec_get(&requestStart1, TIME_UTC);
	CyclicPrefixRem(input_array, output_array);
	timespec_get(&requestEnd1, TIME_UTC);
	double accum = (requestEnd1.tv_sec - requestStart1.tv_sec)
		+ double(requestEnd1.tv_nsec - requestStart1.tv_nsec)
		/ BILLION;
	printf("TIme for task 1 = %lf\n", accum);
	//**************************END OF THE PREFIX REMOVAL BLOCK***************************************


	////**************************START OF THE FFT BLOCK***************************************
	//On the modulator side, we had taken the IFFT of 4 blocks, with IFFT length = 16, so the
	//process is to take the FFT of 4 blocks, each of size = 16 
	//clock_t start2 = clock();
	complex * output_array_FFT = (complex *)malloc(OUTPUT_SIZE * size);
	struct timespec requestStart, requestEnd;
	timespec_get(&requestStart, TIME_UTC);
	FFTCalcBlock(output_array, output_array_FFT, DATA_SUBCARRIER_RECV);
	timespec_get(&requestEnd, TIME_UTC);
	double accum1 = (requestEnd.tv_sec - requestStart.tv_sec)
		+ double(requestEnd.tv_nsec - requestStart.tv_nsec)
		/ BILLION;
	printf("TIme for task 1 = %lf\n", accum1);
	return 1;
}
