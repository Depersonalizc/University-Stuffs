#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE (1 << 20)
#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

// data input and output
__device__ __managed__ uchar input[BUFFER_SIZE];
__device__ __managed__ uchar output[BUFFER_SIZE];

// volume (disk storage)
__device__ __managed__ FileSystem fs;


__device__ void user_program(FileSystem* fs, uchar* input, uchar* output);

__global__ void mykernel(uchar* input, uchar* output) {
	fs_init(&fs);
	user_program(&fs, input, output);
}

__host__ void write_binaryFile(const char fileName[], void* buffer, int bufferSize) {
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(const char fileName[], void* buffer, int bufferSize) {
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}

int main() {
	cudaError_t cudaStatus;
	load_binaryFile(DATAFILE, input, BUFFER_SIZE);

	// Launch to GPU kernel with single thread
	mykernel<<<1, 1>>>(input, output);

	if ( (cudaStatus = cudaGetLastError()) != cudaSuccess ) {
		fprintf(stderr, "mykernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaDeviceSynchronize();
	cudaDeviceReset();

	write_binaryFile(OUTFILE, output, BUFFER_SIZE);


	return 0;
}
