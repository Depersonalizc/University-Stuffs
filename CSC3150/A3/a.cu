#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

__managed__ int n_page_fault = 0;
__managed__ uchar result[DISK_SIZE];
__managed__ uchar input[DISK_SIZE];
__managed__ uchar disk[DISK_SIZE];

__global__ void mykernel(int input_size) {

	/* Physical memory */
	__shared__ uchar data[PHYSICAL_MEM_SIZE];
	__shared__ InvPageTable inv_page_table;

	VirtualMemory vm;

	vm_init(&vm, data, disk, &inv_page_table, &n_page_fault);

	// user program for testing paging
	// user_program(&vm, input, result, input_size);
}

void write_binaryFile(const char *fileName, void *buffer, int bufferSize) {
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

int load_binaryFile(const char *fileName, void *buffer, int bufferSize) {
	FILE *fp;

	fp = fopen(fileName, "rb");
	if (!fp) {
	printf("***Unable to open file %s***\n", fileName);
	exit(1);
	}

	// Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize) {
	printf("****invalid testcase!!****\n");
	printf("****software warrning: the file: %s size****\n", fileName);
	printf("****is greater than buffer size****\n");
	exit(1);
	}

	// Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);

	return fileLen;
}


int main() {
	cudaError_t cudaStatus;
	/* number of bytes loaded from binary file to input buffer */
	int input_size = load_binaryFile(DATAFILE, input, DISK_SIZE);

	/* Launch kernel function in GPU, with single thread */
	mykernel<<<1, 1>>>(input_size);

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return 0;
	}

	printf("input size: %d\n", input_size);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	/* write result buffer to OUTFILE */
	// write_binaryFile(OUTFILE, result, input_size);

	printf("pagefault number is %d\n", n_page_fault);

	return 0;
}