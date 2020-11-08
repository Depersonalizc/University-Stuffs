#include "vm.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

__managed__ int   n_page_fault = 0;
__managed__ uchar result[DISK_SIZE];
__managed__ uchar input[DISK_SIZE];
__managed__ uchar disk[DISK_SIZE];

__device__ void user_program(VirtualMemory* vm, uchar *input,
							 uchar *result, int input_size) {

	for (int i = 0; i < input_size; i++)
		vm_write(vm, i, input[i]);

	for (int i = input_size - 1; i >= input_size - 32769; i--)
		int value = vm_read(vm, i);

	vm_snapshot(vm, result, 0, input_size);

}

__global__ void my_kernel(int input_size) {

	/* Physical memory */
	__shared__ uchar data[PHYSICAL_MEM_SIZE];
	__shared__ short   pt[N_PHYSICAL_PAGES];
	__shared__ u32  count[N_PHYSICAL_PAGES + 1];

	VirtualMemory vm;
	vm_init(&vm, data, disk, pt, count, &n_page_fault);

	user_program(&vm, input, result, input_size);
	vm_print_map(&vm);
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

	/* number of bytes loaded from binary file to input buffer
	 * 131072 bytes == 128 KB == 4 * RAM */
	int input_size = load_binaryFile(DATAFILE, input, DISK_SIZE);

	printf("[main] : Starting my_kernel...\n");
	my_kernel<<<1, 1>>>(input_size);

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "[main] : my_kernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaDeviceSynchronize();

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "[main] : my_kernel sync failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaDeviceReset();

	printf("[main] : my_kernel has ended.\n");

	write_binaryFile(OUTFILE, result, input_size);

	printf("[main] : Input size: %d\n", input_size);
	printf("[main] : Page Fault number is %d.\n", n_page_fault);

	return 0;
}