#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define PAGE_SIZE 		  (1 << 5)   /* PAGE SIZE = 32 bytes */
#define PAGE_TABLE_SIZE	  (1 << 14)  /* PAGE TABLE (on shared mem.) = 16 KB */
#define PHYSICAL_MEM_SIZE (1 << 15)  /* PHYSICAL MEMORY (on shared mem.) = 32 KB */
#define DISK_SIZE 		  (1 << 17)  /* DISK (GLOBAL MEMORY) = 128 KB */

// count the pagefault times
__device__ __managed__ int n_page_fault = 0;

// data input and output
__device__ __managed__ uchar result[DISK_SIZE];
__device__ __managed__ uchar input[DISK_SIZE];

// memory allocation for virtual_memory
__device__ __managed__ uchar disk[DISK_SIZE];
extern __shared__ u32 page_table[];

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *result,
                             int input_size);

__global__ void mykernel(int input_size) {

	// physical memory
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	VirtualMemory vm;
	/* 
	vm_init(VirtualMemory *vm, uchar *buffer, uchar *disk,
			u32 *page_table, int *pagefault_num_ptr,
			int PAGESIZE, int PAGE_TABLE_SIZE,
			int PHYSICAL_MEM_SIZE, int DISK_SIZE,
			int PAGE_ENTRIES); */
	vm_init(&vm, data, disk, page_table, &n_page_fault,
			PAGE_SIZE, PAGE_TABLE_SIZE, PHYSICAL_MEM_SIZE, 
			DISK_SIZE, PHYSICAL_MEM_SIZE / PAGE_SIZE);

	// user program for testing paging
	user_program(&vm, input, result, input_size);
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize) {
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize) {
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
	/* number of bytes loaded from binary file to input buffer*/
	int input_size = load_binaryFile(DATAFILE, input, DISK_SIZE);

	/* Launch kernel function in GPU, with single thread
	and dynamically allocate PAGE_TABLE_SIZE bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel<<<1, 1, PAGE_TABLE_SIZE>>>(input_size);

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n",
				cudaGetErrorString(cudaStatus));
		return 0;
	}

	printf("input size: %d\n", input_size);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	/* write result buffer to OUTFILE */
	write_binaryFile(OUTFILE, result, input_size);

	printf("pagefault number is %d\n", n_page_fault);

	return 0;
}
