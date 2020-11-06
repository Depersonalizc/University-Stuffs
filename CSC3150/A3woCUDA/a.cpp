#include "vm.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

int n_page_fault = 0;
uchar result[DISK_SIZE];
uchar input[DISK_SIZE];
uchar disk[DISK_SIZE];

void mykernel(int input_size) {

	/* Physical memory */
	uchar data[PHYSICAL_MEM_SIZE];
	InvPageTable inv_page_table;

	VirtualMemory vm;

	vm_init(&vm, data, disk, &inv_page_table, &n_page_fault);


	// table_print_map(&inv_page_table);
	// table_print_list(&inv_page_table);


	// /* user_program */
	// uchar c;
	// for (u32 va = 0; va < input_size; va++)
	// 	vm_write(&vm, va, input[va]);
	// for (u32 va = 0; va < input_size; va++) {
	// 	c = vm_read(&vm, va);
	// 	// printf("%d, ", c);
	// }
	for (int i = 0; i < input_size; i++)
		vm_write(&vm, i, input[i]);

	for (int i = input_size - 1; i >= input_size - 32769; i--)
		int value = vm_read(&vm, i);

	vm_snapshot(&vm, result, 0, input_size);

	// table_print_map(&inv_page_table);
	// table_print_list(&inv_page_table);


	// for (u32 va = 0; va < 4 * 1024 * PAGE_SIZE; va++){
	// 	vm_write(&vm, va, (uchar) va);
	// }

	// for (u32 va = 0; va < 4 * 1024 * PAGE_SIZE; va++){
	// 	printf("%d, ", vm_read(&vm, va));
	// }

	// vm_write(&vm, 0*PAGE_SIZE, 0);
	// table_print_map(&inv_page_table);
	// table_print_list(&inv_page_table);


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
	/* number of bytes loaded from binary file to input buffer */
	int input_size = load_binaryFile(DATAFILE, input, DISK_SIZE);
	
	// for (int i = 0; i < 100000; i++) {
	// 	printf("%u, ", input[i]);
	// }

	mykernel(input_size);

	printf("input size: %d\n", input_size);

	// /* write result buffer to OUTFILE */
	write_binaryFile(OUTFILE, result, input_size);

	printf("pagefault number is %d\n", n_page_fault);

	// load_binaryFile(OUTFILE, input, DISK_SIZE);
	// for (int i = 0; i < 100000; i++) {
	// 	printf("%u, ", input[i]);
	// }

	return 0;
}