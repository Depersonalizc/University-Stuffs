#include "vm.h"

void user_program(VirtualMemory *vm, uchar *input,
				  uchar *result, int input_size) {
	/* address i = 0..input_size - 1 */
	for (int i = 0; i < input_size; i++)
		vm_write(vm, i, input[i]);

	for (int i = input_size - 1; i >= input_size - 32769; i--)
		int value = vm_read(vm, i);

	// vm_snapshot(vm, result, 0, input_size);
}
