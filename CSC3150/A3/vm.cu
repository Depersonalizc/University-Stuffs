#include "vm.h"

__device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
						short* pt, u32* count, int* n_page_fault_ptr) {
	vm->data = data;
	vm->disk = disk;
	vm->pt = pt;
	vm->count = count;
	vm->n_page_fault_ptr = n_page_fault_ptr;

	/* Inversed page table init */
	for (int pp = 0; pp < N_PHYSICAL_PAGES; pp++) {
		vm->pt[pp] = -1;	// -1 := invalid
		vm->count[pp] = 0;
	}
	vm->count[N_PHYSICAL_PAGES] = 0; // global clock init
}

/* return physical page number if virtual page found, return -1 otherwise. */
__device__ int vm_translate_vp_to_pp(VirtualMemory* vm, short vp) {
	for (int pp = 0; pp < N_PHYSICAL_PAGES; pp++) {
		if (vm->pt[pp] == vp)
			return pp;
	} return -1;
}

/* return first empty physical page. If no vacancy,
   return first physical page with largest count. */
__device__ int vm_find_LRU_pp(VirtualMemory* vm) {
	int LRU_pp;
	u32 LRU_count = 0xFFFFFFFF;
	for (int pp = 0; pp < N_PHYSICAL_PAGES; pp++) {
		/* empty page found */
		if (vm->pt[pp] < 0) return pp;
		/* older page found */
		if (vm->count[pp] < LRU_count) {
			LRU_pp = pp;
			LRU_count = vm->count[pp];
		}
	} return LRU_pp;
}

__device__ void vm_print_map(VirtualMemory* vm, int first_n){
	
	printf("PAGE TABLE [%d]\n", vm->count[N_PHYSICAL_PAGES]);
	for (int pp = 0; pp < first_n; pp++) {
		printf("PP 0x%04x -> VP ", pp);
		if (vm->pt[pp] >= 0)
			 printf("0x%04x [%d]\n", vm->pt[pp], vm->count[pp]);
		else printf("0x----\n");
	}

	printf("⋮ \n");

	for (int pp = N_PHYSICAL_PAGES - first_n;\
		 pp < N_PHYSICAL_PAGES; pp++) 
	{
		printf("PP 0x%04x -> VP ", pp);
		if (vm->pt[pp] >= 0)
			printf("0x%04x [%d]\n", vm->pt[pp], vm->count[pp]);
		else printf("0x----\n");
	}

	printf("\n");
}

/* Read single byte from physical memory */
__device__ uchar vm_read(VirtualMemory *vm, u32 va) {
	uchar char_read;
	u32 vp = va >> 5;						/* va / PAGE_SIZE */
	u32 offset = va & 0b11111;
	int pp;

	/* UPDATE GLOBAL COUNTER */
	vm->count[N_PHYSICAL_PAGES]++;

	/* Page Fault */
	if ((pp = vm_translate_vp_to_pp(vm, vp)) < 0) {

		/* Increment PF counter */
		(*vm->n_page_fault_ptr)++;

		/* Retrieve data from disk */
		char_read = vm->disk[va];

		/* Update page table */
		pp = vm_find_LRU_pp(vm); // LRU or empty page
		vm->pt[pp] = vp;		 // push vp at pp

		/* Move entire page from disk to data */
		for (int i = 0; i < PAGE_SIZE; i++)
			vm->data[(pp<<5) + i] = vm->disk[(vp<<5) + i];

		// printf("[vm_read] : Page Fault! VA %d (VP %d) not in RAM.\n", va, vp);
		// printf("[vm_read] : Page Fault number is %d.\n", *vm->n_page_fault_ptr);
		// vm_print_map(vm);

	} else {
		/* Read from data */
		char_read = vm->data[(pp<<5) + offset];
		// printf("[vm_read] : VA %d (VP %d) in RAM. Updating page count.\n", va, vp);
	}

	/* Make pp MRU */
	vm->count[pp] = vm->count[N_PHYSICAL_PAGES];

	return char_read;
}

/* Write value into physical memory */
__device__ void vm_write(VirtualMemory *vm, u32 va, uchar value) {
	u32 vp = va >> 5;
	u32 offset = va & 0b11111;
	int pp;

	/* UPDATE GLOBAL COUNTER */
	vm->count[N_PHYSICAL_PAGES]++;

	/* Save a copy of data on disk */
	vm->disk[va] = value;

	/* Page Fault */
	if ((pp = vm_translate_vp_to_pp(vm, vp)) < 0) {

		(*vm->n_page_fault_ptr)++;

		/* Update page table */
		pp = vm_find_LRU_pp(vm); // pp = LRU or empty page
		vm->pt[pp] = vp;		 // map pp to vp

		/* Move entire page from disk to data */
		for (int i = 0; i < PAGE_SIZE; i++)
			vm->data[(pp<<5) + i] = vm->disk[(vp<<5) + i];
		
		// printf("[vm_write] : Page fault! VA %d (VP %d) not in RAM.\n", va, vp);
		// printf("[vm_write] : Page Fault number is %d.\n", *vm->n_page_fault_ptr);
		// vm_print_map(vm);

	} else {
		/* Directly update byte value */
		vm->data[(pp<<5) + offset] = value;
		// printf("[vm_write] : VA %d (VP %d) in RAM. Updating page count.\n", va, vp);
	}

	/* Make pp MRU */
	vm->count[pp] = vm->count[N_PHYSICAL_PAGES];

}

/* load elements from physical memory to result buffer */
__device__ void vm_snapshot(VirtualMemory *vm, uchar *result, int offset, int input_size) {
	for (int i = offset; i < input_size + offset; i++)
		result[i] = vm_read(vm, i);
}

