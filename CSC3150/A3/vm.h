#ifndef VM_H
#define VM_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>

typedef unsigned char uchar;
typedef uint32_t u32;

const int PAGE_SIZE = 1 << 5;			/* 32 B 		*/
const int PHYSICAL_MEM_SIZE = 1 << 15;  /* 32 KB 		*/
const int N_PHYSICAL_PAGES = 1 << 10; 	/* 32 KB / 32 B */
const int DISK_SIZE = 1 << 17;			/* 128 KB 		*/

struct VirtualMemory {
	uchar* data;			/* Physical Memory */
	uchar* disk;			/* External Storage */
	short* pt;				/* Mapping from physical page to virtual */
	u32* count;				/* Counters for LRU. Last one is global counter */
	int* n_page_fault_ptr;	/* Pointer to page fault number */
};

__device__ int vm_translate_vp_to_pp(VirtualMemory* vm, short vp);
__device__ int vm_find_LRU_pp(VirtualMemory* vm);
__device__ void  vm_push_vp_at_pp(VirtualMemory* vm, short vp, short pp);
__device__ void vm_print_map(VirtualMemory* vm, int first_n = 5);

__device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
						short* pt, u32* count, int* n_page_fault_ptr);
__device__ uchar vm_read(VirtualMemory *vm, u32 va);
__device__ void vm_write(VirtualMemory *vm, u32 va, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *result, int offset,
							int input_size);

#endif
