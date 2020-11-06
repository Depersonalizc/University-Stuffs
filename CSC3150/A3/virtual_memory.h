#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

// const int VA_BITS = 17;				/* 17 bits 		*/
const int PAGE_SIZE = 1 << 5;			/* 32 B 		*/
const int PHYSICAL_MEM_SIZE = 1 << 15;  /* 32 KB 		*/
const int N_PHYSICAL_PAGES = 1 << 10; 	/* 32 KB / 32 B */
const int DISK_SIZE = 1 << 17;			/* 128 KB 		*/

/* Doubly linked list for LRU */
struct Node {
	short vp_num;						/* Virtual page number mapped to */
	Node* prev;
	Node* next;
};

struct InvPageTable {
	Node* map[N_PHYSICAL_PAGES];		/* Mapping from PP num to Node with VP num */
	Node* head;							/* ptr to recently used Node	   */
	Node* tail;							/* ptr to least recently used Node	   */
};

__device__ short table_translate_vp_to_pp(InvPageTable* table, short vp_num);
__device__ short table_find_empty_pp(InvPageTable* table);
__device__ short table_pop(InvPageTable* table);
__device__ void table_push(InvPageTable* table, short pp_num, short vp_num);


struct VirtualMemory {
	uchar* data;
	uchar* disk;
	InvPageTable* inv_page_table;
	int* n_page_fault_ptr;
};

// TODO
// __device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
// 						PageTable* page_table, int* n_page_fault_ptr);
__device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
						InvPageTable* inv_page_table, int* n_page_fault_ptr);
__device__ uchar vm_read(VirtualMemory *vm, u32 addr);
__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);
__device__ void vm_snapshot(VirtualMemory *vm, uchar *result, int offset,
							int input_size);

#endif
