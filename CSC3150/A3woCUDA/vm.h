#ifndef VM_H
#define VM_H

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace std;


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
	Node  map[N_PHYSICAL_PAGES];		/* Mapping from PP num to Node with VP num */
	Node* head;							/* ptr to recently used Node	   */
	Node* tail;							/* ptr to least recently used Node	   */
};

short table_translate_vp_to_pp(InvPageTable* table, short vp_num);
short table_find_empty_pp(InvPageTable* table);
short table_pop(InvPageTable* table);
void table_push(InvPageTable* table, short pp_num, short vp_num);

void table_print_map(InvPageTable* table, int first_n = 10);
void table_print_list(InvPageTable* table, int first_n = 10);


struct VirtualMemory {
	uchar* data;
	uchar* disk;
	InvPageTable* inv_page_table;
	int* n_page_fault_ptr;
};

// TODO
// __device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
// 						PageTable* page_table, int* n_page_fault_ptr);
void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
						InvPageTable* inv_page_table, int* n_page_fault_ptr);
uchar vm_read(VirtualMemory *vm, u32 va);
void vm_write(VirtualMemory *vm, u32 va, uchar value);
void vm_snapshot(VirtualMemory *vm, uchar *result, int offset,
							int input_size);

#endif
