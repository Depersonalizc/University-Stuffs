#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_inv_page_table(VirtualMemory *vm) {
	// for (int i = 0; i < N_PHYSICAL_PAGES; i++) {
	// 	vm->inv_page_table->
	// }
}

__device__ void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
	InvPageTable* inv_page_table, int* n_page_fault_ptr) {

	vm->data = data;
	vm->disk = disk;
	vm->inv_page_table = inv_page_table;
	vm->n_page_fault_ptr = n_page_fault_ptr;

	init_inv_page_table(vm);
}

/* return physical page number if virtual page found, return -1 otherwise */
__device__ short table_translate_vp_to_pp(InvPageTable* table, short vp_num) {
	for (int i = 0; i < N_PHYSICAL_PAGES; i++) {
		if (table->map[i] != nullptr && table->map[i]->vp_num == vp_num)
			return i;
	} return -1;
}

/* return first empty pp, return -1 otherwise. */
__device__ short table_find_empty_pp(InvPageTable* table) {
	for (int i = 0; i < N_PHYSICAL_PAGES; i++) {
		if (table->map[i] == nullptr) return i;
	} return -1;
}

/* pop tail (LRU) Node out of list, return the physical page number  */
__device__ short table_pop(InvPageTable* table) {
	Node* tail_node = table->tail;
	short pp_num = table_translate_vp_to_pp(table, tail_node->vp_num);
	table->tail = tail_node->prev;
	table->tail->next = table->head;
	table->head->prev = table->tail;
	table->map[pp_num] = nullptr;
	cudaFree(tail_node);
	return pp_num;
}

/* push Node to head of the list, map pp_num to vp_num */
__device__ void table_push(InvPageTable* table, short pp_num, short vp_num) {
	Node* new_node;
	cudaMalloc((void**)&new_node, sizeof(Node));
	new_node->vp_num = vp_num;
	new_node->prev = table->tail;
	new_node->next = table->head;
	table->tail->next = new_node;
	table->head->prev = new_node;
	table->head = new_node;
	table->map[pp_num] = new_node;
}


__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
	/* Complete vm_read function to read single byte from physical memory */
	u32 vp = addr / PAGE_SIZE;

	return 123;
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	/* Complete vm_write function to write value into physical memory */
	u32 vp = addr / PAGE_SIZE;

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *result, int offset,
                            int input_size) {
	/* Complete snapshot function togther with vm_read to load elements from
	* physical memory to result buffer */
}

