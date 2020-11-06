#include "vm.h"

void init_inv_page_table(VirtualMemory *vm) {
	vm->inv_page_table->head = nullptr;
	vm->inv_page_table->tail = nullptr;
	for (int i = 0; i < N_PHYSICAL_PAGES; i++)
		vm->inv_page_table->map[i] = nullptr;
}

void vm_init(VirtualMemory* vm, uchar* data, uchar* disk,
	InvPageTable* inv_page_table, int* n_page_fault_ptr) {

	vm->data = data;
	vm->disk = disk;
	vm->inv_page_table = inv_page_table;
	vm->n_page_fault_ptr = n_page_fault_ptr;

	init_inv_page_table(vm);
}

/* return physical page number if virtual page found, return -1 otherwise */
short table_translate_vp_to_pp(InvPageTable* table, short vp_num) {
	for (int i = 0; i < N_PHYSICAL_PAGES; i++) {
		if (table->map[i] != nullptr && table->map[i]->vp_num == vp_num)
			return i;
	} return -1;
}

/* return first empty pp, return -1 otherwise. */
short table_find_empty_pp(InvPageTable* table) {
	for (int i = 0; i < N_PHYSICAL_PAGES; i++) {
		if (!table->map[i])
			return i;
	} return -1;
}

/* pop tail (LRU) Node out of list, return the physical page number  */
short table_pop(InvPageTable* table) {
	Node* tail_node = table->tail;
	short pp_num = table_translate_vp_to_pp(table, tail_node->vp_num);
	if (table->tail == table->head) {
		table->tail = nullptr;
		table->head = nullptr;
	} else {
		table->tail = tail_node->prev;
		table->tail->next = table->head;
		table->head->prev = table->tail;
	}
	table->map[pp_num] = nullptr;
	free(tail_node);
	return pp_num;
}

/* push Node to head of the list, map pp_num to vp_num */
void table_push(InvPageTable* table, short pp_num, short vp_num) {
	Node* new_node = (Node*) malloc(sizeof(Node));
	new_node->vp_num = vp_num;

	if (table->head == nullptr) {
		table->head = new_node;
		table->tail = new_node;
		new_node->next = new_node;
		new_node->prev = new_node;
	} else {
		new_node->prev = table->tail;
		new_node->next = table->head;
		table->tail->next = new_node;
		table->head->prev = new_node;
		table->head = new_node;
	}

	table->map[pp_num] = new_node;

	// printf("map[%d] := %d\n", pp_num, table->map[pp_num]);
}

void table_print_map(InvPageTable* table, int first_n){
	for (int i = 0; i < first_n; i++){
		printf("PP 0x%03x -> VP ", i);
		if (table->map[i])
			printf("0x%03x\n", table->map[i]->vp_num);
		else
			printf("0x---\n");
	}
}

void table_print_list(InvPageTable* table, int first_n){
	Node* curr = table->head;
	printf("{");
	if (curr != nullptr) {
		int i = 0;
		for (i = 0; i < first_n; i++) {
			printf("%x", curr->vp_num);
			if (curr == table->tail) break;
			curr = curr->next;
			printf(" <=> ");
		}
		if (curr != table->tail || i == first_n) 
			printf("...");
	}
	printf("}\n\n");
}

/* Read single byte from physical memory */
uchar vm_read(VirtualMemory *vm, u32 va) {
	u16 vp = va / PAGE_SIZE;
	u16 offset = va & 0b11111;
	InvPageTable* table = vm->inv_page_table;
	Node* node;
	uchar char_read;
	short pp;
	if ((pp = table_translate_vp_to_pp(table, vp)) == -1) {
		
		/* Page Fault */
		// printf("[vm_read]: Page fault! Virtual Page 0x%03x not in RAM.\n", vp);
		*(vm->n_page_fault_ptr) += 1;

		/* Retrieve data from disk */
		char_read = vm->disk[va];

		if ((pp = table_find_empty_pp(table)) == -1) {
			// printf("[vm_read]: Cannot find empty Physical Page.\n");
			/* RAM full. Replace LRU page */
			pp = table_pop(table);
			// printf("[vm_read]: Popped LRU Physical Page 0x%03x.\n", pp); 
			table_push(table, pp, vp);
		} else {
			/* Still exists empty page at pp
			 * Insert node, update map */
			table_push(table, pp, vp);
			// printf("[vm_read]: Empty page at PP=0x%03x, mapped to VP=0x%03x.\n", pp, table->map[pp]->vp_num);
		};
		/* In either case, move entire page from vm->disk[vp * PAGE_SIZE] ==> data[pp * PAGE_SIZE] */
		// TODO: move entire page vm->disk[vp * PAGE_SIZE..] ==1-PAGE==> data[pp * PAGE_SIZE]
		for (int i = 0; i < PAGE_SIZE; i++)
			vm->data[pp * PAGE_SIZE + i] = vm->disk[vp * PAGE_SIZE + i];

		// table_print_map(table);
		// table_print_list(table);

	} else {
		/* In RAM. Pop out Node to head. */
		// printf("[vm_read]: Page in RAM. Updating VP=0x%03x as MRU.\n", table->map[pp]->vp_num);
		if ((node = table->map[pp]) != table->head) {
			if (node == table->tail) table->tail = node->prev;
			node->prev->next = node->next;
			node->next->prev = node->prev;
			node->next = table->head;
			node->prev = table->tail;
			table->tail->next = node;
			table->head->prev = node;
			table->head = node;
			// table_print_map(table);
			// table_print_list(table);
		}
		/* Read from data */
		char_read = vm->data[(pp * PAGE_SIZE) + offset];
	}

	return char_read;
}

/* Write value into physical memory */
void vm_write(VirtualMemory *vm, u32 va, uchar value) {
	vm->disk[va] = value;
	u16 vp = va / PAGE_SIZE;
	u16 offset = va & 0b11111;
	InvPageTable* table = vm->inv_page_table;
	Node* node;
	short pp;

	if ((pp = table_translate_vp_to_pp(table, vp)) == -1) {
		
		/* Page fault */
		// printf("[vm_write]: Page fault! Virtual Page 0x%03x not in RAM.\n", vp);
		*(vm->n_page_fault_ptr) += 1;

		if ((pp = table_find_empty_pp(table)) == -1) {
			// printf("[vm_write]: Cannot find empty Physical Page.\n");
			/* RAM full. Replace LRU page */
			pp = table_pop(table);
			// printf("[vm_write]: Popped LRU Physical Page 0x%03x.\n", pp); 
			table_push(table, pp, vp);
		} else {
			/* Still exists empty page at pp
			 * Insert node, update map */
			table_push(table, pp, vp);
			// printf("[vm_write]: Empty page at PP=0x%03x, mapped to VP=0x%03x.\n", pp, table->map[pp]->vp_num);
		};

		// table_print_map(table);
		// table_print_list(table);

	} else {
		/* In RAM. Update page as MRU. */
		// printf("[vm_write]: Page in RAM. Updating VP=0x%03x as MRU.\n", table->map[pp]->vp_num);
		if ((node = table->map[pp]) != table->head) {
			if (node == table->tail) table->tail = node->prev;
			node->prev->next = node->next;
			node->next->prev = node->prev;
			node->next = table->head;
			node->prev = table->tail;
			table->tail->next = node;
			table->head->prev = node;
			table->head = node;
			// table_print_map(table);
			// table_print_list(table);
		}
	}
	vm->data[(pp * PAGE_SIZE) + offset] = value;
}

/* load elements from physical memory to result buffer */
void vm_snapshot(VirtualMemory *vm, uchar *result, 
				 int offset, int input_size) {
	for (int i = offset; i < input_size; i++)
		result[i] = vm_read(vm, i);
}

