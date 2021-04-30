#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

/* Utilities */

/* Return number of blocks needed to store nb bytes of data. 
 * 0 byte needs 1 block to store. */
__device__ u32 bytes2blocks(u32 nb) {
	return nb == 0? 1 : (nb - 1) / DATA_BLOCK_SIZE + 1;
}

/* Adapted from: http://tekpool.wordpress.com/category/bit-count/
 * Return location of first 1-bit in x, LSB to MSB, return 32 if not found. */
__device__ int first_1bit(int x) {
	u32 u = (x & (-x)) - 1;
	u32 uCount = u
	- ((u >> 1) & 033333333333)
	- ((u >> 2) & 011111111111);
	return
	((uCount + (uCount >> 3))
	& 030707070707) % 63;
}

/* Return location of first n consecutive 1-bit's in x, LSB to MSB.
 * Return -1 if not found. */
__device__ int first_n_1bit(u32 x, int n) {
	int ones = 0;
	int count = 0;
    for (; x != 0; x >>= 1, count++) {
        ones = (x & 1)? ones+1 : 0;
        if (ones >= n)
            return count - n + 1;
    }
    return -1;
}

/* https://stackoverflow.com/questions/50196897/creating-a-bit-
 * mask-for-relevant-bits-over-multiple-bytes-programmatically */
__device__ u32 bitmask(int start, int length) {
	u32 mask = 0xffffffff;
	mask >>= 32 - length;
    mask <<= start;
    return mask;
}

__device__ char* my_strcpy(char* dest, const char* src) {
   char *save = dest;
   while (*dest++ = *src++);
   return save;
}

__device__ int my_strcmp(const char* s1, const char* s2) {
    uchar c1, c2;
    do {
        c1 = *s1++;
        c2 = *s2++;
        if (c1 == '\0')
            return c1 - c2;
    } while (c1 == c2);
    return c1 - c2;
}

__device__ void swap(int* x, int* y) { 
    int temp = *x; 
    *x = *y;
    *y = temp; 
} 

__device__ void print_centered(const char *str, int width, const char pad) {
	char* padded = new char[width];
	memset(padded, pad, width);
	const char* s;
	for (s = str; *s; ++s);
	int strLen = s - str;
	int padLen = (width - strLen) / 2;
	memcpy(&padded[padLen], str, strLen);
	printf("%s\n", padded);
	delete[] padded;
}


/* File System */

/* Scan linearly the files array and return the first uninitialized fp. 
 * Return `EMPTY` if all `file` are initialized. */
__device__ u32 fs_find_empty_fp(FileSystem* fs) {
	for (auto fp = 0; fp < FCB_ENTRIES; ++fp) {
		if (fs->files[fp].starting_block == EMPTY)
			return fp;
	}	return EMPTY;
}

__device__ u32 fs_find_name(FileSystem* fs, const char fname[]) {
	u32 fp, fcount;
	for (fp = fcount = 0; fp < FCB_ENTRIES && fcount < fs->nfiles; ++fp) {
		if (fs->files[fp].starting_block != EMPTY) {
			++fcount;
			if (my_strcmp(fs->files[fp].fname, fname) == 0)
				return fp;
		}
	}
	return EMPTY;
}

__device__ void fs_init(FileSystem *fs) {
	/* init all blocks status as empty (1) */
	memset(fs->super, EMPTY, sizeof(fs->super));

	/* init disk */
	memset(fs->data, 0, sizeof(fs->data));

	/* init FCB */
	fs->nfiles = 0;
	for (int i = 0; i < FCB_ENTRIES; ++i)
		fs->files[i].starting_block = EMPTY;
}

__device__ u32 fs_open(FileSystem* fs, const char fname[], int op) {
	u32 fp;
	if ( (fp = fs_find_name(fs, fname)) != EMPTY ) {
		printf("[fs_open] : File \"%s\" opened, fp: %d\n", fname, fp);
		fs->files[fp].btime = ++gtime;
		return fp;
	}

	// file doesn't exist. create new one.
	printf("[fs_open] : Creating file \"%s\"\n", fname);

	// error if number of files reached max
	if (fs->nfiles >= FCB_ENTRIES) {
		printf("[fs_open] : Failed. Number of files (%d) reached maximum!\n", FCB_ENTRIES);
		return EMPTY;
	}

	// find empty fp
	fp = fs_find_empty_fp(fs);

	// find empty block for new file
	int block_offset;
	for (auto i = 0; i < N_SUPERBLOCKS; ++i) {
		if ( (block_offset = first_1bit(fs->super[i])) < 32 ) {
			my_strcpy(fs->files[fp].fname, fname);
			fs->files[fp].starting_block = i * 32 + block_offset;
			fs->files[fp].fsize = 0;
			fs->files[fp].btime = ++gtime;
			fs->super[i] ^= (1 << block_offset);
			++fs->nfiles;
			return fp;
		}
	}

	// cannot find empty block
	printf("[fs_open] : Failed. Cannot find any empty block!\n");
	return EMPTY;

}

__device__ void fs_read(FileSystem *fs, uchar* output, u32 size, u32 fp) {
	if (fs->files[fp].starting_block == EMPTY)
		printf("[fs_read] : fp %d does not exist.\n", fp);
	else {
		size = (size <= fs->files[fp].fsize)? size : fs->files[fp].fsize;
		auto starting_byte = fs->files[fp].starting_block * DATA_BLOCK_SIZE;
		memcpy(output, &fs->data[starting_byte], size);
		printf("[fs_read] : %d bytes of file \"%s\" read to output buffer.\n", size, fs->files[fp].fname);
		fs->files[fp].btime = ++gtime;
	}
}

// Assume fp exists.
__device__ void fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp) {
	
	if (size > MAX_FILE_SIZE) {
		printf("[fs_write] : File size limit exceeded (%d > %d)",
		size, MAX_FILE_SIZE);
		return;
	}

	// old starting block
	auto starting_block = fs->files[fp].starting_block;
	// old number of blocks occupied
	auto old_blocks = bytes2blocks(fs->files[fp].fsize);
	// new number of blocks required
	auto new_blocks = bytes2blocks(size);

	// free old blocks temporarily.
	fs->super[starting_block / 32] ^= bitmask(starting_block % 32, old_blocks);
	// need to find larger unoccupied blocks to write
	if (new_blocks > old_blocks) {
		int offset;
		for (int i = 0; i < N_SUPERBLOCKS; ++i) {
			offset = first_n_1bit(fs->super[i], new_blocks);
			if (offset != -1) {
				starting_block = i * 32 + offset;
				break;
			}
		}
		// failed to find larger unoccupied blocks, resume super status and return.
		if (offset == -1) {
			printf("[fs_write] : Failed to write to fp %d. Not enough data blocks!\n", fp);
			fs->super[starting_block / 32] ^= bitmask(starting_block % 32, old_blocks);
			return;
		}
	}

	// found suitable starting block to write, rewrite data
	printf("[fs_write] : %d bytes written to file \"%s\"!\n", size, fs->files[fp].fname);
	memcpy(&fs->data[starting_block * DATA_BLOCK_SIZE], input, size);

	// update file info
	fs->files[fp].starting_block = starting_block;
	fs->files[fp].fsize = size;
	fs->files[fp].btime = ++gtime;

	// update fs: flip occupancy bits of new blocks and increment file counter.
	fs->super[starting_block / 32] ^= bitmask(starting_block % 32, new_blocks);
}

__device__ void fp_bubble_sort(FileSystem* fs, int* fp, int n, 
							   int(*cmp)(const File*, const File*)) {
	auto* files = fs->files;
	bool swapped;
	for (int i = 0; i < n - 1; ++i) {
		for (int j = 0; j < n - i - 1; ++j) {
			auto* l = &files[fp[j]];
			auto* r = &files[fp[j+1]];
			if (cmp(l, r) > 0 || (cmp(l, r) == 0 && l->btime < r->btime)) {
				swap(&fp[j], &fp[j+1]);
				swapped = true;
			}
		}
		if (!swapped) return;
	}
}

__device__ void fs_gsys(FileSystem* fs, int op) {
	/* Operations encoded by following op's
	 *	LS_D: list files by Time (latest first)
	 *	LS_S: list files by Size (largest first)
	 *	LS_N: list files by Name (lexical order)
	 *	LS_F: list files by fp (smalledst first) */

	auto fcount = 0;
	int nfiles = fs->nfiles;
	printf("\n%d file%c\t\tGtime: %d\n",
			nfiles, nfiles > 1? 's' : '\0', gtime);

	// obtain array of all current fp
	int* fp = new int[nfiles];
	for (int i = 0; i < FCB_ENTRIES && fcount < nfiles; ++i) {
		if (fs->files[i].starting_block != EMPTY)
			fp[fcount++] = i;
	}

	// bubble sort the fp array according to specified op
	switch (op) {
		case LS_D:
			print_centered("Sort by Time", 60, '-');
			printf("%-20s %-10s %-10s %-10s %-10s\n",
				   "Name", "fp", "Size", "Time*", "Blocks");
			fp_bubble_sort(fs, fp, nfiles, [](auto* f1, auto* f2) {
						   return int(f2->btime - f1->btime);});
			break;
		case LS_S:
			print_centered("Sort by Size", 60, '-');
			printf("%-20s %-10s %-10s %-10s %-10s\n",
				   "Name", "fp", "Size*", "Time", "Blocks");
			fp_bubble_sort(fs, fp, nfiles, [](auto* f1, auto* f2) {
						   return int(f2->fsize - f1->fsize);});
			break;
		case LS_N:
			print_centered("Sort by Name", 60, '-');
			printf("%-20s %-10s %-10s %-10s %-10s\n",
				   "Name*", "fp", "Size", "Time", "Blocks");
			fp_bubble_sort(fs, fp, nfiles, [](auto* f1, auto* f2) {
						   return my_strcmp(f1->fname, f2->fname);});
			break;
		case LS_F:
			print_centered("Sort by fp", 60, '-');
			printf("%-20s %-10s %-10s %-10s %-10s\n",
				   "Name", "fp*", "Size", "Time", "Blocks");
			break;
		default: 
			printf("[fs_gsys] : Invalid operation! (%d)\n", op); 
			delete[] fp; return;
	}

	File* file;
	for (int i = 0; i < nfiles; ++i) {
		file = &fs->files[fp[i]];
		printf("%-20s %-10d %-10d %-10d %d-%d\n",
			file->fname, fp[i],
			file->fsize,
			file->btime,
			file->starting_block,
			file->starting_block
			+ bytes2blocks(file->fsize) - 1);
	}
	printf("\n");

	delete[] fp;
}

__device__ void fs_gsys(FileSystem* fs, int op, const char fname[]) {
	/* Operations encoded by following op's
	 *	RM: remove a file */

	switch (op) {
		case RM:
			u32 fp;
			if ( (fp = fs_find_name(fs, fname)) != EMPTY ) {
				printf("[fs_gsys] : File \"%s\" removed!\n", fname);
				auto staring_block = fs->files[fp].starting_block;
				auto blocks = bytes2blocks(fs->files[fp].fsize);
				// deactivate fp
				fs->files[fp].starting_block = EMPTY;
				// update fs: free blocks in super and decrement file counter
				fs->super[staring_block / 32] ^= bitmask(staring_block % 32, blocks);
				fs->nfiles--;			
			} else 	// file doesn't exist.
				printf("[fs_gsys] : file to remove \"%s\" doesn't exist!\n", fname);
			break;

		default: 
			printf("[fs_gsys] : Invalid operation! (%d)\n", op); 
			return;
	}

}
