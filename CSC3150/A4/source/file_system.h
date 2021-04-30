#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

#define EMPTY 0xffffffff
#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define LS_F 2
#define LS_N 3
#define RM 4

#define N_SUPERBLOCKS (1 << 10) // N_DATA_BLOCKS / 32(uint32_t)/SUPERBLOCK = 1 K
#define FCB_ENTRIES (1 << 10)	// 1K entries
#define DATA_SIZE (1 << 20) // 1024 KB
#define DATA_BLOCK_SIZE (1 << 5) // 32 bytes
// #define N_DATA_BLOCKS (1 << 15) // DATA_SIZE / DATA_BLOCK_SIZE = 32K

#define MAX_FILENAME_SIZE 20
#define MAX_FILE_NUM (1 << 10)	// 1 K == 1024
#define MAX_FILE_SIZE (1 << 10) // 1 KB == sizeof 1 superblock

struct File {						/* File Control Block			32 bytes */
	char fname[MAX_FILENAME_SIZE];	/* file name (null terminated)	20 bytes */
	u32  starting_block;			/* starting data block number	4  bytes */
	u32  fsize;						/* file size in bytes			4  bytes */
	u32  btime;						/* last modified time			4  bytes */
};

struct FileSystem {					/* File System (Volume)			1060 KB  */
	u32   super[N_SUPERBLOCKS];		/* block occupancy	 			4  	 KB  */
	File  files[FCB_ENTRIES];		/* 1K FCB's						32 	 KB  */
	uchar data[DATA_SIZE];			/* file contents in 32K blocks	1024 KB  */
	u32	  nfiles;
};

/* Utilities */
__device__ u32 bytes2blocks(u32 bytes);
__device__ int first_1bit(int x);
__device__ int first_n_1bit(u32 x, int n);
__device__ u32 bitmask(int start, int length);
__device__ char* my_strcpy(char* dest, const char* src);
__device__ int my_strcmp(const char* s1, const char* s2);
__device__ void swap(int* x, int* y);

/* File System */
__device__ void fs_init(FileSystem* fs);
__device__ u32  fs_open(FileSystem* fs, const char fname[], int op);
__device__ void fs_read(FileSystem* fs, uchar* output, u32 size, u32 fp);
__device__ void fs_write(FileSystem* fs, uchar* input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem* fs, int op);
__device__ void fs_gsys(FileSystem* fs, int op, const char fname[]);


#endif
