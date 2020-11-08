# CSC3150 Assignment 3

CHEN Ang (*118010009*)

In this assignment, we are asked to implement a CUDA kernel function that simulates the mechanism of Virtual Memory (VM). We treat the Global (large-sized, high access latency) and Shared memories (small-sized, fast access speed) in the CUDA GPU as the External Storage and Physical Memory in a traditional CPU VM setup respectively. The Least Recently Used (LRU) page swapping algorithm is implemented via a counter method.

## Environment

OS: Windows 10

CUDA version: CUDA 10.2

GPU model: Nvidia GeForce GTX 970

## How to Run the Program

...

## Program Design

....

## Page Fault Number and Explanation

The page fault number outputted is `8193`. The `vm_write` in the first loop produces one page fault every time it comes across a new page, creating `128 KB / 32 B == 4096` page faults. The `vm_read` in the second loop starts from `i == input_size - 1` backwards for `32769 bytes == 32 KB + 1 byte`. The first `32 KB` access of bytes are through the page table. However the last byte is not, which produces `1` page fault. Finally the `vm_snapshot` function calls `vm_read` from the first byte to the `input_size`-th byte, which is effectively the same as `vm_write` in the first loop, creating an additional `4096` page faults. Combined, there are exactly `4096 + 1 + 4096 == 8193` page faults.

```C++
/* input_size == 131072 bytes == 128 KB == 4 * RAM */

for (int i = 0; i < input_size; i++)	// 4096
    vm_write(vm, i, input[i]);

for (int i = input_size - 1; i >= input_size - 32769; i--)	// 1
    int value = vm_read(vm, i);

vm_snapshot(vm, result, 0, input_size);	// 4096
```

## Problems I met



## Screenshots



## What I learned

