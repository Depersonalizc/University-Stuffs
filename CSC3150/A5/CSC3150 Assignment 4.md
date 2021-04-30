# CSC3150 Assignment 4

CHEN Ang (*118010009*)

In this assignment, we are asked to create a device in Linux that computes simple arithmetic expressions and desired prime numbers. Supporting driver functions for the device, including I/O control and R/W are implemented in a kernel module. Kernel memory is allocated to simulate registers on the device.

## How to Run the Program

The following commands assume the user already has `sudo` permissions.

Inside the `source` folder, type in the terminal

```shell
make
```

to compile the device driver module, the test program, and insert the module `mydev.ko` into the kernel. The terminal should then display part of the kernel log. Find the last line which reads

```shell
OS_AS5:init_modules(): registering chrdev([MAJOR],[MINOR])
```

To build the file node, type in the terminal

```
./mkdev.sh [MAJOR] [MINOR]
```

By then you can use the test program to test the device. Type in the terminal

```shell
./test
```

to start testing. After the test, you can type in

```shell
make clean
```

to remove `mydev.ko` from the kernel and have a screenshot of the kernel log. 

Finally to remove the file node, type in

```
./rmdev.sh
```

## Program Design

### Module Initialization (`init_modules`)

Upon initialization, The kernel module first asks the system to reserve a range of device numbers by invoking `alloc_chrdev_region(dev)`. It then obtains a set of major and minor device numbers using `MAJOR(dev)` and `MINOR(dev)` macros, and `printk` the obtained device numbers to the kernel logs. 

For the device to work, we need to initialize a `cdev` `struct`:

```c++
dev_cdev = cdev_alloc();
dev_cdev->ops = &fops;
dev_cdev->owner = THIS_MODULE;
```

Here `fops` is a `file_operations` `struct` to bind user-end file operations with kernel-end functions implemented:

```c
static struct file_operations fops = {
//    user           : kernel
      owner          : THIS_MODULE,
      read           : drv_read,
      write          : drv_write,
      unlocked_ioctl : drv_ioctl,
      open           : drv_open,
      release        : drv_release,
};
```

We can then make the device live, using

```c
cdev_add(dev_cdev, dev, 1)
```

For later work, we allocate a Direct Memory Access (DMA) buffer `dma_buf` to simulate the registers on a physical device. We also allocate enough space for the work routine, both using `kmalloc` at the end of the module initialization.

### Read Operation (`drv_read`)

The `read` operation is bound with the `drv_read` kernel function. The implementation of `drv_write` is rather straightforward. One thing to note is that since data needs to be transferred between user and kernel space, the `put_user` function is invoked to place the answer of the expression at `dma_buf[DMAANSADDR]` to the user-end `buffer`. After this the answer in `dma_buf` is cleared with zero bits and the readability status `dma_buf[DMAREADBLEADDR]` is set to unreadable (`0`).

### Write Operation (`drv_write`)

The `write` operation is bound with the `drv_write` kernel function. Our device only handles arithmetic expressions, and we assume the input `buffer` from the user is organized in the following way:

```c
typedef struct expr {
    char a;
    int b;
    short c;
} expr_t;
```

The expression is then copied to the corresponding locations in `dma_buf`:

```c
get_user( *(char *)(dma_buf+DMAOPCODEADDR), (char*)buffer);
get_user( *(int  *)(dma_buf+DMAOPERANDBADDR), &(data->b) );
get_user( *(short*)(dma_buf+DMAOPERANDCADDR), &(data->c) );
```

Until then, we are ready to enqueue the `drv_arithmetic_routine` to the `work_q` with

```c
INIT_WORK(work_q, drv_arithmetic_routine);
schedule_work(work_q);
```

Now is a critical moment with regard to the behavior of `drv_write`, namely, either it uses blocking I/O or non-blocking one. If the user has chosen blocking I/O, `drv_write` needs to wait for the work in the `work_q` to finish before proceeding. However if the user has chosen non-blocking I/O, `drv_write` immediately returns. The `flush_scheduled_work()` function can thus be of our use:

```c
// BLOCKING I/O: Wait until all work flushed
if (myini(DMABLOCKADDR) == 1) {
    printk("%s:%s(): blocking\n", PREFIX_TITLE, __func__);
    flush_scheduled_work();
}
return 0;
```

### I/O Control (`drv_ioctl`)

The user uses `ioctl` to control the I/O behavior of the device. In particular, the user can pass different command to either set entries in `dma_buf` (e.g., to set non/blocking IO with `HW5_IOCSETBLOCK`), or wait for the device to respond with an interrupt signal when the answer is readable (`HW_IOCWAITREADABLE`). This behavior is implemented with an indefinite loop:

```c
while( myini(DMAREADABLEADDR) != 1 )
	msleep(1000);
```

After the sleep is finished, we send the signal of `READABLE` to user by

```c
put_user(1, (unsigned int*)arg)
```

### Arithmetic Routine (`drv_arithmetic_routine`)

In `drv_arithmetic_routine`, the module evaluates the expression stored in `dma_buf` and places the answer at `dma_buf[DMAANSADDR]`. Since the answer is then ready to read, we also need to set the readability status to `READABLE` (`1`) in case of non-blocking I/O.

### Module Exit (`exit_modules`)

When exiting, the module

- frees the IRQ via `free_irq` and `printk()` the interrupt count
- unregisters the device via `unregister_chrdev_region`
- deletes `cdev` structs by invoking `cdev_del`
- `kfree` the `dma_buf` and `work_routine`

### Bonus

We first use

```shell
watch -n 1 cat /proc/interrupts
```

to find out the IRQ number for Keyboard is `1`.

To count the number of interrupts from Keyboard, we make use of `request_irq` to add an interrupt handler named `irq_handler` which increments a global `count` every time we have an interrupt request from Keyboard. We remove the handler with `free_irq` and `printk` the interrupt number at module exit.

## Problems I met

This project went smooth overall and most difficulties I met were minor, e.g., some typos in the code breaking the program which were fairly easy to debug.

## Screenshots

- Kernel log after insertion of the driver module and creation of device file node

  <img src="H:\University-Stuffs\CSC3150\A5\微信图片_202012030319251.png" alt="微信图片_202012030319251" style="zoom:50%;" />

- Output of the test program to the user

  <img src="H:\University-Stuffs\CSC3150\A5\微信图片_20201203031925.png" alt="微信图片_20201203031925" style="zoom: 33%;" />

- Kernel log during the execution of the test program

  <img src="H:\University-Stuffs\CSC3150\A5\微信图片_202012030319253.png" alt="微信图片_202012030319253" style="zoom:50%;" />

- Kernel log after removal of the kernel module.

  <img src="H:\University-Stuffs\CSC3150\A5\微信图片_202012030319252.png" alt="微信图片_202012030319252" style="zoom:50%;" />


## What I learned

I learned the basic working principles of a device driver in Linux and how I can implement it on my own. I get to know the difference between blocking and non-blocking I/O modes. I learned what a DMA buffer is. I also learned how to add an interrupt handler for an IRQ number.
