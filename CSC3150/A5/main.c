#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

/* consts */
#define PREFIX_TITLE "OS_AS5"

/* device consts */
#define DEV_NAME "mydev"           // name for alloc_chrdev_region
#define DEV_BASEMINOR 0            // baseminor for alloc_chrdev_region
#define DEV_COUNT 1                // count for alloc_chrdev_region
#define IRQ_NO 1

/* DMA consts */
#define DMA_BUFSIZE     64
#define DMASTUIDADDR    0x0        // Student ID
#define DMARWOKADDR     0x4        // RW function complete
#define DMAIOCOKADDR    0x8        // ioctl function complete
#define DMAIRQOKADDR    0xc        // ISR function complete
#define DMACOUNTADDR    0x10       // interrupt count function complete
#define DMAANSADDR      0x14       // Computation answer
#define DMAREADABLEADDR 0x18       // READABLE variable for synchronize
#define DMABLOCKADDR    0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR   0x20       // data.a opcode
#define DMAOPERANDBADDR 0x21       // data.b operand1
#define DMAOPERANDCADDR 0x25       // data.c operand2

/* device variables */
static int dev_major;
static int dev_minor;
static struct cdev* dev_cdev;
// IRQ
static int count = 0;
static unsigned long dev_id;

/* DMA variables */
void* dma_buf;

/* Prototypes of file operations */
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int     drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int     drv_release(struct inode*, struct file*);
static long    drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// IO functions
void myoutc(unsigned char data, unsigned short int port);
void myouts(unsigned short data, unsigned short int port);
void myouti(unsigned int data, unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct* work_q;

// Input data struct
typedef struct expr {
    char a;
    int b;
    short c;
} expr_t;


// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

// Prime computation
int isPrime(int x);
int prime(int base, short nth);

// IRQ interrupt
static irqreturn_t irq_handler(int irq, void* dev_id) {
	if (irq == IRQ_NO)
		++count;
	return IRQ_NONE;
}


/* Functions implementation */

int isPrime(int x) {
	int i = 2;
    for (; i <= x / 2; ++i) {
        if (x % i == 0)
            return 0;
    }
    return 1;
}

int prime(int base, short nth) {
	int fnd = -1;
    for (; fnd < nth; fnd += isPrime(base++)) {}
    return --base;
}

// Input and output data from/to DMA
void myoutc(unsigned char data, unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}

void myouts(unsigned short data, unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}

void myouti(unsigned int data, unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}

unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}

unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}

unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}

static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    printk("%s:%s(): device closed\n", PREFIX_TITLE, __func__);
	return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	printk("%s:%s(): ans = %d\n", PREFIX_TITLE, __func__, myini(DMAANSADDR));
	put_user(myini(DMAANSADDR), (int*)buffer);
	myouti(0, DMAANSADDR);
	myouti(0, DMAREADABLEADDR);
	return 0;
}

static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	
	expr_t* data = (expr_t*)buffer;

	// Write user expression to dma buffer
	get_user( *(char *)(dma_buf+DMAOPCODEADDR), (char*)buffer);
	get_user( *(int  *)(dma_buf+DMAOPERANDBADDR), &(data->b) );
	get_user( *(short*)(dma_buf+DMAOPERANDCADDR), &(data->c) );

	// Enqueue work
	INIT_WORK(work_q, drv_arithmetic_routine);
	schedule_work(work_q);
	printk("%s:%s(): work enqueued\n", PREFIX_TITLE, __func__);

	// BLOCKING IO: Wait until all work flushed
	if (myini(DMABLOCKADDR) == 1) {
		printk("%s:%s(): blocking\n", PREFIX_TITLE, __func__);
		flush_scheduled_work();
	}

	return 0;
}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {

	unsigned int val;
	get_user(val, (unsigned int*)arg);

	switch (cmd) {
		case HW5_IOCSETSTUID:
			myouti(val, DMASTUIDADDR);
			printk("%s:%s(): STUID = %u\n", 
					PREFIX_TITLE, __func__, myini(DMASTUIDADDR));
			return 0;
		case HW5_IOCSETRWOK :
			myouti(val, DMARWOKADDR); 
			printk("%s:%s(): RW OK\n", PREFIX_TITLE, __func__);
			return 0;
		case HW5_IOCSETIOCOK:
			myouti(val, DMAIOCOKADDR);
			printk("%s:%s(): IOC OK\n", PREFIX_TITLE, __func__);
			return 0;
		case HW5_IOCSETIRQOK:
			myouti(val, DMAIRQOKADDR);
			printk("%s:%s(): IRQ OK\n", PREFIX_TITLE, __func__);
			return 0;
		case HW5_IOCSETBLOCK:
			myouti(val, DMABLOCKADDR);
			printk("%s:%s(): %s IO\n", PREFIX_TITLE, __func__,
					val? "Blocking" : "Non-Blocking");
			return 0;
		case HW5_IOCWAITREADABLE:
			// Sleep till READABLE is true
			while( myini(DMAREADABLEADDR) != 1 )
				msleep(1000);
			// Send interrupt to user
			printk("%s:%s(): readable now, sending interrupt to user\n", 
					PREFIX_TITLE, __func__);
			put_user(1, (unsigned int*)arg);
			return 0;
		default: return -EINVAL;
	}
}

static void drv_arithmetic_routine(struct work_struct* ws) {
	
	int ans;
	expr_t data;

	data.a = myinc(DMAOPCODEADDR);
    data.b = myini(DMAOPERANDBADDR);
    data.c = myins(DMAOPERANDCADDR);

    switch (data.a) {
        case '+':
            ans = data.b + data.c;
            break;
        case '-':
            ans = data.b - data.c;
            break;
        case '*':
            ans = data.b * data.c;
            break;
        case '/':
            ans = data.b / data.c;
            break;
        case 'p':
            ans = prime(data.b, data.c);
            break;
        default:
            ans = 0;
    }

    printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__,
    		data.b, data.a, data.c, ans);

    // put answer at ANSADDR
    myouti(ans, DMAANSADDR);

    // set READABLE to true for non-blocking IO
    if (myini(DMABLOCKADDR) == 0)
    	myouti(1, DMAREADABLEADDR);

}

static int __init init_modules(void) {

	int rt;
	dev_t dev;

	printk("%s:%s(): ...............Start...............\n", PREFIX_TITLE, __func__);

	/* bonus */
	rt = request_irq(IRQ_NO, irq_handler, IRQF_SHARED, "bonus", (void*)&dev_id);
	if (rt < 0) {
		printk(KERN_ALERT"Register irq failed!\n");
		return -1;
	} else
		printk("%s:%s(): request on irq %d succeeded with return %d\n",
				PREFIX_TITLE, __func__, IRQ_NO, rt);


	/* Register chrdev */ 
	if (alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0) {
		printk(KERN_ALERT"Register chrdev failed!\n");
		return -1;
    } else {
		printk("%s:%s(): registering chrdev(%i,%i)\n",
				PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
    }

	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);

	/* Init cdev and make it live */
	dev_cdev = cdev_alloc();
   	dev_cdev->ops = &fops;
    dev_cdev->owner = THIS_MODULE;

	if (cdev_add(dev_cdev, dev, 1) < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
   	}

	/* Allocate DMA buffer */
	printk("%s:%s(): allocating dma buffer\n", PREFIX_TITLE, __func__);
	dma_buf = kmalloc(DMA_BUFSIZE, GFP_KERNEL);

	/* Allocate work routine */
	work_q = kmalloc(sizeof(typeof(*work_q)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void) {

	/* Free IRQ */
	free_irq(IRQ_NO, (void*)&dev_id);
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, count);

	/* Free DMA buffer when exiting modules */
	kfree(dma_buf);
	printk("%s:%s(): dma buffer freed\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major, dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);
	printk("%s:%s(): chrdev unregistered\n", PREFIX_TITLE, __func__);

	/* Free work routine */
	kfree(work_q);
	printk("%s:%s(): ................End................\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
