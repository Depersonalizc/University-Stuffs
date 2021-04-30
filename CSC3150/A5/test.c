#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "ioc_hw5.h"

typedef struct datain {
    char a;
    int b;
    short c;
} datain_t;

int isPrime(int x) {
    for (int i = 2; i <= x / 2; ++i) {
        if (x % i == 0)
            return 0;
    }
    return 1;
}

int prime(int base, short nth) {
    for (int fnd = -1; fnd < nth; fnd += isPrime(base++)) {}
    return --base;
}

int arithmetic(int fd, char operator, int operand1, short operand2) {

    int ans;
    int readable;
    int ret;
    datain_t data;

    data.a = operator;
    data.b = operand1;
    data.c = operand2;

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

    printf("\n\n***** %d %c %d = %d *****\n\n",
    		data.b, data.a, data.c, ans);

    /***************** Blocking IO *****************/
    printf("Blocking IO\n");

    ret = 1;
    if (ioctl(fd, HW5_IOCSETBLOCK, &ret) < 0) {
        printf("set blocking failed\n");
        return -1;
    }

    printf("Queuing work\n");
    write(fd, &data, sizeof(data));

    // wait until work queue flushed clean

    read(fd, &ret, sizeof(int));

    printf("ans=%d ret=%d\n\n", ans, ret);
    /***********************************************/



    /*************** Non-Blocking IO ***************/
    printf("Non-Blocking IO\n");

    ret = 0;
    if (ioctl(fd, HW5_IOCSETBLOCK, &ret) < 0) {
        printf("set non-blocking failed\n");
        return -1;
    }

    printf("Queuing work\n");
    write(fd, &data, sizeof(data));

    // Synchronization via interrupt
    printf("Waiting for interrupt\n");
    ioctl(fd, HW5_IOCWAITREADABLE, &readable);

    if (readable == 1) {
        printf("Interrupted! Reading result...\n");
        read(fd, &ret, sizeof(int));
    	printf("ans=%d ret=%d\n", ans, ret);
    }
    /***********************************************/

    printf("\n***** Test case finished *****\n\n");

    return ans;
}

int main()
{
    printf("...............Start...............\n");

    //open my char device:
    int fd = open("/dev/mydev", O_RDWR);
    if (fd == -1) {
        printf("can't open device!\n");
        return -1;
    }

    int ret;

    ret = 118010009;
    if (ioctl(fd, HW5_IOCSETSTUID, &ret) < 0) {
        printf("set stuid failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETRWOK, &ret) < 0) {
        printf("set rw failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETIOCOK, &ret) < 0) {
        printf("set ioc failed\n");
        return -1;
    }

    ret = 1;
    if (ioctl(fd, HW5_IOCSETIRQOK, &ret) < 0) {
        printf("set irq failed\n");
        return -1;
    }

    arithmetic(fd, '+', 100, 10);
    arithmetic(fd, '-', 100, 10);
    arithmetic(fd, '*', 100, 10);
    arithmetic(fd, '/', 100, 10);
    arithmetic(fd, 'p', 100, 10);
    arithmetic(fd, 'p', 100, 20000);


    printf("...............End...............\n");

    return 0;
}
