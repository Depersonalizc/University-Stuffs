obj-m	:= program2.o
KVERSION := $(shell uname -r)
PWD	:= $(shell pwd)

# CFILES:= $(shell ls|grep test.c)
# PROGS:=$(patsubst %.c,%,$(CFILES))


all: program2.c test.c
	$(CC) test.c -o test
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules
# 	$(PROGS)

# %:%.c
# 	$(CC) -o $@ $<

clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean
	rm test

# 	$(PROGS)
# 	rm $(PROGS)
