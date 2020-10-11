#include <unistd.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

int main(int argc,char* argv[]){
	printf("--------USER PROGRAM--------\n");
	int i=0;
	// abort();
	// sleep(5);
	raise(SIGBUS);
	printf("user process success!!\n");
	printf("--------USER PROGRAM--------\n");
	return 250;
}
