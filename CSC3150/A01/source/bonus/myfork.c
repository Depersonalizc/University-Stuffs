#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>


int main(int argc,char *argv[]){
	pid_t root, pid;
	int status;
	int i, j;

	/* copy arguments */
	char* arg[argc];
	for (i = 0; i < argc - 1; i++) {
		arg[i] = argv[i + 1];
	}
	arg[argc - 1] = NULL;

	/* initialzation */
	root = getpid();
	pid = 0;

	/* checking error case */
	if (argc == 1) {
		printf("ERROR: Please specify programs as input arguments!\n");
		exit(1);
	}

	/* start forking! */
	printf("ROOT %d FORKING...\n", root);
	for (j = 0; j < argc; j++) {

		/* child or root process */
		if (pid == 0) {

			/* reached last child */
			if (j == argc - 1) {
				printf("REACHED END OF THE FORKING TREE! STARTING EXECUTION...\n");
				printf("------------------------------------------------------\n");
				printf("Child %d executing program %d\n", getpid(), j);
				execve(arg[j-1], arg, NULL);
			}

			/* not yet last child. fork! */
			pid = fork();

			/* child immediately becomes a parent after forking */
			if (pid > 0) goto parent;
		}

		/* parent process */
		else {
			parent:
			printf("%d -> %d\n", getpid(), pid);
			waitpid(pid, &status, WUNTRACED);

			/* check child process' termination status */
			if (WIFEXITED(status)) {
				printf("Child %d exited with EXIT STATUS = %d\n", pid, WEXITSTATUS(status));
			}
			else if (WIFSIGNALED(status)) {
				printf("CHILD %d EXECUTION TERMINATED BY SIGNAL: %d\n", pid, WTERMSIG(status));
				if      (WTERMSIG(status) == SIGBUS ) printf("bus error\n");
				else if (WTERMSIG(status) == SIGFPE ) printf("floating point exception\n");
				else if (WTERMSIG(status) == SIGHUP ) printf("hangup\n");
				else if (WTERMSIG(status) == SIGILL ) printf("illegal instruction\n");
				else if (WTERMSIG(status) == SIGINT ) printf("interrupt\n");
				else if (WTERMSIG(status) == SIGALRM) printf("alarm clock\n");
				else if (WTERMSIG(status) == SIGABRT) printf("abort\n");
				else if (WTERMSIG(status) == SIGKILL) printf("kill\n");
				else if (WTERMSIG(status) == SIGPIPE) printf("pipe error\n");
				else if (WTERMSIG(status) == SIGQUIT) printf("quit\n");
				else if (WTERMSIG(status) == SIGSEGV) printf("segmentation violation\n");
				else if (WTERMSIG(status) == SIGTERM) printf("software termination\n");
				else if (WTERMSIG(status) == SIGTRAP) printf("trace trap\n");
			}
			else if (WIFSTOPPED(status)) {
				printf("CHILD %d EXECUTION STOPPED BY SIGNAL: %d\n", pid, WSTOPSIG(status));
			}
			else {
				printf("CHILD %d CONTINUED\n", pid);
			}

			/* waiting ends here. start execution */
			if (getpid() == root) {
				printf("ROOT %d EXITING...\n", root);
				exit(0);
			}
			printf("Child %d executing program %d\n", getpid(), j);
			execve(arg[j-1], arg, NULL);
		}
	}

	return 0;
}
