#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	int status;
	pid_t pid;

	/* fork a child process */
	pid = fork();

	if (pid == -1) {
		perror("fork");
		exit(1);
	}

	else {

		/* child process */
		if (pid == 0) {

			/* copy arguments */
			char* arg[argc];
			for (int i = 0; i < argc - 1; i++) {
				arg[i] = argv[i + 1];
			}
			arg[argc - 1] = NULL;

			printf("I'm the child process, my pid = %d\n", getpid());

			/* execute test program */
			execve(arg[0], arg, NULL);

		}
		
		/* parent process */
		else {
			
			printf("I'm the parent process, my pid = %d\n", getpid());

			/* wait for child process to terminate */
			printf("Parent process waiting for the SIGCHLD signal...\n");
			waitpid(pid, &status, WUNTRACED);

			/* check child process' termination status */
			if (WIFEXITED(status)) {
				printf("Child exited with EXIT STATUS = %d\n", WEXITSTATUS(status));
			}
			else if (WIFSIGNALED(status)) {
				printf("CHILD EXECUTION TERMINATED BY SIGNAL: %d\n", WTERMSIG(status));
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
				printf("CHILD EXECUTION STOPPED BY SIGNAL: %d\n", WSTOPSIG(status));
			}
			else {
				printf("CHILD PROCESS CONTINUED\n");
			}

			exit(0);

		}
	}

	return 0;
	
}
