/* includes */

#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>


/* marcros */

/* Copyright (c) 1982, 1986, 1989 The Regents of the University of California.
 * All rights reserved.
 *
 * Macros to test the exit status returned by wait
 * and extract the relevant values.
 *
 *	@(#)wait.h	7.17 (Berkeley) 6/19/91
 */
#ifdef _POSIX_SOURCE
#define	_W_INT(i)	(i)
#else
#define	_W_INT(w)	(*(int *)&(w))	/* convert union wait to int */
#define	WCOREFLAG	0200
#endif

#define	_WSTATUS(x)	(_W_INT(x) & 0177)
#define	_WSTOPPED	0177		/* _WSTATUS if process is stopped */
#define WIFSTOPPED(x)	(_WSTATUS(x) == _WSTOPPED)
#define WSTOPSIG(x)	(_W_INT(x) >> 8)
#define WIFSIGNALED(x)	(_WSTATUS(x) != _WSTOPPED && _WSTATUS(x) != 0)
#define WTERMSIG(x)	(_WSTATUS(x))
#define WIFEXITED(x)	(_WSTATUS(x) == 0)
#define WEXITSTATUS(x)	(_W_INT(x) >> 8)
#ifndef _POSIX_SOURCE
#define WCOREDUMP(x)	(_W_INT(x) & WCOREFLAG)

#define	W_EXITCODE(ret, sig)	((ret) << 8 | (sig))
#define	W_STOPCODE(sig)		((sig) << 8 | _WSTOPPED)
#endif


/* global variables */

static struct task_struct* task;
int status;
pid_t pid;


/* structs */

struct wait_opts {
	enum pid_type		wo_type;
	int			wo_flags;
	struct pid		*wo_pid;

	struct siginfo __user	*wo_info;
	int __user		*wo_stat;
	struct rusage __user	*wo_rusage;

	wait_queue_t		child_wait;
	int			notask_error;
};


/* function prototypes */

int my_exec(void);
int my_fork(void* argc);
void my_wait(pid_t pid, int* stat, int flags);
static int __init program2_init(void);
static void __exit program2_exit(void);
extern long do_wait (struct wait_opts* wo);
extern long _do_fork(unsigned long clone_flags,
				     unsigned long stack_start,
				     unsigned long stack_size,
				     int __user *parent_tidptr,
				     int __user *child_tidptr,
				     unsigned long tls);
extern int do_execve(struct filename *filename,
					 const char __user *const __user *__argv,
					 const char __user *const __user *__envp);
extern struct filename* getname(const char __user * filename);


/* implementations */

void my_wait(pid_t pid, int* stat, int flags) {
	struct wait_opts wo;

	wo.wo_type = PIDTYPE_PID;
	wo.wo_pid = find_get_pid(pid);
	wo.wo_flags = flags;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*) stat;
	wo.wo_rusage = NULL;

	do_wait(&wo);
	put_pid(wo.wo_pid);

	return;
}

int my_exec(void) {
	int result;
	const char path[] = "/home/seed/work/assignment1/source/program2/test;";
	const char* const argv[] = {path, NULL, NULL};
	const char* const envp[] = {"HOME=/", "PATH=/sbin:/user/sbin:/bin:/usr/bin", NULL};
	struct filename* my_filename = getname(path);

	/* execute a test program in child process */
	printk("[program2] : I'm child, my pid = %d\n", pid);
	result = do_execve(my_filename, argv, envp);

	if (!result) return 0;
	do_exit(result);
}

int my_fork(void* argc){
	/* set default sigaction for current process */
	int i;
	struct k_sigaction* k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);
	printk("[program2] : I'm parent, my pid = %d\n",  current->pid);
	printk("[program2] : I'm parent, my child has pid = %d\n", pid);

	/* wait until child process terminates */
	printk("[program2] : Parent waiting for the SIGCHLD signal...\n");
	my_wait(pid, &status, WEXITED | WUNTRACED);

	if (WIFEXITED(status)) {
		printk("[program2] : Child exited with EXIT STATUS = %d\n", WEXITSTATUS(status));
	}
	else if (WIFSIGNALED(status)) {
		printk  ("[program2] : CHILD EXECUTION TERMINATED BY SIGNAL: %d\n", WTERMSIG(status));
		if      (WTERMSIG(status) == SIGBUS ) printk("bus error\n");
		else if (WTERMSIG(status) == SIGFPE ) printk("floating point exception\n");
		else if (WTERMSIG(status) == SIGHUP ) printk("hangup\n");
		else if (WTERMSIG(status) == SIGILL ) printk("illegal instruction\n");
		else if (WTERMSIG(status) == SIGINT ) printk("interrupt\n");
		else if (WTERMSIG(status) == SIGALRM) printk("alarm clock\n");
		else if (WTERMSIG(status) == SIGABRT) printk("abort\n");
		else if (WTERMSIG(status) == SIGKILL) printk("kill\n");
		else if (WTERMSIG(status) == SIGPIPE) printk("pipe error\n");
		else if (WTERMSIG(status) == SIGQUIT) printk("quit\n");
		else if (WTERMSIG(status) == SIGSEGV) printk("segmentation violation\n");
		else if (WTERMSIG(status) == SIGTERM) printk("software termination\n");
		else if (WTERMSIG(status) == SIGTRAP) printk("trace trap\n");
	}
	else if (WIFSTOPPED(status)) {
		printk("[program2] : CHILD EXECUTION STOPPED BY SIGNAL: %d\n", WSTOPSIG(status));
	}
	else {
		printk("[program2] : CHILD PROCESS CONTINUED\n");
	}

	return 0;
}

static int __init program2_init(void){

	printk("[program2] : module_init\n");

	/* create a kernel thread to run my_fork */
	printk("[program2] : module_init create kthread starts\n");
	task = kthread_create(&my_fork, NULL, "MyForkThread");

	/* wake up the kernel thread */
	if(!IS_ERR(task)) {
		printk("[program2] : module_init kthread starts\n");
		wake_up_process(task);
	}

	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : module_exit\n");
}

MODULE_LICENSE("GPL");
module_init(program2_init);
module_exit(program2_exit);
