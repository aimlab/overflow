
#ifndef MONITOR_H
#define MONITOR_H

#include <unistd.h> //sleep
#include <signal.h> //sigaction
#include <errno.h> //program_invocation_name, ESRCH
#include <dlfcn.h> //dlsym
#include <setjmp.h> //siglongjmp
#include <time.h> //nanosleep
#include <malloc.h> 
#include "thread_record.h"
#include "list.h"

namespace cruiser{
static void* monitor(void *);
static void* transmitter(void*);
static int processNode(const CruiserNode &);
void* monitor(void *){ 
#ifdef PROTECT_TIME
long protect_number=0;
double	start, finish;
start = GetMSTime();
#endif

	if(!g_nodeContainer){
	g_nodeContainer = new List;
	}
	while(g_initialized != 2)
		sleep(0);
	#define SLEEP_CONDITION 10
	while((g_delayedBufferCount = 0, g_nodeContainer->traverse(processNode))){ 
#ifdef PROTECT_TIME
		protect_number++;
#endif

		if(g_exit_procedure == EXIT_HOOKED){
			g_exit_procedure = MONITOR_BEGIN;
			continue;
		}else if(g_exit_procedure == MONITOR_BEGIN){
			g_exit_procedure = MONITOR_DONE;
			break;
		}
	}

#ifdef PROTECT_TIME
	finish = GetMSTime();
	printf("total protect time %.2f ms, total protect nubmer %ld \n", (finish - start), protect_number);
	printf("average protect time : %.3f us\n", ((finish - start)*1000)/protect_number );
#endif
	return NULL;
}

static void attackDetected(void *user_addr, int reason){
	switch(reason){
		case 0: 
			fprintf(stderr, "\nError: When monitor thread checks user chunk,\n");
			break;
		case 1:
			fprintf(stderr, "\nError: When free call checks user chunk,\n");
			break;
		case 2:
			fprintf(stderr, "\nError: When realloc call checks user chunk,\n");
			break;
		case 3:
			fprintf(stderr, "\nError: When realloc executes CAS,\n");
			break;
	}
	fprintf(stderr, "buffer overflow is detected at user address %p\n", user_addr);
	switch(g_pro_attack){
		case TO_ABORT:
			fprintf(stderr, "The process is going to abort due to an attack...\n");
			abort();
			break;
		case TO_EXIT:
			//todo: it would be better if we traverse the list for more attack detection before exit
			fprintf(stderr, "The process is going to exit due to an attack...\n");
			exit(-1);
			break;
		case TO_GOON:
		default:
			break;
	}
}



int processNode(const CruiserNode & node){
	

	void *addr = node.userAddr;

#ifdef PRINT_ADDR
	printf("police %p\n", addr);
#endif
	if(__builtin_expect(!addr, 0)) // Dummy node
		return 2;
	
#ifdef PAGE_FAULT
	unsigned long volatile *p = (unsigned long*)(addr) - 2;
#else
	unsigned long volatile *p = (unsigned long*)(addr) - (EXTRA_WORDS/2);
#endif
	unsigned long volatile canary_left = p[0];
	volatile size_t word_size = p[1];
	if(p[0] != canary_left) 
		return 1;
	
	unsigned long expected_canary = (g_canary ^ word_size);//^ (unsigned long)p;
	unsigned long canary_free = (g_canary_free ^ word_size);//^ (unsigned long)p;

#ifdef PAGE_FAULT
if(canary_left == canary_free){
		if( p[2 + word_size] != expected_canary ){		

			fprintf(stderr, "a buffer is overflowed then freed:\
				addr(user) %p, word_size=0x%lx, p[1]= 0x%lx, \
				p[0]= 0x%lx, p[end]=0x%lx, expected_canary=0x%lx\n", 
				addr, word_size, p[1], p[0], p[2 + word_size], expected_canary);
			attackDetected(addr, 0);
		}

#else
	if(canary_left == canary_free){
		if( p[(EXTRA_WORDS/2) + word_size] != expected_canary ){
			fprintf(stderr, "a buffer is overflowed then freed:\
				addr(user) %p, word_size=0x%lx, p[1]= 0x%lx, \
				p[0]= 0x%lx, p[end]=0x%lx, expected_canary=0x%lx\n", 
				addr, word_size, p[1], p[0], p[(EXTRA_WORDS/2) + word_size], expected_canary);
			attackDetected(addr, 0);
		}

#endif
		g_delayedBufferCount++;
		real_cuMemFree((CUdeviceptr)p);
		return 3;
	}
	unsigned long end = -1L;
#ifdef PAGE_FAULT
if( canary_left != expected_canary || 
		(end = p[2 + word_size]) != expected_canary ){
#else
	if( canary_left != expected_canary || 
		(end = p[(EXTRA_WORDS/2) + word_size]) != expected_canary ){
#endif

		fprintf(stderr, "Normal check, attack warning: addr(not user) %p, \
			word_size=0x%lx, canary_left=0x%lx, p[1]= 0x%lx, p[0]= 0x%lx, \
			p[end]=0x%lx (~0 means it is not assigned yet), expected_canary\
			=0x%lx, exptected_canary_free=0x%lx\n", 
			p, word_size, canary_left, p[1], p[0], end, expected_canary, 
			canary_free);
		attackDetected(addr, 0);
	}
	return 1;
}

}
#endif 
