
#ifndef MONITOR_H
#define MONITOR_H

#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <dlfcn.h>
#include <setjmp.h>
#include <time.h>
#include <malloc.h>
#include "thread_record.cuh"
#include "common.cuh"
//#endif

namespace gmod{

__device__ static void* transmitter(void*);
__device__ static int processNode(unsigned long* userAddr);
__device__ void* transmitter() {
	unsigned long* userAddr = 0;
	while (true) {
#ifdef GUARDTIME
	execution[threadIdx.x]++;
#endif
		g_threadrecordlist[threadIdx.x]->addressC->isreorganize();
		if (reorganizeFlag) {
			__syncthreads();
			g_threadrecordlist[threadIdx.x]->addressC->reorganize();

			if(g_exit_procedure[threadIdx.x/32] == TRANSMITTER_BEGIN)
				g_exit_procedure[threadIdx.x/32]  = EXIT_HOOKED;
		} else {
			while ((userAddr = g_threadrecordlist[threadIdx.x]->consume())) {
				processNode(userAddr);
			}
			g_threadrecordlist[threadIdx.x]->addressC->resetDecodeIndex();

		}
		if (g_exit_procedure[threadIdx.x/32]  == EXIT_HOOKED) {
			g_exit_procedure[threadIdx.x/32]  = TRANSMITTER_BEGIN;
			continue;
		} else if (g_exit_procedure[threadIdx.x/32]  == TRANSMITTER_BEGIN) {
			g_exit_procedure[threadIdx.x/32]  = TRANSMITTER_DONE;
			break;
		}
	}
#ifdef RADIO
	/***********************************************get byte*************************************************************/
	printf("threadIdx %d,index=%d,id_index\n",threadIdx.x,g_threadrecordlist[threadIdx.x]->addressC->index_->index,
			g_threadrecordlist[threadIdx.x]->addressC->index_->id_index);
	/***************************************************************************************************************/
#endif
	return NULL;
}
__device__ void* transmitter_max_guard(long long int* duration) {
	unsigned long* userAddr = 0;
	duration[threadIdx.x+blockIdx.x*blockDim.x]=0;
	while (true) {
#ifdef GUARDTIME
	execution[threadIdx.x]++;
#endif
		g_threadrecordlist[threadIdx.x]->addressC->isreorganize();
		if (reorganizeFlag) {
			__syncthreads();
			g_threadrecordlist[threadIdx.x]->addressC->reorganize();
			if(g_exit_procedure[threadIdx.x/32] == TRANSMITTER_BEGIN)
				g_exit_procedure[threadIdx.x/32]  = EXIT_HOOKED;
		} else {
			long long int start=clock64();
			while ((userAddr = g_threadrecordlist[threadIdx.x]->consume())) {

				processNode(userAddr);
			}
			g_threadrecordlist[threadIdx.x]->addressC->resetDecodeIndex();
			long long int end=clock64();
			if(duration[threadIdx.x+blockIdx.x*blockDim.x]<(end-start)){
				duration[threadIdx.x+blockIdx.x*blockDim.x]=end-start;
			}
		}
		if (g_exit_procedure[threadIdx.x/32]  == EXIT_HOOKED) {
				g_exit_procedure[threadIdx.x/32]  = TRANSMITTER_BEGIN;
				continue;
			} else if (g_exit_procedure[threadIdx.x/32]  == TRANSMITTER_BEGIN) {
				g_exit_procedure[threadIdx.x/32]  = TRANSMITTER_DONE;
				break;
			}
	}
#ifdef RADIO
	/***********************************************get byte*************************************************************/
	printf("threadIdx %d,index=%d,id_index\n",threadIdx.x,g_threadrecordlist[threadIdx.x]->addressC->index_->index,
			g_threadrecordlist[threadIdx.x]->addressC->index_->id_index);
	/***************************************************************************************************************/
#endif
	return NULL;
}
__device__ static void attackDetected(void *user_addr, int reason) {
	switch (reason) {
	case 0:
		printf("\nError: When monitor thread checks user chunk,\n");
		break;
	case 1:
		printf("\nError: When free call checks user chunk,\n");
		break;
	case 2:
		printf("\nError: When realloc call checks user chunk,\n");
		break;
	case 3:
		printf("\nError: When realloc executes CAS,\n");
		break;
	}
	printf("buffer overflow is detected at user address %p\n", user_addr);
	switch (g_pro_attack) {
	case TO_ABORT:
		printf("The process is going to abort due to an attack...\n");
		asm("trap;");
		break;
	case TO_EXIT:
		printf("The process is going to exit due to an attack...\n");
		asm("trap;");
		break;
	case TO_GOON:
	default:
		break;
	}
}


__device__ int processNode(unsigned long* userAddr) {
	unsigned long volatile *p = (unsigned long*) userAddr - 2;

	unsigned long volatile canary_left = p[0];
	size_t word_size = p[1];
	unsigned long expected_canary = (g_canary ^ word_size);
	unsigned long canary_free = (g_canary_free ^ word_size);
	if(canary_left == canary_free){
			if( p[2 + word_size] != expected_canary ){
				attackDetected(userAddr, 0);
			}
			g_threadrecordlist[threadIdx.x]->addressC->signZero();
		free((unsigned long*)p);
			return 3;
		}
	if( canary_left != expected_canary ||
		(p[2 + word_size]) != expected_canary ){
		attackDetected(userAddr, 0);
	}
	return 1;
}
}
#endif
