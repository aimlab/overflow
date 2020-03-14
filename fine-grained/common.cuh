
#ifndef CRUISER_H
#define CRUISER_H

#include <pthread.h>
#include "utility.cuh"

namespace gmod{

#define	EXTRA_WORDS 3

enum   EXIT_PROCEDURE		{RUNNING, EXIT_HOOKED, TRANSMITTER_BEGIN, 
				TRANSMITTER_DONE, MONITOR_BEGIN, MONITOR_DONE};

enum   PRO_ATTACK 		{TO_ABORT, TO_EXIT, TO_GOON};


__device__ static EXIT_PROCEDURE volatile 	g_exit_procedure[GUARD_THREADSUM/32];
static const PRO_ATTACK			g_pro_attack = TO_ABORT;

__device__ static int	volatile			g_initialized = 0;
__device__ static int	volatile			init_f = 0;
__device__ static unsigned long 			g_canary=0xcccccccc;
__device__ static unsigned long			g_canary_free=0xfefefedd;
__device__ static unsigned long			g_canary_realloc=0x10101010;


static unsigned					g_init_begin_time;

__device__ static unsigned int volatile	g_transmitter_still_count;

}

#endif
