#ifndef UTILITY_H
#define UTILITY_H

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

namespace gmod{
#define THREADSUM BLOCKNUM * THREAD_PER_BLOCK
#define BLOCKNUM 10
#define DATANUM 2000000
/*#ifdef G_THREAD1
#define GUARD_THREADSUM 32
#elif G_THREAD2
#define GUARD_THREADSUM 64
#elif G_THREAD3
#define GUARD_THREADSUM 96
#elif G_THREAD4
#define GUARD_THREADSUM 128
#elif G_THREAD5
#define GUARD_THREADSUM 192
#elif G_THREAD6
#define GUARD_THREADSUM 256
#elif G_THREAD7
#define GUARD_THREADSUM 320
#else
#define GUARD_THREADSUM 384
#endif
*/
#define GUARD_THREADSUM 32
#define THREAD_PER_BLOCK 256
#define THREAD_PER_MALLOC 1
__device__ int reorgnize_num=DATANUM;

#ifdef GUARDTIME
__device__ int execution[GUARD_THREADSUM];
#endif	
}
#endif //UTILITY_H
