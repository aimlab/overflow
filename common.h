

#ifndef CRUISER_H
#define CRUISER_H
#include <cuda.h>
#include <pthread.h>
#include "utility.h"
#ifndef L1_CACHE_BYTES
#define L1_CACHE_BYTES  64 // Double check your system!!
#endif


namespace cruiser{
#ifdef PAGE_FAULT
 #define EXTRA_WORDS 3
#else
#define	EXTRA_WORDS 1024 
#endif



typedef CUresult CUDAAPI (*fnMemFree)(CUdeviceptr dptr);
typedef CUresult CUDAAPI (*fnMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (*fnMemAllocPitch) ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes );


extern "C" { void* __libc_dlsym (void *map, const char *name); }

#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

typedef void* (*fnDlsym)(void*, const char*);
static fnDlsym internal_dlsym=NULL;
static void* real_dlsym(void *handle, const char* symbol)
{
   if(__builtin_expect(!internal_dlsym,0))
   internal_dlsym = (fnDlsym)__libc_dlsym(dlopen("libdl.so.2", RTLD_LAZY), "dlsym");
   return (*internal_dlsym)(handle, symbol);
}

static fnMemFree  real_cuMemFree=NULL;
static fnMemAlloc  real_cuMemAlloc=NULL;
static fnMemAllocPitch real_cuMemAllocPitch=NULL;


struct CruiserNode{
	void			*userAddr;
};


class NodeContainer{
public:
	virtual ~NodeContainer(){}
	virtual bool insert(const CruiserNode &) = 0;
	virtual bool multi_thread_insert(const CruiserNode &) = 0;
	virtual int traverse( int (*pfn)(const CruiserNode &) ) = 0;
};


enum   EXIT_PROCEDURE		{RUNNING, EXIT_HOOKED, TRANSMITTER_BEGIN, 
				TRANSMITTER_DONE, MONITOR_BEGIN, MONITOR_DONE};


enum   PRO_ATTACK 		{TO_ABORT, TO_EXIT, TO_GOON};


static EXIT_PROCEDURE volatile 	g_exit_procedure = RUNNING;
static const PRO_ATTACK			g_pro_attack = TO_ABORT;
static int	volatile			g_initialized = 0;

static unsigned long 			g_canary; 

static unsigned long			g_canary_free;


static pthread_t 				g_monitor; // The monitor thread ID
static pthread_t				g_transmitter; 
static NodeContainer			*g_nodeContainer = NULL;

static unsigned					g_init_begin_time;
typedef cudaError_t					(*cudafree_type)(void*);
static volatile pid_t 			g_pid;

static unsigned int volatile	g_transmitter_still_count;

static unsigned					g_delayedBufferCount;


}

#endif 
