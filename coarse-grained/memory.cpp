
#include <cuda_runtime_api.h>
#include <math.h>
#include<fcntl.h>
#include <sys/mman.h> // mmap, munmap
#include <string.h>
#include "monitor.h"
#define CUMALLOC
namespace cruiser{

void beforeExit(void);
void  init(){
	if (g_initialized)
		return;
	g_initialized = 1; 
	g_exit_procedure = RUNNING;
	printf("begin init\n");
	
		g_canary = 0xcccccccc; //0x87654321;
		g_canary_free = 0xfefefedd; //0xfedcba98;
	int thread_ret = pthread_create(&g_monitor, NULL, monitor, NULL);
	if(thread_ret){
		fprintf(stderr, "Error: monitor thread cannote be created (%d)\n", 
				thread_ret);
		exit(-1);
	}

	if(atexit(beforeExit))
		fprintf(stderr, "Error: atexit(beforeExit) failed");
	g_initialized = 2;
}

#ifdef PAGE_FAULT
inline void afterMalloc(void* addr, size_t word_size){



	unsigned long *p = (unsigned long *)addr;

	p[1] = word_size;
	p[2 + word_size] = p[0] = (g_canary ^ word_size);// ^ (unsigned long)p;
	CruiserNode node;
	node.userAddr = p + 2;
	g_nodeContainer->insert(node);
}
#else
inline void afterMalloc(void* addr, size_t word_size){
	unsigned long *p = (unsigned long *)addr;

	p[1] = word_size;
	p[(EXTRA_WORDS/2) + word_size] = p[0] = (g_canary ^ word_size);// ^ (unsigned long)p;
	CruiserNode node;
	node.userAddr = p + (EXTRA_WORDS/2);

	cudaMemAdvise(p,(EXTRA_WORDS/2)*sizeof(long),cudaMemAdviseSetReadMostly,0);
	cudaMemAdvise(p+(EXTRA_WORDS/2)+word_size,sizeof(long)*(EXTRA_WORDS/2),cudaMemAdviseSetReadMostly,0);
	cudaMemPrefetchAsync(p+(EXTRA_WORDS/2),word_size*sizeof(long),0);
	g_nodeContainer->insert(node);

}
#endif //page_fault

inline static void beforeFree(void* addr){
#ifdef PAGE_FAULT
 unsigned long *p = (unsigned long*)addr - 2;
#else
	unsigned long *p = (unsigned long*)addr - (EXTRA_WORDS/2);
#endif



	if(__builtin_expect(p[0] == (g_canary_free ^ p[1]), 0)){
		fprintf(stderr, "Duplicate frees are detected\n");
		//todo: set error no.
		return;
	}
	p[0] ^= (g_canary ^ g_canary_free); 

}
#ifdef PAGE_FAULT
static CUresult CUDAAPI malloc_wrapper(void** p_addr, size_t size){
	if(__builtin_expect(!g_initialized, 0))
		init();
	size_t word_size = size / sizeof(long) + (size%sizeof(long)?1:0);
	cudaMallocManaged(p_addr,(word_size + 3) * sizeof(long));
#ifdef MEMORY
	printf("original memory : %ld B\n", size);
	printf("final memory : %ld B\n", (word_size + 2) * sizeof(long));
#endif
	if(__builtin_expect(!(*p_addr), 0)){
		printf("allocate memory failed\n");
		return CUDA_ERROR_INVALID_VALUE;
	}
	afterMalloc((*p_addr), word_size);
	(*p_addr)=(long*)(*p_addr)+2;
#ifdef PRINT_ADDR
	printf("original %p\n",(*p_addr));
#endif
	return CUDA_SUCCESS;
}

#else

static CUresult CUDAAPI malloc_wrapper(void** p_addr, size_t size){
	if(__builtin_expect(!g_initialized, 0))
		init();
	size_t word_size = size /sizeof(long) + (size%sizeof(long)?1:0);
	 word_size = word_size / (EXTRA_WORDS/2) + (word_size%(EXTRA_WORDS/2)?1:0);
	word_size = word_size*(EXTRA_WORDS/2);
	cudaMallocManaged(p_addr,(word_size + EXTRA_WORDS) * sizeof(long));
#ifdef MEMORY
	printf("original memory : %ld B\n", size);
	printf("final memory : %ld B\n", (word_size + EXTRA_WORDS) * sizeof(long));
#endif
	if(__builtin_expect(!(*p_addr), 0)){
		printf("allocate memory failed\n");
		return CUDA_ERROR_INVALID_VALUE;
	}
	afterMalloc((*p_addr), word_size);
	(*p_addr)=(long*)(*p_addr)+(EXTRA_WORDS/2);
#ifdef PRINT_ADDR
	printf("original %p\n",(*p_addr));
#endif
	return CUDA_SUCCESS;
}
#endif //page fault

static CUresult CUDAAPI free_wrapper(void* addr){
	//	fprintf(stderr, "end hooking cuMemFree.\n");

	if(__builtin_expect(!addr, 0)){
		fprintf(stderr, "Free NULL pointer.\n");
		return CUDA_SUCCESS;
		//return CUDA_ERROR_INVALID_VALUE;
	}
		

	beforeFree(addr);
	
	return CUDA_SUCCESS;
}

void  beforeExit(void){
	cruiser::g_exit_procedure = cruiser::EXIT_HOOKED;
	void* status;
	int ret = pthread_join(cruiser::g_monitor, &status );
	if( ret != 0 ){
	    printf("thread join error=%d\n",ret);
	 }

}
}

extern "C" {
CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes ){
	if(!cruiser::real_cuMemAllocPitch)
	fprintf(stderr, "begin hooking cuMemAllocPitch.\n");
	*pPitch=WidthInBytes;
	return  cruiser::malloc_wrapper((void**)(dptr),WidthInBytes*Height);
}
}
extern "C" {

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr){
if(!cruiser::real_cuMemFree){
fprintf(stderr, "begin hooking cuMemFree.\n");
cruiser::real_cuMemFree=(cruiser::fnMemFree)cruiser::real_dlsym(dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL), CUDA_SYMBOL_STRING(cuMemFree));
}
//fprintf(stderr, "end hooking cuMemFree.\n");
 cudaDeviceSynchronize();

#ifdef CUMALLOC
return cruiser::free_wrapper((void*)dptr);
#else
return cruiser::real_cuMemFree(dptr);
#endif
}
}

extern "C" {

CUresult CUDAAPI cuMemFreeHost ( void* dptr ){
//	fprintf(stderr, "begin hooking cuMemFreeHost.\n");
	

if(!cruiser::real_cuMemFreeHost){
fprintf(stderr, "begin hooking cuMemFreeHost.\n");
cruiser::real_cuMemFreeHost=(cruiser::fnMemFreeHost)cruiser::real_dlsym(dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL), CUDA_SYMBOL_STRING(cuMemFreeHost));
}

//fprintf(stderr, "end hooking cuMemFree.\n");
 cudaDeviceSynchronize();
 
return cruiser::free_wrapper((void*)dptr);
//return cruiser::real_cuMemFree(dptr);
}
}

extern "C" {
CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize){
//	fprintf(stderr, "============.\n");
if(!cruiser::real_cuMemAlloc){
fprintf(stderr, "begin hooking cuMemAlloc.\n");
cruiser::real_cuMemAlloc=(cruiser::fnMemAlloc)cruiser::real_dlsym(dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL), CUDA_SYMBOL_STRING(cuMemAlloc));
}
return cruiser::malloc_wrapper((void**)(dptr),bytesize);

}
}


extern "C" {
CUresult CUDAAPI cuMemHostAlloc ( void** dptr, size_t bytesize, unsigned int  Flags ){
//	fprintf(stderr, "begin hooking cuMemHostAlloc.\n");
if(!cruiser::real_cuMemHostAlloc){
fprintf(stderr, "begin hooking cuMemHostAlloc.\n");
cruiser::real_cuMemHostAlloc=(cruiser::fnMemHostAlloc)cruiser::real_dlsym(dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL), CUDA_SYMBOL_STRING(cuMemHostAlloc));
}
//	fprintf(stderr, "pointer=%p, flag=%u\n",(void*)dptr, Flags);

if(Flags!=0)
fprintf(stderr, "**************There is a implicit problem when Flag=!0***********.\n");

return cruiser::malloc_wrapper((void**)(dptr),bytesize);

//return cruiser::real_cuMemHostAlloc( dptr,  bytesize, Flags );//这里怎么free是个问题
}
}

// extern "C" {
// CUresult CUDAAPI cudaMallocHost ( void** ptr, size_t size ){
// 	fprintf(stderr, "cudaMallocHost============.\n");
// if(!cruiser::real_fncudaMallocHost){
// fprintf(stderr, "begin hooking cuMemAlloc.\n");
// cruiser::real_fncudaMallocHost=(cruiser::fncudaMallocHost)cruiser::real_dlsym(RTLD_NEXT, "cudaMallocHost");
// }
// return cruiser::malloc_wrapper((void**)(dptr),bytesize);

// }
// }

void* dlsym(void *handle, const char *symbol)
{
//		printf("symbol %s\n",symbol);
    if (strncmp(symbol, "cu", 2) != 0) {
        return (cruiser::real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
        return (void*)(&cuMemFree);
    }
	#ifdef CUMALLOC
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
		//printf("symbol %s\n",symbol);
        return (void*)(&cuMemAlloc);
    }
	#endif
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAllocPitch)) == 0) {
            return (void*)(&cuMemAllocPitch);
    }
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemHostAlloc)) == 0) {
	//	printf("===========symbol %s\n",symbol);
            return (void*)(&cuMemHostAlloc);
		 //  return NULL;
    }
	else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFreeHost)) == 0) {
	//			printf("=============symbol %s\n",symbol);

            return (void*)(&cuMemFreeHost);
    }
    return (cruiser::real_dlsym(handle, symbol));
}



