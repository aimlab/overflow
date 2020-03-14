
#include <math.h>
#include<fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include "monitor.cuh"

namespace gmod{
void getGPUmemoryInfo(){
	size_t free_byte ;
	size_t total_byte ;
	cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
	if ( cudaSuccess != cuda_status ){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
		exit(1);
   }
	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
			used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

__global__ void init(int8_t* dev_datanum){
#ifdef GUAEDTIME
execution[threadIdx.x]=0;
#endif
if(threadIdx.x==0){
	datanum=dev_datanum;
	for(int i=0;i<GUARD_THREADSUM/32;i++)
	g_exit_procedure[i]=RUNNING;
}
	g_threadrecordlist[threadIdx.x] =(ThreadRecordList*)malloc(sizeof(ThreadRecordList));
	g_threadrecordlist[threadIdx.x]->ThreadRecordListinit(DATANUM*threadIdx.x+dev_datanum);
}
__global__ void police(){
		transmitter();
}
__global__ void police_max_guard(long long int* duration){
	transmitter_max_guard(duration);
}


__global__ void beforeExit(void){
	for(int i=0;i<GUARD_THREADSUM/32;i++)
	g_exit_procedure[i] = EXIT_HOOKED;
#ifdef MEMORYUSAGE
	long sum=0;
	for(int i=0;i<GUARD_THREADSUM;i++){
		sum+=g_threadrecordlist[i]->addressC->dataline->data_index;
	}
	printf("sum: %ld\n",sum);
#endif
}

 __global__ void beforeExit_G(float* sumc){
         for(int i=0;i<GUARD_THREADSUM/32;i++)
         g_exit_procedure[i] = EXIT_HOOKED;
 #ifdef MEMORYUSAGE
         long sum=0;
         for(int i=0;i<GUARD_THREADSUM;i++){
                 sum+=g_threadrecordlist[i]->addressC->dataline->data_index;
         }
         printf("sum: %ld\n",sum);
 #endif
 
 #ifdef GUARDTIME
 *sumc=0;
 for(int i=0;i<GUARD_THREADSUM;i++){
                 *sumc+=execution[i];
         }
*sumc=(*sumc)/GUARD_THREADSUM;
#endif
  }

__device__ static unsigned long id = 0;
__device__ inline void afterMalloc(void* addr, size_t word_size){

	unsigned long *p = (unsigned long *)addr;
		p[1] = word_size;
		p[0]=(g_canary ^ word_size);
		p[2 + word_size] = p[0];
		g_threadrecordlist[(threadIdx.x+blockIdx.x*blockDim.x)%GUARD_THREADSUM]->produce(p+2);
		return;

}

__device__ inline static void beforeFree(void* addr){
	unsigned long *p = (unsigned long*)addr - 2;
	if(p[0] == (g_canary_free ^ p[1])){
		printf("Duplicate frees are detected\n");
	}
	p[0] ^= (g_canary ^ g_canary_free);

}
__device__ int i=0;
__device__ static void* malloc_wrapper(size_t size){
	size_t word_size = size / sizeof(long) + (size%sizeof(long)?1:0);
	void *addr = malloc((word_size + EXTRA_WORDS) * sizeof(long));
	if(!addr)
		return NULL;

	afterMalloc(addr, word_size);
	return (long*)addr + 2;

}

__device__ static void free_wrapper(void* addr){
	beforeFree(addr);
	return;
}


}


__device__ void* mallocN(size_t size){
	return gmod::malloc_wrapper(size);

}

__device__ void freeN(void* addr){
	gmod::free_wrapper(addr);
}

