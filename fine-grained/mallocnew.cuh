#include "memory.cuh"

namespace gmod{
#ifdef GUARDTIME
double t3,t4;
float* d_sumc;
#endif
void cudaMallocN(void **p, size_t s){
	size_t word_size = s / sizeof(long) + (s%sizeof(long)?1:0);
	cudaMalloc(p,(word_size + EXTRA_WORDS) * sizeof(long));
	*((long**)p)=*((long**)p)+2;
		if(!*p)
		printf("invaild pointer\n");
}

void cudaFreeN(void* addr){
	addr=(long*)addr-2;
	cudaFree(addr);
}

__device__ void cudaMallocN_aftermalloc(void* addr, size_t byte_num){
	size_t word_size = byte_num / sizeof(long) + (byte_num%sizeof(long)?1:0);
	afterMalloc((void*)((long*)addr-2),word_size);
}

int8_t* dev_datanum;
void my_ctor(void){
	cudaStream_t stream1;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 200 * 1024 * 1024);
	cudaStreamCreate(&stream1);
	cudaMalloc((void**)&dev_datanum,sizeof(int8_t)*GUARD_THREADSUM*DATANUM);
	cudaMemset(dev_datanum, 0, GUARD_THREADSUM*DATANUM);
	gmod::init<<<1,GUARD_THREADSUM,0,stream1>>>(dev_datanum);
	cudaDeviceSynchronize();
#ifdef GUARDTIME
	cudaMalloc( &d_sumc, sizeof(float));
	
	t3 = omp_get_wtime();
#endif
	printf("GUARD_THREAD=%d\n",GUARD_THREADSUM);

	gmod::police<<<1,GUARD_THREADSUM,0,stream1>>>();
	}


void my_dtor(void) {
#ifdef GUARDTIME
	gmod::beforeExit_G<<<1,1>>>(d_sumc);
#else
	gmod::beforeExit<<<1,1>>>();
#endif
	cudaDeviceSynchronize();
#ifdef GUARDTIME
	t4 = omp_get_wtime();
#endif
	cudaFree(dev_datanum);

#ifdef GUARDTIME
	float sumc=0;
	cudaMemcpy( &sumc, d_sumc, sizeof(float), cudaMemcpyDeviceToHost);
	printf("sumc = %f\n",sumc);
	printf("guard kernel time: %4.2lf ms\n", (t4-t3) * 1e3);
	printf("guard-kernel-time: %4.2lf ms\n", ((t4-t3)/sumc) * 1e3);

#endif
}

}
