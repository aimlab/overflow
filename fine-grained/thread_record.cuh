

#ifndef THREAD_RECORD_H
#define THREAD_RECORD_H

#include "common.cuh"
#include"addressCode.cuh"


namespace gmod{
class ThreadRecordList{
public:
	AddressCode * addressC;
	__device__ void ThreadRecordListinit(int8_t *p){
		addressC=(AddressCode*)malloc(sizeof(AddressCode));
		addressC->addressCodeinit(p);
	}
	__device__ void produce(unsigned long* addr){
		addressC->add_Address((long)addr);
	}
	__device__ unsigned long* consume(){
		return (unsigned long*)addressC->decode();
	}
};

__device__ static ThreadRecordList * g_threadrecordlist[GUARD_THREADSUM];


}

#endif
