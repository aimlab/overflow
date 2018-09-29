
#ifndef THREAD_RECORD_H
#define THREAD_RECORD_H

#include "common.h"

namespace cruiser{

#define RING_SIZE 1024u
#define MAX_RING_SIZE 1u<<22

class Ring{
public:	
	
	char			cache_pad0[L1_CACHE_BYTES];
	CruiserNode		*array;
	unsigned	 	ringSize;
	Ring			*next;
	char			cache_pad1[L1_CACHE_BYTES - 3 * sizeof(int*)];
	unsigned 		volatile pi; // producer index
	unsigned		ci_snapshot;
	char			cache_pad2[L1_CACHE_BYTES -2 * sizeof(int)];
	unsigned 		volatile ci; //consumer index
	unsigned		pi_snapshot;
	char			cache_pad3[L1_CACHE_BYTES - 2 * sizeof(int)];

	unsigned		toIndex(unsigned i){return i % (ringSize - 1);}
	Ring(unsigned int size):ringSize(size), next(NULL), pi(0), ci_snapshot(0), 
							ci(0), pi_snapshot(0){
		array = new CruiserNode[size];
		assert(array);
	}
	
	~Ring(){delete [] array;}
	
	unsigned getSize(){return ringSize;}
	
	bool	produce(const CruiserNode & node){
#ifdef CRUISER_DEBUG
		fprintf(stderr, "produce: This thread id %lu, user addr %p, ring %p, \
				ringSize %u, ci %u, pi %u\n", (unsigned long)(pthread_self()), 
				node.userAddr, this, ringSize, ci, pi);
#endif
		ASSERT(node.userAddr);
		if((pi -ci_snapshot) >= ringSize){
			if((pi -ci) >= ringSize)
				return false;
			ci_snapshot = ci;
		}
		array[toIndex(pi)] = node; 


		pi++;
		return true;
	}
	
	bool	consume(CruiserNode & node){
		if(ci == pi_snapshot){
			if( ci == pi)
				return false;
			pi_snapshot = pi;
		}
			
		node = array[toIndex(ci)];
		ci++;
		return true;
	}	
};


}

#endif 
