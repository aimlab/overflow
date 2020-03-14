
#ifndef LIST_H
#define LIST_H

#include "common.h"

namespace cruiser{
#define 		LIST_RING_SIZE 0x400000U 
#define			BATCH_SIZE	(4 * L1_CACHE_BYTES / sizeof(int*)) 
template<typename T, unsigned int ringSize> 
class RingT{
private:
	char				cache_pad0[L1_CACHE_BYTES];
	T* 				array[ringSize];
	unsigned int	volatile 	pi; // The producer index; read by the consumer
	char 				cache_pad1[L1_CACHE_BYTES-sizeof(int)];
	unsigned int	volatile 	ci; // The consumer index; read by the producer
	char 				cache_pad2[L1_CACHE_BYTES-sizeof(int)];
	// The consumer's local variables
	unsigned int 			pi_snapshot;
	unsigned int 			ci_current; // ci = ci_current, per batch
	unsigned int 			ci_batch;
	char				cache_pad3[L1_CACHE_BYTES-3*sizeof(int)];
	// The producer's local variables
	unsigned int 			ci_snapshot;
	unsigned int 			pi_current; // pi = pi_current, per batch
	unsigned int 			pi_batch;
	char				cache_pad4[L1_CACHE_BYTES-3*sizeof(int)];

	//bool			isFull(){return (pi -ci >= ringSize);}
	//bool			isEmpty(){return ci == pi;}
	unsigned		toIndex(unsigned i){return i % (ringSize - 1);}
	
public:
	RingT(unsigned preFilled){
		assert(preFilled < ringSize);
		for(unsigned int i = 0; i < preFilled; i++){
			array[i] = (T*) malloc(sizeof(T));
		}
		pi = pi_current = pi_snapshot = preFilled;
		ci = ci_current = ci_snapshot = ci_batch = pi_batch = 0;
	}
	bool produce(T *node){
		if(pi_current - ci_snapshot >= ringSize){
			if(pi_current - ci >= ringSize)
				return false;
			else
				ci_snapshot = ci;
		}
		array[toIndex(pi_current)] = node; // Entry value assignment
		pi_current++;
		pi_batch++;
		if(pi_batch >= BATCH_SIZE){
			pi_batch = 0;
			pi = pi_current;
		}
		return true;
	}
	
	bool consume(T * &node){
		if(ci_current == pi_snapshot){
			if(ci_current == pi)
				return false;
			else
				pi_snapshot = pi;
		}
		node = array[toIndex(ci_current)];
		ci_current++;
		ci_batch++;
		if(ci_batch >= BATCH_SIZE){
			ci_batch = 0;
			ci = ci_current;
		}
		return true;
	}
};


class List:public NodeContainer{
private:
	class ListNode{
	public:
		CruiserNode	cn;
		ListNode	*next;
		void markDelete(){cn.userAddr = (void*)-1L;}
		bool isMarkedDelete(){return cn.userAddr == (void*)-1L;}
	};
	
	RingT<ListNode, LIST_RING_SIZE>	ring;
	
	ListNode 		dummy; 
	
public:	
	#define PRE_ALLOCATED_FACTION 0
	
	List():ring(PRE_ALLOCATED_FACTION * LIST_RING_SIZE){
		dummy.next = NULL;
		dummy.cn.userAddr = NULL;
	}

	bool insert(const CruiserNode & node){
		ListNode* pn;
		if(!ring.consume(pn))
			pn = (ListNode*)malloc( sizeof(ListNode) );
		assert(pn);
		pn->cn = node;
		pn->next = dummy.next;
		dummy.next = pn;
		return true;
	}
	bool multi_thread_insert(const CruiserNode & node){
		ListNode* pn;
		ListNode* old_dummy_next;
		if(!ring.consume(pn))
			pn = (ListNode*)malloc( sizeof(ListNode) );
		assert(pn);
		pn->cn = node;
		do{
				old_dummy_next = dummy.next;
				pn->next = old_dummy_next;
			}while(!__sync_bool_compare_and_swap(&dummy.next, old_dummy_next, pn));

		return true;
	}
	
	int traverse( int (*pfn)(const CruiserNode &) );
};

int List::traverse( int (*pfn)(const CruiserNode &) ){
	ListNode *prev, *cur, *next;
	cur = dummy.next;
	if(!cur)
		return 1;
	if(!cur->isMarkedDelete()){
		// pfn Return values:
		// 	0: to stop monitoring (obsolete);
		//	1: have checked one node
		//	2: have encountered a dummy node (should never happen)
		//	3: a node is to be removed
		if(pfn(cur->cn) == 3)
			cur->markDelete();
	}
	
	prev = cur;
	cur = cur->next;
	while(NULL != cur){
		next = cur->next;
		if(cur->isMarkedDelete()){
			prev->next = next;
			if(!ring.produce(cur))
				free(cur);//delete cur;
		}else{
			switch(pfn(cur->cn)){
				// As the "stop monitoring" feature may be exploited,
				// we disallow it in this implementation.
				//case 0: // Stop monitoring
				//	return 0;
				case 1:
					prev = cur;
					break;
				// Should never happen because for this implementation
				// there is only one dummy node.
				//case 2:
				//	return 2;
				case 3:
					prev->next = next;
					if(!ring.produce(cur))
						free(cur);
					break;
				default:
					break;
			}
		}
		cur = next;
	}
	return 1;
}

}
#endif
