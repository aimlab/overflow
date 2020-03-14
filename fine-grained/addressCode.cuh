#ifndef addressCode
#define addressCode
#include <iostream>
#include <stdint.h>
#include<stdio.h>
#include <math.h>

namespace gmod{
struct switchData{
	int8_t* data;
	int data_index;
};
__device__ int8_t* datanum;
__device__ bool reorganizeFlag=false;
__device__ int8_t* datanum2;
__device__ int8_t* data2[GUARD_THREADSUM];
__device__ switchData* dataline2[GUARD_THREADSUM];
__device__ switchData* old_dataline[GUARD_THREADSUM];
class AddressCode{
public:
	unsigned  long firstaddress;
	switchData* dataline;
	int decode_index;
	int previous_decode_index;
	__device__ void addressCodeinit(int8_t * p){
		firstaddress=0;
		decode_index=0;
		previous_decode_index=0;
		dataline= new switchData();
		dataline->data=p;
		dataline->data_index=0;
	}
	__device__ inline void resetDecodeIndex(){
		decode_index=0;
		previous_decode_index=0;
	}
	__device__ void add_Address(long address){
			int old_index;
			int8_t* old_data=0;
			int tmp_index;
			do{
					old_index=dataline->data_index;
					old_data=dataline->data;
					tmp_index=old_index+9;
					}while((old_index)!=atomicCAS(&dataline->data_index,old_index,tmp_index));
			int flag=1;
			memcpy(old_data+old_index+1,&address,8);
			memcpy(old_data+old_index,&flag,1);
			return;
		}
	__device__ void encode_signed_data(long deviation,int numbyte,int flag){
		int old_index;
		int8_t* old_data=0;
		int tmp_index;
		uint8_t code_id=0;
		do{
				old_index=dataline->data_index;
				old_data=dataline->data;
				tmp_index=old_index+1+numbyte;
				}while((old_index)!=atomicCAS(&dataline->data_index,old_index,tmp_index));
		code_id |= flag;
		code_id <<= 4;
		code_id |= numbyte;
		code_id <<= 3;
		code_id |= 7;
		 memcpy(old_data+old_index+1,&deviation,numbyte);
		memcpy(old_data+old_index,&code_id,1);
		return;
	}

	__device__ int getSize(long address,  long & deviation,int &flag){
		unsigned long old,tmp;
		do{
			if(firstaddress!=0)
				break;
			old=firstaddress;
					tmp=address;
					}while((unsigned long long int)old!=atomicCAS((unsigned long long int* )&firstaddress, (unsigned long long int)old,
							(unsigned long long int)tmp));

		deviation=address-firstaddress;

		if(deviation==0){
			return 1;
		}
		if(deviation<0){
		flag=1;
		deviation=-deviation;
		}
		int numbyte;
		numbyte=ceil((log2((double)(deviation+1)))/8);
#ifdef DEBUG
		if(numbyte>4)
			printf("error");
#endif
		return numbyte;
	}
	__device__ void signZero(){
		int8_t flag=0;
		memcpy(dataline->data+previous_decode_index,&flag,1);

	}
	__device__ void isreorganize(){
		if(dataline->data_index>=(reorgnize_num-3*reorgnize_num/10))
				reorganizeFlag=true;
	}
	__device__ void getLargeSize(){

	}
	__device__ void reorganize(){
	if(threadIdx.x==0){
		reorgnize_num=2*reorgnize_num;
			printf("gabage collecton %d\n", reorgnize_num);
			datanum2=new int8_t[GUARD_THREADSUM*reorgnize_num];
		}
	__syncthreads();
		data2[threadIdx.x]=reorgnize_num*threadIdx.x+datanum2;
		dataline2[threadIdx.x]=new switchData();
		dataline2[threadIdx.x]->data=data2[threadIdx.x];
		dataline2[threadIdx.x]->data_index=0;
		old_dataline[threadIdx.x]=dataline;
		dataline=dataline2[threadIdx.x];
		for(int i=0;i<old_dataline[threadIdx.x]->data_index;){
			int8_t flag=0;
			memcpy(&flag,old_dataline[threadIdx.x]->data+i,1);
			i+=9;
			if(flag!=0){
				int old_index;
				int tmp_index;
				do{
					old_index=dataline->data_index;
					tmp_index=old_index+9;
						}while((old_index)!=atomicCAS(&dataline->data_index,old_index,tmp_index));
						memcpy(dataline->data+old_index+1,old_dataline[threadIdx.x]->data+i-8,8);
						memcpy(dataline->data+old_index,old_dataline[threadIdx.x]->data+i-9,1);
			}
		}
		__syncthreads();
		if(threadIdx.x==0){
			free(old_dataline[0]->data);
			reorganizeFlag=false;
		}
	}
	__device__ void encode(long address){
		long deviation;
		int numbyte;
		int flag=0;
		numbyte=getSize(address,deviation,flag);
		encode_signed_data(deviation,numbyte,flag);
		return;
			}

	__device__ long decode(){

		long address=0;
		int8_t flag=0;
		while(decode_index<dataline->data_index){
			previous_decode_index=decode_index;
			memcpy(&flag,dataline->data+decode_index,1);

			decode_index+=9;
				if(flag==0){
				continue;
			}
			memcpy(&address,dataline->data+decode_index-8,8);
			return address;
						}
			return 0;
#ifdef INDEX
		printf("threadIdx.x=%d,%d\n",threadIdx.x,data_index);
#endif
	}
};
}

#endif
