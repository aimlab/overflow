
#ifndef UTILITY_H
#define UTILITY_H

#include <execinfo.h> // backtrace
#include <stdio.h>
#include <stdlib.h> // rand, exit, malloc, free
#include <sys/time.h> // gettimeofday
#include <assert.h>

namespace cruiser{
#if defined( NPROTECT ) 
static __thread int		t_protect 		= 0;
#else
static __thread int		t_protect 		= 1;
#endif
	
#ifdef PROTECT_TIME
double GetMSTime(void)
{
  struct timeval stNowTime;

  gettimeofday(&stNowTime,NULL);

  return (1.0*(stNowTime.tv_sec)*1000+1.0*(stNowTime.tv_usec)/1000);
}
#endif

#ifdef CRUISER_DEBUG
#define ASSERT(x)  do { \
	int _old_protect = t_protect; \
	t_protect = 0; \
	assert(x); \
	t_protect = _old_protect; \
	} while(0) //this is used to "swallow" the semicolon.
#else
#define ASSERT(x) //assert(x)
#endif

inline static unsigned int getUsTime(void){
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void msSleep(int msTime){
	if(msTime == -1)
		return;
	if(msTime > 999)
		msTime = 999;
	struct timespec	sleepTime;
	sleepTime.tv_sec= 0;
	sleepTime.tv_nsec= msTime*1000000;
	nanosleep(&sleepTime, NULL);
}

}

#endif 
