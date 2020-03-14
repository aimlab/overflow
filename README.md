# GMODx

## The fine-grained memory management

### Usage

- Including mallocNew.cuh file into application.
- Before launching user kernels,  adding my_ctor() function to initialize GMOD and launch guard kernel.
- After completing user kernels, adding my_dtor() function to stop guard kernel.

- Replacing malloc and free function with mallocN and freeN function.

### Compiling

add **-default-stream per-thread** flag to make guard kernel and user kernel  concurrently run.

## The coarse-grained memory management

### Usage

- Executing ./getDynLib.sh to get dynamic shared library.
- Using LD_PRELOAD to load this dynamic shared library.



