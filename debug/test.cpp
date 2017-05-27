#include <stdio.h>
#include <stdlib.h> 


#ifdef __APPLE__ 

#include <OpenCL/cl.h> 

#elif defined(__linux__) 

#include <CL/cl.h> 
#include <CL/opencl.h> 

#endif 

int main()
{
    cl_int status=0;
    size_t deviceListSize;
    cl_uint numPlatforms;
    cl_platform_id platfomr=NULL;

    //status = clGetPlatformIDs(0,NULL,&numPlatforms);

    printf("status=%d\n",status);

    return 0;
}
//export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}