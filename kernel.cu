#ifndef KERNEL_CU
#define KERNEL_CU

__global__ void kernel(float * g_idata, float * g_odata)
{
  unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
  g_odata[id] = 2*g_idata[id];  
}
#endif
