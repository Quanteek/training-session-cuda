#include <fstream>
#include <iostream>
#include <stdexcept>
#include <ctime>
#include <cuda.h>
#include "kernel.cu"

int main(int argc, char ** argv)
{
  cudaSetDevice(0);
  //Mise en place des donnees sur le host
  unsigned int nbr_data = 1024;
  unsigned int mem_size = sizeof(float)*nbr_data;
  float * h_idata = (float*)malloc(mem_size);
  for(int i = 0 ; i < nbr_data ; ++i)
    {
      h_idata[i] = 1;
    }

  //Allocation des espaces sur le device et copie du host sur le device
  float * d_idata;
  cudaMalloc((void**)&d_idata, mem_size);
  cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

  float * d_odata;
  cudaMalloc((void**)&d_odata, mem_size);

  //Mise en place du decoupage en fixant le nombre de threads
  unsigned int nbr_threads = 32;

  //unsigned int shared_mem_size = sizeof(float)*nbr_threads;

  //Calcul du nombre de blocs necessaires en fonction du nombre de threads et du nombre de donnees
  unsigned int nbr_blocks = (nbr_data+nbr_threads-1)/nbr_threads;

  //Allocation des grilles et threads
  dim3 grid(nbr_blocks, 1, 1);
  dim3 threads(nbr_threads, 1, 1);

  kernel<<<grid, threads>>>(d_idata, d_odata);

  cudaMemcpy(h_idata, d_odata, mem_size, cudaMemcpyDeviceToHost);
  
  std::ofstream file("./output.txt");
  if(!file.is_open())
    throw std::runtime_error("file error !"); 
  for(int i = 0 ; i < nbr_data ; ++i)
    {
      file << h_idata[i] << "\n";
    }
  file.close();
  
  cudaFree(d_odata);
  cudaFree(d_idata);
  free(h_idata);
  
  cudaThreadExit();
  return 0;
}
