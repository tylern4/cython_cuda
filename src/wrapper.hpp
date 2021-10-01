#include <iostream>
#include <vector>

#define REAL float
#define BLOCK_SIZE 16
// flexible between REAL and double
#define DEFAULT_DIMSIZE 1024

bool init_cuda();


bool print_cuda_properties(){
  try{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i+1);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << deviceProp.name << std::endl;
    std::cout << "Total Memory:\t\t" << deviceProp.totalGlobalMem/1E9 << " GB" << std::endl;
    std::cout << "Warp Size:\t\t" << deviceProp.warpSize << std::endl;
    std::cout << "Max Threads Per Block:\t" << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Clock Speed:\t\t" << deviceProp.clockRate/1E6 << " GHz"<< std::endl;
    std::cout << "Multi Processor Count:\t" << deviceProp.multiProcessorCount << std::endl;

  } catch(const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }

  return true;
}