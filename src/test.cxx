#include <iostream>
#include <vector>
#include <omp.h>

#ifdef TARGET_GPU
#include "wrapper.hpp"
#endif

std::vector<float> spmv(std::vector<float> a, std::vector<float> v, std::vector<float> x, int nrows) {

    int num_threads = omp_get_num_procs();

    #pragma omp parallel for num_threads(num_threads)
    for(int z=0; z<nrows*nrows; ++z ){
        float val=0.0f;
        #pragma acc kernel
        {
        #pragma acc parallel loop
        for(int i=0; i<nrows; ++i ){
            for(int n=0; n<nrows; ++n){
                val += a[n*nrows+i]*v[i*nrows+n] + x[i*nrows+n];
            }
        }
        // #pragma acc wait
        x[z%nrows] = val;
        }
    }
    return x;
} 

int main(int argc, char const *argv[]) {

    #ifdef TARGET_GPU
    print_cuda_properties();
    #endif

    const int nrows = 1000;

    std::vector<float> a(nrows*nrows), v(nrows*nrows), x(nrows*nrows);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<nrows; ++i ){
        for(int j=0; j<nrows; ++j ){
            a[i*nrows+j] = static_cast<float>(random());
            v[i*nrows+j] = static_cast<float>(random());
            x[i*nrows+j] = static_cast<float>(random());
        }
    }


    spmv(a, v, x, nrows);

    return 0;
}
