#include <iostream>
#include "wrapper.hpp"

void spmv(float *a, float *v, float *x, int nrows, int* rowndx, int* colndx) {

    // int num_gpus = acc_get_num_devices(acc_device_nvidia);

    // #pragma omp parallel num_threads(num_gpus)
    #pragma acc parallel loop vector_length(128)
    for(int i=0; i<nrows; ++i ){
        float val=0.0f;
        #pragma acc loop vector reduction(+:val)
        for(int n=0; n<nrows; ++n)
            val += a[n*nrows+i]*v[i*nrows+n] + x[i*nrows+n];

        x[i] = val;
    }
} 

int main(int argc, char const *argv[]) {
    print_cuda_properties();
    const int nrows = 1 << 20;

    float* a[nrows*nrows], v[nrows*nrows], x[nrows*nrows];

    #pragma acc parallel loop vector_length(128)
    for(int i=0; i<nrows; ++i ){
        for(int j=0; j<nrows; ++j ){
            a[i*nrows+j] = random();
            v[i*nrows+j] = random();
            x[i*nrows+j] = random();
        }
    }


    spmv(a, v, x, nrows);

    return 0;
}
