# distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython


cdef extern from "wrapper.hpp":
    void jacobi_cuda(long, long, float, float, float, float,
                 float*, float*, float, int)
    void initialize(long, long, float, float*, float*, float*, float*)
    bool print_cuda_properties()


def jacobi(int n, int m, float alpha, float relax, float tol, int mits):
    cdef float *u = <float *>malloc(sizeof(float) * n * m)
    cdef float *f = <float *>malloc(sizeof(float) * n * m)

    cdef float dx; 
    cdef float dy; 
    initialize(n, m, alpha, &dx, &dy, u, f)
    jacobi_cuda(n, m, dx, dy, alpha, relax, u, f, tol, mits)

    output = np.zeros([n,m])
    for _n in range(n):
        for _m in range(m):
            output[_n][_m] = u[_n * n + _m]

    return output

def cuda_properties():
    return print_cuda_properties()