# distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sin, cos, sqrt



cdef float MP = 0.93827208816
cdef float E0 = 4.81726
cdef float ME = 0.00051099895

cdef float p_targ_px = 0.0
cdef float p_targ_py = 0.0
cdef float p_targ_pz = 0.0
cdef float p_targ_E = MP

cdef float e_beam_px = 0.0
cdef float e_beam_py = 0.0
cdef float e_beam_pz = sqrt(E0**2-ME**2)
cdef float e_beam_E = E0


cdef extern from "wrapper.hpp":
    void jacobi_cuda(long, long, float, float, float, float,
                 float*, float*, float, int)
    void initialize(long, long, float, float*, float*, float*, float*)
    bool print_cuda_properties()
    vector[float] calc_W(float beam_E, vector[float] e_p, vector[float] e_theta, vector[float] e_phi)
    vector[float] calc_Q2(float beam_E, vector[float] e_p, vector[float] e_theta, vector[float] e_phi)


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

def cuda_w(beam_E, e_p, e_theta, e_phi):
    x = calc_W(beam_E, e_p, e_theta, e_phi)
    return x

def cuda_q2(beam_E, e_p, e_theta, e_phi):
    x = calc_Q2(beam_E, e_p, e_theta, e_phi)
    return x

@np.vectorize
def cython_w(float e_p, float e_theta, float e_phi):
    cdef float e_prime_px = e_p*sin(e_theta)*cos(e_phi)
    cdef float e_prime_py = e_p*sin(e_theta)*sin(e_phi)
    cdef float e_prime_pz = e_p*cos(e_theta)
    cdef float e_prime_E = sqrt(e_prime_px**2 + e_prime_py**2 + e_prime_pz**2 - ME**2)
    
    cdef float temp_px = e_beam_px - e_prime_px + p_targ_px
    cdef float temp_py = e_beam_py - e_prime_py + p_targ_py
    cdef float temp_pz = e_beam_pz - e_prime_pz + p_targ_pz
    cdef float temp_E = e_beam_E - e_prime_E + p_targ_E
    
    
    cdef float temp2 = temp_px**2+temp_py**2+temp_pz**2-temp_E**2
    cdef float temp3 = sqrt(-temp2)
    
    return temp3


@np.vectorize
def cython_q2(float e_p, float e_theta, float e_phi):
    cdef float e_prime_px = e_p*sin(e_theta)*cos(e_phi)
    cdef float e_prime_py = e_p*sin(e_theta)*sin(e_phi)
    cdef float e_prime_pz = e_p*cos(e_theta)
    cdef float e_prime_E = sqrt(e_prime_px**2 + e_prime_py**2 + e_prime_pz**2 - ME**2)
    
    cdef float temp_px = e_beam_px - e_prime_px
    cdef float temp_py = e_beam_py - e_prime_py
    cdef float temp_pz = e_beam_pz - e_prime_pz
    cdef float temp_E = e_beam_E - e_prime_E

    cdef float temp2 = temp_px**2+temp_py**2+temp_pz**2-temp_E**2

    return temp2