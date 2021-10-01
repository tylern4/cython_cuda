#include <vector>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "wrapper.hpp"

__device__ float MP = 0.93827208816;
__device__ float ME = 0.00051099895;

__device__ float p_targ_px = 0.0;
__device__ float p_targ_py = 0.0;
__device__ float p_targ_pz = 0.0;
__device__ float p_targ_E = 0.93827208816;

__constant__ float _beam_E;


__device__ float norm(const float& x, const float& y, const float& z, const float& e){
    float norm2 = x*x + y*y + z*z - e*e;
    if(norm2 < 0) norm2 = -norm2;
    return sqrtf(norm2);
}
