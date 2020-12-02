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

__device__ float norm2(const float& x, const float& y, const float& z, const float& e){
    return x*x + y*y + z*z - e*e;
}

__device__ float norm(const float& x, const float& y, const float& z, const float& e){
    float norm2 = x*x + y*y + z*z - e*e;
    if(norm2 < 0) norm2 = -norm2;
    return sqrtf(norm2);
}

__global__ void W_kernel(float* e_p, float* e_theta, float* e_phi, float* out) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float e_beam_px = 0;
    float e_beam_py = 0;
    float e_beam_pz = sqrt(_beam_E*_beam_E-ME*ME);
    float e_beam_E = _beam_E;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = norm(e_prime_px, e_prime_py, e_prime_pz, ME);
    
    float temp_px = e_beam_px - e_prime_px + p_targ_px;
    float temp_py = e_beam_py - e_prime_py + p_targ_py;
    float temp_pz = e_beam_pz - e_prime_pz + p_targ_pz;
    float temp_E = e_beam_E - e_prime_E + p_targ_E;

    out[tid] = norm(temp_px,temp_py,temp_pz,temp_E);
}

__global__ void q2_kernel(float* e_p, float* e_theta, float* e_phi, float* out){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float e_beam_px = 0;
    float e_beam_py = 0;
    float e_beam_pz = sqrt(_beam_E*_beam_E-ME*ME);
    float e_beam_E = _beam_E;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = norm(e_prime_px, e_prime_py, e_prime_pz, ME);
    
    float temp_px = e_beam_px - e_prime_px;
    float temp_py = e_beam_py - e_prime_py;
    float temp_pz = e_beam_pz - e_prime_pz;
    float temp_E = e_beam_E - e_prime_E;

    out[tid] = norm2(temp_px,temp_py,temp_pz,temp_E);
}

std::vector<float> calc_W(float beam_E, std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi) {
#ifdef TIME_FUNC
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Get size of vectors and N for use later
    size_t size = sizeof(float) * e_p.size();
    size_t N = e_p.size();

    // Make device side vectors and malloc on device
    float *_e_p;
    float *_e_theta;
    float *_e_phi;
    cudaMalloc((void **)&_e_p, size);
    cudaMalloc((void **)&_e_theta, size);
    cudaMalloc((void **)&_e_phi, size);

    // Make and malloc output vector
    float *_out;
    cudaMalloc((void **)&_out, size);

    // Copy in vectors to device arrays
    cudaMemcpyToSymbol(_beam_E, &beam_E, sizeof(REAL), 0, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_e_p, &e_p[0], size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(_e_theta, &e_theta[0], size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(_e_phi, &e_phi[0], size, cudaMemcpyHostToDevice, 0);

    // Make host output
    float *out = (float *)malloc(size);

    // Call kernel for each in N do 1
    W_kernel <<<N, 1>>>(_e_p, _e_theta, _e_phi, _out);

    // Copy output from device
    cudaMemcpyAsync(out, _out, size, cudaMemcpyDeviceToHost, 0);

    // Make copy of array as vector
    std::vector<float> vec {out, out + N};

#ifdef TIME_FUNC
    std::chrono::duration<double> elapsed_full = (std::chrono::high_resolution_clock::now() - start);
    std::cout << N / elapsed_full.count() << " Hz" << std::endl;
#endif

    // Free memory at the end
    cudaFree(_e_p);
    cudaFree(_e_theta);
    cudaFree(_e_phi);
    cudaFree(_out);

    return vec;
}

std::vector<float> calc_Q2(float beam_E, std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi) {
#ifdef TIME_FUNC
    auto start = std::chrono::high_resolution_clock::now();
#endif
    // Get size of vectors and N for use later
    size_t size = sizeof(float) * e_p.size();
    size_t N = e_p.size();

    // Make device side vectors and malloc on device
    float *_e_p;
    float *_e_theta;
    float *_e_phi;
    cudaMalloc((void **)&_e_p, size);
    cudaMalloc((void **)&_e_theta, size);
    cudaMalloc((void **)&_e_phi, size);

    // Make and malloc output vector
    float *_out;
    cudaMalloc((void **)&_out, size);

    // Copy in vectors to device arrays
    cudaMemcpyToSymbol(_beam_E, &beam_E, sizeof(REAL), 0, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_e_p, &e_p[0], size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(_e_theta, &e_theta[0], size, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(_e_phi, &e_phi[0], size, cudaMemcpyHostToDevice, 0);

    // Make host output
    float *out = (float *)malloc(size);

    // Call kernel for each in N do 1
    q2_kernel <<<N, 1>>> (_e_p, _e_theta, _e_phi, _out);

    // Copy output from device
    cudaMemcpyAsync(out, _out, size, cudaMemcpyDeviceToHost, 0);

    // Make copy of array as vector
    std::vector<float> vec {out, out + N};

#ifdef TIME_FUNC
    std::chrono::duration<double> elapsed_full = (std::chrono::high_resolution_clock::now() - start);
    std::cout << N / elapsed_full.count() << " Hz" << std::endl;
#endif

    // Free memory at the end
    cudaFree(_e_p);
    cudaFree(_e_theta);
    cudaFree(_e_phi);
    cudaFree(_out);

    return vec;
}