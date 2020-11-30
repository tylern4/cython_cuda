#include <vector>
#include <iostream>
#include <stdlib.h>
#include <chrono>

__constant__ float MP = 0.93827208816;
__constant__ float E0 = 4.81726;
__constant__ float ME = 0.00051099895;

__constant__ float p_targ_px = 0.0;
__constant__ float p_targ_py = 0.0;
__constant__ float p_targ_pz = 0.0;
__constant__ float p_targ_E = 0.93827208816;

__constant__ float e_beam_px = 0.0;
__constant__ float e_beam_py = 0.0;
__constant__ float e_beam_pz = 4.81726;
__constant__ float e_beam_E = 4.81726;

__global__
void W_kernel(float* e_p, float* e_theta, float* e_phi, float* out) {
    int tid = blockIdx.x;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = sqrt(e_prime_px*e_prime_px + e_prime_py*e_prime_py + e_prime_pz*e_prime_pz - ME*ME);
    
    float temp_px = e_beam_px - e_prime_px + p_targ_px;
    float temp_py = e_beam_py - e_prime_py + p_targ_py;
    float temp_pz = e_beam_pz - e_prime_pz + p_targ_pz;
    float temp_E = e_beam_E - e_prime_E + p_targ_E;

    float temp2 = temp_px*temp_px+temp_py*temp_py+temp_pz*temp_pz-temp_E*temp_E;

    out[tid] = sqrt(-temp2);
}

std::vector<float> calc_W(std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi) {
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
    cudaMemcpy(_e_p, &e_p[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(_e_theta, &e_theta[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(_e_phi, &e_phi[0], size, cudaMemcpyHostToDevice);

    // Make host output
    float *out = (float *)malloc(size);

    // Call kernel for each in N do 1
    W_kernel <<<N, 1>>> (_e_p, _e_theta, _e_phi, _out);
    // Copy output from device
    cudaMemcpy(out, _out, size, cudaMemcpyDeviceToHost);

    // Make copy of array as vector
    std::vector<float> vec {out, out + N};

#ifdef TIME_FUNC
    std::chrono::duration<double> elapsed_full = (std::chrono::high_resolution_clock::now() - start);
    std::cout << N / elapsed_full.count() << " Hz" << std::endl;
#endif

    return vec;
}


__global__ 
void q2_kernel(float* e_p, float* e_theta, float* e_phi, float* out){
    int tid = blockIdx.x;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = sqrt(e_prime_px*e_prime_px + e_prime_py*e_prime_py + e_prime_pz*e_prime_pz - ME*ME);
    
    float temp_px = e_beam_px - e_prime_px;
    float temp_py = e_beam_py - e_prime_py;
    float temp_pz = e_beam_pz - e_prime_pz;
    float temp_E = e_beam_E - e_prime_E;

    float temp2 = temp_px*temp_px+temp_py*temp_py+temp_pz*temp_pz-temp_E*temp_E;

    out[tid] = temp2;
}

std::vector<float> calc_Q2(std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi) {
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
    cudaMemcpy(_e_p, &e_p[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(_e_theta, &e_theta[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(_e_phi, &e_phi[0], size, cudaMemcpyHostToDevice);

    // Make host output
    float *out = (float *)malloc(size);

    // Call kernel for each in N do 1
    q2_kernel <<<N, 1>>> (_e_p, _e_theta, _e_phi, _out);
    // Copy output from device
    cudaMemcpy(out, _out, size, cudaMemcpyDeviceToHost);

    // Make copy of array as vector
    std::vector<float> vec {out, out + N};

#ifdef TIME_FUNC
    std::chrono::duration<double> elapsed_full = (std::chrono::high_resolution_clock::now() - start);
    std::cout << N / elapsed_full.count() << " Hz" << std::endl;
#endif

    return vec;
}