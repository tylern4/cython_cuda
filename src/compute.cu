#include <vector>
#include <iostream>

__global__
void __calc_W(float* e_p, float* e_theta, float* e_phi, float* out){
    int tid = blockIdx.x;
    printf("Here %d",tid);
    float MP = 0.93827208816;
    float E0 = 4.81726;
    float ME = 0.00051099895;

    float p_targ_px = 0.0;
    float p_targ_py = 0.0;
    float p_targ_pz = 0.0;
    float p_targ_E = MP;

    float e_beam_px = 0.0;
    float e_beam_py = 0.0;
    float e_beam_pz = sqrtf(E0*E0-ME*ME);
    float e_beam_E = E0;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = sqrtf(e_prime_px*e_prime_px + e_prime_py*e_prime_py + e_prime_pz*e_prime_pz - ME*ME);
    
    float temp_px = e_beam_px - e_prime_px + p_targ_px;
    float temp_py = e_beam_py - e_prime_py + p_targ_py;
    float temp_pz = e_beam_pz - e_prime_pz + p_targ_pz;
    float temp_E = e_beam_E - e_prime_E + p_targ_E;
    
    
    float temp2 = temp_px*temp_px+temp_py*temp_py+temp_pz*temp_pz-temp_E*temp_E;
    float temp3 = sqrtf(-temp2);
    out[tid] = temp3;
}

std::vector<float> calc_W(std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi)
{
    float *_e_p;
    float *_e_theta;
    float *_e_phi;
    float *_out;

    cudaMalloc(&_e_p, sizeof(float) * e_p.size());
    cudaMalloc(&_e_theta, sizeof(float) * e_theta.size());
    cudaMalloc(&_e_phi, sizeof(float) * e_phi.size());
    cudaMalloc(&_out, sizeof(float) * e_p.size());

    cudaMemcpy(_e_p,&e_p[0],sizeof(float)*e_p.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(_e_theta,&e_theta[0],sizeof(float)*e_theta.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(_e_phi,&e_phi[0],sizeof(float)*e_phi.size(),cudaMemcpyHostToDevice);
    
    dim3 dimBlock(512, 512);
    dim3 dimGrid(1024 / dimBlock.x, 1024 / dimBlock.y);
    __calc_W<<<dimGrid, dimBlock>>>(_e_p, _e_theta, _e_phi, _out);
    cudaDeviceSynchronize();

    std::vector<float> x;
    cudaMemcpy(&x[0],_out,sizeof(float)*e_phi.size(),cudaMemcpyDeviceToHost);

    return x;
}


__device__ 
void __calc_q2(float* e_p, float* e_theta, float* e_phi, float* out){
    int tid = blockIdx.x;
    float E0 = 4.81726;
    float ME = 0.00051099895;

    float e_beam_px = 0.0;
    float e_beam_py = 0.0;
    float e_beam_pz = sqrtf(E0*E0-ME*ME);
    float e_beam_E = E0;

    float e_prime_px = e_p[tid]*sinf(e_theta[tid])*cosf(e_phi[tid]);
    float e_prime_py = e_p[tid]*sinf(e_theta[tid])*sinf(e_phi[tid]);
    float e_prime_pz = e_p[tid]*cosf(e_theta[tid]);
    float e_prime_E = sqrtf(e_prime_px*e_prime_px + e_prime_py*e_prime_py + e_prime_pz*e_prime_pz - ME*ME);
    
    float temp_px = e_beam_px - e_prime_px;
    float temp_py = e_beam_py - e_prime_py;
    float temp_pz = e_beam_pz - e_prime_pz;
    float temp_E = e_beam_E - e_prime_E;

    float temp2 = temp_px*temp_px+temp_py*temp_py+temp_pz*temp_pz-temp_E*temp_E;

    out[tid] = temp2;
}