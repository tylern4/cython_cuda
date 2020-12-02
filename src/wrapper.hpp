#include <iostream>
#include <vector>

#define REAL float
#define BLOCK_SIZE 16
// flexible between REAL and double
#define DEFAULT_DIMSIZE 1024
// std::vector<float> calc_W(std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi);

// float calc_q2(std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi);

bool init_cuda();
/*      subroutine initialize (n,m,alpha,dx,dy,u,f)
 ******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/
void initialize(long n, long m, REAL alpha, REAL *dx, REAL *dy, REAL *u_p, REAL *f_p);

void jacobi_seq(long n, long m, float dx, float dy, float alpha, float relax, float *u_p, float *f_p, float tol,
                int mits);
void jacobi_cuda(long n, long m, float dx, float dy, float alpha, float relax, float *u_p, float *f_p, float tol,
                 int mits);

bool print_cuda_properties();

std::vector<float> calc_W(float beam_E, std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi);
std::vector<float> calc_Q2(float beam_E, std::vector<float> e_p, std::vector<float> e_theta, std::vector<float> e_phi);