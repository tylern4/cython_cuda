#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include "wrapper.hpp"

/************************************************************
 * program to solve a finite difference
 * discretization of Helmholtz equation :
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * using Jacobi iterative method.
 *
 * Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
 *
 * This c version program is translated by
 * Chunhua Liao, University of Houston, Jan, 2005
 *
 * Directives are used in this code to achieve parallelism.
 * All do loops are parallelized with default 'static' scheduling.
 *
 * Input :  n - grid dimension in x direction
 *          m - grid dimension in y direction
 *          alpha - Helmholtz constant (always greater than 0.0)
 *          tol   - error tolerance for iterative solver
 *          relax - Successice over relaxation parameter
 *          mits  - Maximum iterations for iterative solver
 *
 * On output
 *       : u(n,m) - Dependent variable (solutions)
 *       : f(n,m) - Right hand side function
 *************************************************************/

 void initialize(long n, long m, REAL alpha, REAL *dx, REAL *dy, REAL *u_p,
                REAL *f_p)
{
    long i;
    long j;
    long xx;
    long yy;
    REAL(*u)[m] = (REAL(*)[m])u_p;
    REAL(*f)[m] = (REAL(*)[m])f_p;

    //double PI=3.1415926;
    *dx = (2.0 / (n - 1));
    *dy = (2.0 / (m - 1));
    /* Initialize initial condition and RHS */
    #pragma omp parallel for private(xx,yy,j,i)
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
        {
            xx = ((int)(-1.0 + (*dx * (i - 1))));
            yy = ((int)(-1.0 + (*dy * (j - 1))));

            if (j == 0)
              u[i][j] = 1.0;
            else
              u[i][j] = 0.0;
            
              f[i][j] = (((((-1.0 * alpha) * (1.0 - (xx * xx))) * (1.0 - (yy * yy))) -
                        (2.0 * (1.0 - (xx * xx)))) -
                       (2.0 * (1.0 - (yy * yy))));
        }
}


/*  subroutine error_check (n,m,alpha,dx,dy,u,f)
 implicit none
 ************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/
double error_check(long n, long m, REAL alpha, REAL dx, REAL dy, REAL *u_p,
                   REAL *f_p) {
  int i;
  int j;
  REAL xx;
  REAL yy;
  REAL temp;
  double error;
  error = 0.0;
  REAL(*u)[m] = (REAL(*)[m])u_p;
// REAL(*f)[m] = (REAL(*)[m])f_p;
#pragma omp parallel for private(xx, yy, temp, j, i) reduction(+ : error)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      xx = (-1.0 + (dx * (i - 1)));
      yy = (-1.0 + (dy * (j - 1)));
      temp = (u[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy))));
      error = (error + (temp * temp));
    }
  error = (sqrt(error) / (n * m));
  return error;
}
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax,
                REAL *u_p, REAL *f_p, REAL tol, int mits);
void jacobi_cuda(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax,
                 REAL *u_p, REAL *f_p, REAL tol, int mits);


bool print_cuda_properties(){
  try{
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

/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,mits)
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tol    Tolerance for iterative solver
 *         mits  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************/

__constant__ float c_ax;
__constant__ float c_ay;
__constant__ float c_b;
__constant__ float c_omega;
__constant__ long c_n;
__constant__ long c_m;

__global__ void jacobi_kernel(REAL *u, REAL *uold, REAL *resid, REAL *cuda_f) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row == 0 || col == 0)
    return;
  if (row >= (c_n - 1) || col >= (c_m - 1))
    return;

  resid[row * c_n + col] =
      (c_ax * (uold[(row - 1) * c_n + col] + uold[(row + 1) * c_n + col]) +
       c_ay * (uold[row * c_n + (col - 1)] + uold[row * c_n + (col + 1)]) +
       c_b * uold[row * c_n + col] - cuda_f[row * c_n + col]) /
      c_b;
  u[row * c_n + col] = uold[row * c_n + col] - c_omega * resid[row * c_n + col];
  // resid[row * c_n + col] = resid[row * c_n + col] * resid[row * c_n + col];
}

void jacobi_cuda(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega,
                 REAL *u_p, REAL *f_p, REAL tol, int mits) {
  long i, j, k;
  REAL error;
  REAL *temp;
  REAL *resid = (REAL *)malloc((sizeof(REAL) * n * m));
  REAL *uold = (REAL *)malloc((sizeof(REAL) * n * m));
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;

  /* Initialize coefficients */
  /* X-direction coef */
  REAL ax = (1.0 / (dx * dx));
  /* Y-direction coef */
  REAL ay = (1.0 / (dy * dy));
  /* Central coeff */
  REAL b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
  error = (10.0 * tol);
  k = 1;
  /* TODO #2: CUDA memory allocation for u, f and uold and copy data for u and f
   * from host memory to GPU memory, depending on how error
   * will be calculated (see below), a [n][m] array or a one-element array need
   * to be allocated as well. */
  int size = (sizeof(REAL) * n * m);
  REAL *cuda_u;
  REAL *cuda_f;
  REAL *cuda_uold;
  REAL *cuda_resid;

  // Copy u to cuda memory
  cudaMalloc((void **)&cuda_u, size);
  cudaMemcpy(cuda_u, u, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&cuda_f, size);
  cudaMemcpy(cuda_f, f, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&cuda_uold, size);
  cudaMalloc((void **)&cuda_resid, size);

  cudaMemcpyToSymbol(c_ax, &ax, sizeof(REAL), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_ay, &ay, sizeof(REAL), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_b, &b, sizeof(REAL), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_omega, &omega, sizeof(REAL), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_n, &n, sizeof(long), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_m, &m, sizeof(long), 0, cudaMemcpyHostToDevice);

  /* TODO #4: set 16x16 threads/block and n/16 x m/16 blocks/grid for GPU
   * computation (assuming n and m are dividable by 16 */
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);

  while ((k <= mits) && (error > tol)) {
    error = 0.0;

    /* TODO #3: swap u and uold */
    temp = cuda_u;
    cuda_u = cuda_uold;
    cuda_uold = temp;
    /* TODO #5: launch jacobi_kernel */
    jacobi_kernel <<<dimGrid, dimBlock>>>
        (cuda_u, cuda_uold, cuda_resid, cuda_f);
    /* TODO #6: compute error on CPU or GPU. error is calculated by accumulating
    *          resid*resid computed by each thread. There are multiple
    * approaches to compute the error. E.g. 1). A array of resid[n][m]
    *          could be allocated and store the resid computed by each thread.
    * After the computation, all the resids in the array are
    *          accumulated on either CPU or GPU. 2). A simpler implementation
    * could be just using CUDA atomicAdd, check.
    *
 (http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    */
    /* Error check */
    /* TODO #7: copy the error from GPU to CPU */
    cudaMemcpy(resid, cuda_resid, size, cudaMemcpyDeviceToHost);
    for (i = 1; i < (n - 1); i++)
      for (j = 1; j < (m - 1); j++) {
        error += resid[i * n + j] * resid[i * n + j];
      }

    if (k % 500 == 0)
      printf("Finished %ld iteration with error: %g\n", k, error);

    error = sqrt(error) / (n * m);
    k = k + 1;
  } /*  End iteration loop */
    /* TODO #8: GPU memory deallocation */
  
  cudaMemcpy(u_p, cuda_u, size, cudaMemcpyDeviceToHost);


  cudaFree(cuda_u);
  cudaFree(cuda_f);
  cudaFree(cuda_uold);
  cudaFree(cuda_resid);
  printf("Total Number of Iterations: %ld\n", k);
  printf("Residual: %.15g\n", error);

}
