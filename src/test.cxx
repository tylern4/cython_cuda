#include "wrapper.hpp"
#include <iostream>

int main(int argc, char const *argv[])
{
    long n = DEFAULT_DIMSIZE;
    long m = DEFAULT_DIMSIZE;
    REAL alpha = 0.0543;
    REAL tol = 0.0000000001;
    REAL relax = 1.0;
    int mits = 10000;

    REAL *u = (REAL *)malloc(sizeof(REAL) * n * m);
    REAL *f = (REAL *)malloc(sizeof(REAL) * n * m);

    REAL dx; /* grid spacing in x direction */
    REAL dy; /* grid spacing in y direction */

    initialize(n, m, alpha, &dx, &dy, u, f);
    jacobi_cuda(n, m, dx, dy, alpha, relax, u, f, tol, mits);

    return 0;
}
