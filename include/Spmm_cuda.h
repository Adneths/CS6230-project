#ifndef SPMV_CUDA_H
#define SPMV_CUDA_H

#include "typedef.h"

namespace cuda
{
    CSRMatrix<double> *spmm(CSRMatrix<double> *A, double *DenseMatrixB);
}
#endif