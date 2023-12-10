#ifndef SPMM_CUDA_H
#define SPMM_CUDA_H

#include "typedef.h"

namespace cuda
{
    CSRMatrix<double> *spmm(CSRMatrix<double> *A, dense_mat<double> *DenseMatrixB);
}
#endif