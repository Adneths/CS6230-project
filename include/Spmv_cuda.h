#ifndef SPMV_CUDA_H
#define SPMV_CUDA_H

#include "typedef.h"

namespace cuda
{
    CSRMatrix<double> *spmv(CSRMatrix<double> *A, double *DenseVectorB);
}
#endif