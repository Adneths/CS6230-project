#ifndef SPMM_GCOO_H
#define SPMM_GCOO_H

#include "typedef.h"

namespace GCOOSPMM
{
    CSRMatrix<double> *spmm(GCOO<double> *A, dense_mat<double> *DenseMatrixB);
}
#endif