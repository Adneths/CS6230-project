#ifndef SPMM_GCOO_H
#define SPMM_GCOO_H

#include "typedef.h"

// hyper-paratemers: is dealed by a thread block
#define b_value 64 // number of columns in a group in dense matrix = number of threads in a thread block
#define p_value 64 // number of rows in a group in sparse matrix

namespace GCOOSPMM
{
    CSRMatrix<double> *spmm(GCOO<double> *A, dense_mat<double> *DenseMatrixB);
}
#endif