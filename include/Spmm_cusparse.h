#ifndef SPMM_CUSPARSE_H
#define SPMM_CUSPARSE_H

#include "typedef.h"

namespace cusparse
{
    CSRMatrix<double> *spmm(CSRMatrix<double> *A, dense_mat<double> *B);
}
#endif