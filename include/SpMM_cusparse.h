#ifndef SPMM_CUSPARSE_H
#define SPMM_CUSPARSE_H

#include "typedef.h"

namespace cusparse {
Matrix<double>* spmm(CSRMatrix<double>* A, Matrix<double>* B);
}
#endif
