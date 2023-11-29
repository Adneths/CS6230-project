#ifndef SPGEMM_CUSPARSE_H
#define SPGEMM_CUSPARSE_H

#include "typedef.h"

namespace cusparse {
CSRMatrix<double>* spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B);
}
#endif
