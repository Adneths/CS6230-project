#ifndef SPGEMM_CUDA_H
#define SPGEMM_CUDA_H

#include "typedef.h"

namespace cuda {
CSRMatrix<double>* spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B);
}
#endif
