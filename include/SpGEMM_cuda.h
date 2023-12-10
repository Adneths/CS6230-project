#ifndef SPGEMM_CUDA_H
#define SPGEMM_CUDA_H

#include "typedef.h"

#define MAX_GPU 8

namespace cuda {
CSRMatrix<double>* dacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B);
CSRMatrix<double>* sacc_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B);
}
#endif
