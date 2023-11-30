#ifndef SPMSPV_CUDA_H
#define SPMSPV_CUDA_H

#include "typedef.h"

namespace cuda {
SpVector<double>* spmspv_naive_matdriven(CSRMatrix<double>* A, SpVector<double>* B);

LF_SpVector<double>* spmspv_naive_vecdriven(CSCMatrix<double>* A, LF_SpVector<double>* B);

// SpVector<double>* gen_rand_spvec(int len, double sparsity);
}
#endif
