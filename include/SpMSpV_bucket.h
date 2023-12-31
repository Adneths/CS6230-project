#ifndef SPMSPV_BUCKET_H
#define SPMSPV_BUCKET_H

#include "typedef.h"

namespace cuda {
LF_SpVector<double>* spmspv_bucket(CSCMatrix<double>* A, LF_SpVector<double>* B);
}
#endif