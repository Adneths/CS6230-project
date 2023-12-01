#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>

#include "typedef.h"
#include "SpMSpV_cuda.h"
// #include "SpMSpV_cusparse.h"

extern "C" {
#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>
#include <bebop/smc/csr_matrix.h>
}


static bool compare(CSRMatrix<double>* a, CSRMatrix<double>* b, double epsilon = 0.00001) {
    if (a==nullptr || a->rowPtr==nullptr || a->dataCol==nullptr || a->dataVal==nullptr) {
        printf("Matrix A is invalid\n");
        return false;
    }
    if (b==nullptr || b->rowPtr==nullptr || b->dataCol==nullptr || b->dataVal==nullptr) {
        printf("Matrix B is invalid\n");
        return false;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Dimensions don't match A:(%dx%d) B:(%dx%d)\n", a->rows, a->cols, b->rows, b->cols);
        return false;
    }

    bool match = true;
    if (a->nnz != b->nnz) {
        printf("Number of non-zero terms don't match A:%d B:%d\n", a->nnz, b->nnz);
        match = false;
    }
    for (int i = 0; i < a->rows; i++) {
        int as = a->rowPtr[i], ae = a->rowPtr[i+1];
        int bs = b->rowPtr[i], be = b->rowPtr[i+1];
        while (as < ae && bs < be) {
            if (a->dataCol[as] == b->dataCol[bs]) {
                if (std::abs(a->dataVal[as] - b->dataVal[bs]) >= epsilon) {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, b->dataCol[bs], b->dataVal[bs]);
                    match = false;
                }
                as++; bs++;
            }
            else {
                if (a->dataCol[as] < b->dataCol[bs]) {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, a->dataCol[as], 0.0);
                    match = false;
                    as++;
                }
                else {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, b->dataCol[as], 0.0, i, b->dataCol[as], b->dataVal[as]);
                    match = false;
                    bs++;
                }
            }
        }
        if (as != ae && bs == be)
            while (as < ae) {
                printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, a->dataCol[as], 0.0);
                match = false;
                as++;
            }
        if (as == ae && bs != be)
            while (bs < be) {
                printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, b->dataCol[as], 0.0, i, b->dataCol[as], b->dataVal[as]);
                match = false;
                bs++;
            }
    }
    
    return match;
}

std::ostream& operator<<(std::ostream& stream, struct csr_matrix_t* mat) {
    stream << mat->m << "x" << mat->n << ": " << mat->nnz << std::endl;

    const double* const values = (const double* const) (mat->values);
    for (int r = 0; r < mat->m; r++) {
        int s = mat->rowptr[r];
        int e = mat->rowptr[r+1];
        for (int c = 0; c < mat->n; c++) {
            if (s < e && c == mat->colidx[s]) {
                stream << values[s] << "\t";
                s++;
            }
            else
                stream << "0\t";
        }
        if (r != mat->m-1) stream << std::endl;
    }
    return stream;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: ./<program> <harwell-boeing-file>");
        return 1;
    }
    const char *const filename = argv[1];
    enum sparse_matrix_file_format_t file_format = sparse_matrix_file_format_t::HARWELL_BOEING;
	struct sparse_matrix_t* spm = load_sparse_matrix(file_format, filename);
    if (spm == NULL)
        return 1;
    printf("%s: ", argv[1]);
	struct sparse_matrix_t* csc_spm = load_sparse_matrix(file_format, filename);
    if (csc_spm == NULL) {
        printf("NULL Loaded.\n");
        return 1;
    }
    printf("In main:\n");
    printf("spm is symmetric: %d\n",((struct csc_matrix_t*)(spm->repr))->symmetry_type);
    ((struct csc_matrix_t*)(spm->repr))->symmetry_type = UNSYMMETRIC; 
    printf("spm is symmetric: %d\n",((struct csc_matrix_t*)(spm->repr))->symmetry_type);

    CSCMatrix<double>* csc_matrix = new CSCMatrix<double>((struct csc_matrix_t*)csc_spm->repr);
    // printf("In main:\n");
    // printf("1.csc_matrix->cols: %d\n", csc_matrix->cols);
    // printf("csc_matrix addr:%d\n", csc_matrix);

    struct csr_matrix_t* csr_mat = csc_to_csr((struct csc_matrix_t*)spm->repr);
    // printf("2.csc_matrix->cols: %d\n", csc_matrix->cols);

    CSRMatrix<double>* matrix = new CSRMatrix<double>(csr_mat);

    matrix->info();
    SpVector<double> *sp_matdriven, *sp_cusparse, *sp_rand;
    LF_SpVector<double> *lfsp_vecdriven, *lfsp_rand;
    
    double sparsity = 0.15;
    int* ind_spvec = (int*)malloc(csc_matrix->cols * sizeof(int));
    double* data_spvec = (double*)malloc(csc_matrix->cols * sizeof(double));
    sp_rand = cuda::gen_rand_spvec(csc_matrix->cols, sparsity, ind_spvec, data_spvec);
    // sp_rand = SpVector(len, sparsity);
    // printf("3.csc_matrix->cols: %d\n", csc_matrix->cols);
    lfsp_rand = new LF_SpVector<double>(*sp_rand);
    // printf("4.csc_matrix->cols: %d\n", csc_matrix->cols);

    sp_matdriven = cuda::spmspv_naive_matdriven(matrix, sp_rand);

    // printf("In main:\n");
    // printf("csc_smp->repr->m: %d\n", ((struct csc_matrix_t*)(csc_spm->repr))->m);
    // printf("5.csc_matrix->cols: %d\n", csc_matrix->cols);
    // printf("spm != csc_spm: %d\n", static_cast<int>(spm!=csc_spm));
    // printf("csc_matrix addr:%d\n", csc_matrix);
    lfsp_vecdriven = cuda::spmspv_naive_vecdriven(csc_matrix, lfsp_rand);
    // result_cusparse = cusparse::spgemm(matrix, matrix);
    //std::cout << matrix << std::endl;
    //std::cout << result_cuda << std::endl;
    //std::cout << result_cusparse << std::endl;

    std::cout << "input_csr_mat:\n" << matrix << std::endl;
    // std::cout << "input csc_mat:\n" << csc_matrix << std::endl;
    std::cout << "input_spvec:\n" << sp_rand << std::endl;
    std::cout << "output_naive_matdriven\n" << sp_matdriven << std::endl;
    

    printf("Cuda Results: ");
    printf("matdriven output equals vecdriven output: %d\n",static_cast<int>(*sp_matdriven == *lfsp_vecdriven));
    // if (result_cuda) result_cuda->info(); else printf("nullptr\n");
    // printf("CuSparse Results: ");
    // if (result_cusparse) result_cusparse->info(); else printf("nullptr\n");
    // if (compare(result_cuda, result_cusparse))
    //     printf("Cuda Result matches CuSparse Result\n");
    // else
    //     printf("Cuda Result does not match CuSparse Result\n");


    delete matrix;
    delete csc_matrix;
    delete sp_matdriven;
    delete lfsp_vecdriven;
    // delete sp_rand;
    // delete lfsp_rand;

	return 0;
}