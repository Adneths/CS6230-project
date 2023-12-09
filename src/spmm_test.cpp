#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <iostream>

#include "typedef.h"
#include "Spmm_cuda.h"
#include "Spmm_gcoo.h"

extern "C"
{
#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>
#include <bebop/smc/csr_matrix.h>
}

static bool compare(CSRMatrix<double> *a, CSRMatrix<double> *b, double epsilon = 0.00001)
{
    if (a == nullptr || a->rowPtr == nullptr || a->dataCol == nullptr || a->dataVal == nullptr)
    {
        printf("Matrix A is invalid\n");
        return false;
    }
    if (b == nullptr || b->rowPtr == nullptr || b->dataCol == nullptr || b->dataVal == nullptr)
    {
        printf("Matrix B is invalid\n");
        return false;
    }
    if (a->rows != b->rows || a->cols != b->cols)
    {
        printf("Dimensions don't match A:(%dx%d) B:(%dx%d)\n", a->rows, a->cols, b->rows, b->cols);
        return false;
    }

    bool match = true;
    if (a->nnz != b->nnz)
    {
        printf("Number of non-zero terms don't match A:%d B:%d\n", a->nnz, b->nnz);
        match = false;
    }
    for (int i = 0; i < a->rows; i++)
    {
        int as = a->rowPtr[i], ae = a->rowPtr[i + 1];
        int bs = b->rowPtr[i], be = b->rowPtr[i + 1];
        while (as < ae && bs < be)
        {
            if (a->dataCol[as] == b->dataCol[bs])
            {
                if (std::abs(a->dataVal[as] - b->dataVal[bs]) >= epsilon)
                {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, b->dataCol[bs], b->dataVal[bs]);
                    match = false;
                }
                as++;
                bs++;
            }
            else
            {
                if (a->dataCol[as] < b->dataCol[bs])
                {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, a->dataCol[as], 0.0);
                    match = false;
                    as++;
                }
                else
                {
                    printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, b->dataCol[as], 0.0, i, b->dataCol[as], b->dataVal[as]);
                    match = false;
                    bs++;
                }
            }
        }
        if (as != ae && bs == be)
            while (as < ae)
            {
                printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, a->dataCol[as], a->dataVal[as], i, a->dataCol[as], 0.0);
                match = false;
                as++;
            }
        if (as == ae && bs != be)
            while (bs < be)
            {
                printf("A[%d,%d] = %f, B[%d,%d] = %f\n", i, b->dataCol[as], 0.0, i, b->dataCol[as], b->dataVal[as]);
                match = false;
                bs++;
            }
    }

    return match;
}

std::ostream &operator<<(std::ostream &stream, struct csr_matrix_t *mat)
{
    stream << mat->m << "x" << mat->n << ": " << mat->nnz << std::endl;

    const double *const values = (const double *const)(mat->values);
    for (int r = 0; r < mat->m; r++)
    {
        int s = mat->rowptr[r];
        int e = mat->rowptr[r + 1];
        for (int c = 0; c < mat->n; c++)
        {
            if (s < e && c == mat->colidx[s])
            {
                stream << values[s] << "\t";
                s++;
            }
            else
                stream << "0\t";
        }
        if (r != mat->m - 1)
            stream << std::endl;
    }
    return stream;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: ./<program> <harwell-boeing-file>");
        return 1;
    }
    const char *const filename = argv[1];
    enum sparse_matrix_file_format_t file_format = sparse_matrix_file_format_t::HARWELL_BOEING;
    struct sparse_matrix_t *spm = load_sparse_matrix(file_format, filename);
    if (spm == NULL)
        return 1;
    printf("%s: ", argv[1]);
    struct csr_matrix_t *csr_mat = csc_to_csr((struct csc_matrix_t *)spm->repr);
    CSRMatrix<double> *matrix = new CSRMatrix<double>(csr_mat);

    matrix->info();

    if (matrix == nullptr)
    {
        std::cerr << "Error: mat is nullptr" << std::endl;
    }
    else
    {
        std::cout << "mat is valid, rows: " << matrix->rows << std::endl;
    }

    GCOO<double> *gcoo_spm = new GCOO<double>(p_value);
    // std::cout << "input sparse matrix:\n";
    // std::cout << "row pointer:";
    // for (int i = 0; i < matrix->rows + 1; i++)
    // {
    //     std::cout << matrix->rowPtr[i] << ",";
    // }
    // std::cout << "\n";
    // std::cout << "data column:\n";
    // for (int i = 0; i < matrix->nnz; i++)
    // {
    //     std::cout << matrix->dataCol[i] << ",";
    // }
    // std::cout << "\n";
    // std::cout << "data value:\n";
    // for (int i = 0; i < matrix->nnz; i++)
    // {
    //     std::cout << matrix->dataVal[i] << ",";
    // }
    // std::cout << "\n";
    // std::cout << "\n";
    // std::cout << matrix << std::endl;
    // dense_mat<double> *dense_matrix(matrix->cols, matrix->cols);
    dense_mat<double> *dense_matrix = new dense_mat<double>(matrix->rows, matrix->cols);
    // std::cout << "dense matrix:\n";
    // for (int i = 0; i < matrix->cols; i++)
    // {
    //     std::cout << "column" << i << ":";
    //     for (int j = 0; j < matrix->rows; j++)
    //     {
    //         std::cout << dense_matrix->matrix[j * matrix->cols + i] << ",";
    //     }
    //     std::cout << "\n";
    // }
    // std::cout << "\n";
    // std::cout << "\n";
    CSRMatrix<double> *spmm_result_cuda, *spmm_result_cusparse, *spmm_result_gcoo;
    spmm_result_cuda = cuda::spmm(matrix, dense_matrix);
    spmm_result_gcoo = GCOOSPMM::spmm(gcoo_spm, dense_matrix);
    // spmm_result_cusparse = cusparse::spmv(matrix, dense_vector);
    // std::cout << result_cuda << std::endl;

    // std::cout << result_cusparse << std::endl;

    printf("SPMM Cuda Results: ");
    if (spmm_result_cuda)
        spmm_result_cuda->info();
    else
        printf("nullptr\n");

    printf("SPMM GCOO Results: ");
    if (spmm_result_gcoo)
        spmm_result_gcoo->info();
    else
        printf("nullptr\n");
    // printf("CuSparse Results: ");
    // if (result_cusparse)
    //     result_cusparse->info();
    // else
    //     printf("nullptr\n");
    // if (compare(result_cuda, result_cusparse))
    //     printf("Cuda Result matches CuSparse Result\n");
    // else
    //     printf("Cuda Result does not match CuSparse Result\n");

    // std::cout << "result sparse matrix:\n";
    // std::cout << "row pointer:";
    // for (int i = 0; i < spmm_result_cuda->rows + 1; i++)
    // {
    //     std::cout << spmm_result_cuda->rowPtr[i] << ",";
    // }
    // std::cout << "\n";
    // std::cout << "data column:";
    // for (int i = 0; i < spmm_result_cuda->nnz; i++)
    // {
    //     std::cout << spmm_result_cuda->dataCol[i] << ",";
    // }
    // std::cout << "\n";
    // std::cout << "data value:";
    // for (int i = 0; i < spmm_result_cuda->nnz; i++)
    // {
    //     std::cout << spmm_result_cuda->dataVal[i] << ",";
    // }
    delete matrix;
    delete spmm_result_cuda;
    delete dense_matrix;
    delete gcoo_spm;
    delete spmm_result_gcoo;
    // delete spmm_result_cusparse;
    return 0;
}