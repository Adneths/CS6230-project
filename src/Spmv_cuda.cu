#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "SpGEMM_cuda.h"

#ifdef PROFILE
#include "timer.h"
#endif

namespace cuda
{

    // Sparse General Matrix Multiplication with Dense Accumulator
    __global__ void spmv_kx(int *rowPtrA, int *dataColA, double *dataValA,
                            double *dense_vector,
                            double *dataValC)
    {
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        if (tx < colsB)
        {
            int as = rowPtrA[bx];
            int ae = rowPtrA[bx + 1];
            for (int i = as; i < ae; i++)
            {
                double valA = dataValA[i];
                int c = dataColA[i];

                int bs = rowPtrB[c];
                int be = rowPtrB[c + 1];
                if (bs + tx < be)
                {
                    dataValC[bx * colsB + dataColB[bs + tx]] += valA * dataValB[bs + tx];
                }
            }
        }
    }

    CSRMatrix<double> *spmv(CSRMatrix<double> *A, double *DenseVectorB)
    {
#ifdef PROFILE
        Timer timer;
        auto time = timer.tick();
#endif
        int *d_rowPtrA, *d_dataColA;
        double *d_dataValA, *VectorOnGpu, *d_dataValC, *h_dataValC;

        size_t
            rowPtrA_size = (A->rows + 1) * sizeof(int),
            dataColA_size = (A->nnz) * sizeof(int),
            dataValA_size = (A->nnz) * sizeof(double),
            DenseVectorB_size = (DenseVectorB->row_num) * sizeof(double),
            dataValC_size = (A->rows * DenseVectorB->cols) * sizeof(double);

        cudaMalloc(&d_rowPtrA, rowPtrA_size);
        cudaMalloc(&d_dataColA, dataColA_size);
        cudaMalloc(&d_dataValA, dataValA_size);
        cudaMalloc(&VectorOnGpu, DenseVectorB_size);
        cudaMalloc(&d_dataValC, dataValC_size);

        cudaMemcpy(d_rowPtrA, A->rowPtr, rowPtrA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(VectorOnGpu, DenseVectorB, DenseVectorB_size, cudaMemcpyHostToDevice);
        cudaMemset(d_dataValC, 0, dataValC_size);

        dim3 threadsPerBlock(32 * ((B->cols + 31) / 32));
        dim3 numBlocks(A->rows);
#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Setup: " << time << std::endl;
        timer.tick();
#endif

        spmv_kx<<<numBlocks, threadsPerBlock>>>(d_rowPtrA, d_dataColA, d_dataValA,
                                                DenseVectorB,
                                                d_dataValC);
#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Compute: " << time << std::endl;
        timer.tick();
#endif

        h_dataValC = (double *)malloc(dataValC_size);
        cudaMemcpy(h_dataValC, d_dataValC, dataValC_size, cudaMemcpyDeviceToHost);
        CSRMatrix<double> *ret = new CSRMatrix<double>(A->rows, DenseVectorB->cols, h_dataValC);

        cudaFree(d_rowPtrA);
        cudaFree(d_dataColA);
        cudaFree(d_dataValA);
        cudaFree(VectorOnGpu);
        cudaFree(d_dataValC);
#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Teardown: " << time << std::endl;
#endif

        return ret;
    }

}
