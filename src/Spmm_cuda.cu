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
        __global__ void spmm_kx(int *rowPtrA, int *dataColA, double *dataValA,
                                double *dense_matrix, int num_colB,
                                double *dataValC)
        {
                int tx = threadIdx.x;
                int bx = blockIdx.x;
                if (tx < num_colB)
                {
                        int as = rowPtrA[bx];
                        int ae = rowPtrA[bx + 1];
                        for (int i = as; i < ae; i++)
                        {
                                double valA = dataValA[i];
                                int c = dataColA[i];

                                double valB = dense_matrix[c * num_colB + tx];

                                dataValC[bx * num_colB + tx] += valA * valB;
                        }
                }
        }

        CSRMatrix<double> *spmm(CSRMatrix<double> *A, dense_mat<double> *DenseMatrixB)
        {
#ifdef PROFILE
                Timer timer;
                auto time = timer.tick();
#endif
                int *d_rowPtrA, *d_dataColA;
                double *d_dataValA, *MatrixOnGpu, *d_dataValC, *h_dataValC;

                size_t
                    rowPtrA_size = (A->rows + 1) * sizeof(int),
                    dataColA_size = (A->nnz) * sizeof(int),
                    dataValA_size = (A->nnz) * sizeof(double),
                    DenseMatrixB_size = (DenseMatrixB->total_size) * sizeof(double),
                    dataValC_size = (A->rows * DenseMatrixB->col_num) * sizeof(double);

                cudaMalloc(&d_rowPtrA, rowPtrA_size);
                cudaMalloc(&d_dataColA, dataColA_size);
                cudaMalloc(&d_dataValA, dataValA_size);
                cudaMalloc(&MatrixOnGpu, DenseMatrixB_size);
                cudaMalloc(&d_dataValC, dataValC_size);

                cudaMemcpy(d_rowPtrA, A->rowPtr, rowPtrA_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_dataColA, A->dataCol, dataColA_size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_dataValA, A->dataVal, dataValA_size, cudaMemcpyHostToDevice);
                cudaMemcpy(MatrixOnGpu, DenseMatrixB->matrix, DenseMatrixB_size, cudaMemcpyHostToDevice);
                cudaMemset(d_dataValC, 0, dataValC_size);

                dim3 threadsPerBlock(32 * ((DenseMatrixB->col_num + 31) / 32));
                dim3 numBlocks(A->rows);
#ifdef PROFILE
                time = timer.tick();
                std::cout << "Cuda Setup: " << time << std::endl;
                timer.tick();
#endif

                spmm_kx<<<numBlocks, threadsPerBlock>>>(d_rowPtrA, d_dataColA, d_dataValA,
                                                        MatrixOnGpu, DenseMatrixB->col_num,
                                                        d_dataValC);
                cudaDeviceSynchronize();
#ifdef PROFILE
                time = timer.tick();
                std::cout << "Cuda Compute: " << time << std::endl;
                timer.tick();
#endif

                h_dataValC = (double *)malloc(dataValC_size);
                cudaMemcpy(h_dataValC, d_dataValC, dataValC_size, cudaMemcpyDeviceToHost);
                CSRMatrix<double> *ret = new CSRMatrix<double>(A->rows, DenseMatrixB->col_num, h_dataValC);

                cudaFree(d_rowPtrA);
                cudaFree(d_dataColA);
                cudaFree(d_dataValA);
                cudaFree(MatrixOnGpu);
                cudaFree(d_dataValC);
#ifdef PROFILE
                time = timer.tick();
                std::cout << "Cuda Teardown: " << time << std::endl;
#endif

                return ret;
        }

}
