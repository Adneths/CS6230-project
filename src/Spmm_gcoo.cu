#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Spmm_gcoo.h"

#ifdef PROFILE
#include "timer.h"
#endif

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return nullptr;                                            \
        }                                                              \
    }

namespace GCOOSPMM
{
    // CUDA kernel function
    // *a thread block is responsible for a group calculation
    __global__ void GCOOSpMMKernel(double *values, int *cols, int *rows, int *gIdxes,
                                   int *nnzPerGroup, int wA, int hA, double *B, int wB, int hB, double *C)
    {
        // find the location of the thread in C's column
        int Cj = blockIdx.y * b_value + threadIdx.x;
        // find the "start" location of the thread in C's row
        int Ci0 = blockIdx.x * p_value;
        // a local array for storing the submatrix of C calculated by a thread
        double c[p_value] = {0};

        int CurGroupNNZ = nnzPerGroup[blockIdx.x];

        // find the "start" index of current group
        double *vals = values + gIdxes[blockIdx.x];
        int *co_cols = cols + gIdxes[blockIdx.x];
        int *co_rows = rows + gIdxes[blockIdx.x];

        // calculate the iteration of current group, b also means the number of thread in a thread block
        int iter = (CurGroupNNZ + b_value - 1) / b_value; // inorder to move all the data of a thread block into the shared memory

        // 遍历当前组中的所有非零元素
        for (int i = 0; i < iter; ++i)
        {
            int cooOffset = i * b_value; // 计算COO偏移量

            __shared__ double sVals[b_value];
            __shared__ int sCols[b_value];
            __shared__ int sRows[b_value];

            // 加载当前组的COO数据到共享内存
            if (threadIdx.x < b_value && (cooOffset + threadIdx.x) < CurGroupNNZ)
            { // 确保不会超出当前组的大小
                sVals[threadIdx.x] = vals[cooOffset + threadIdx.x];
                sCols[threadIdx.x] = co_cols[cooOffset + threadIdx.x];
                sRows[threadIdx.x] = co_rows[cooOffset + threadIdx.x];
            }
            else
            {
                // 超出部分的线程不加载任何数据
                sVals[threadIdx.x] = 0;
                sCols[threadIdx.x] = -1; // 使用-1来标记无效的索引
                sRows[threadIdx.x] = -1;
            }
            __syncthreads(); // 确保所有线程都加载完数据

            // 如果当前线程对应的列索引在B矩阵范围内，则执行计算
            if (Cj < wB)
            {
                for (int j = 0; j < b_value && sCols[j] != -1; ++j)
                {
                    int col = sCols[j];
                    int row = sRows[j];
                    double av = sVals[j];
                    if (col == -1)
                        continue; // 跳过无效数据
                    double bv = B[col * wB + Cj];
                    int outIdx = row % p_value;
                    c[outIdx] += av * bv; // 执行乘加操作

                    // 内部循环，用于寻找可以重用bv值的连续非零元素
                    int k = 1;
                    while (j + k < b_value && sCols[j + k] == col)
                    {
                        if (j + k >= CurGroupNNZ)
                            break;
                        av = sVals[j + k];
                        row = sRows[j + k];
                        outIdx = row % p_value;
                        c[outIdx] += av * bv;
                        k++;
                    }
                    j += k - 1; // 跳过已经处理的元素
                }
            }
            __syncthreads(); // 等待所有线程完成这一轮计算
        }

        // 将中间结果数组c中的值写回到全局内存的C矩阵
        if (Cj < wB)
        {
            for (int i = 0; i < p_value; ++i)
            {
                C[(Ci0 + i) * wB + Cj] = c[i];
                printf("c[%d]=%f\n", i, c[i]);
            }
        }
    }

    CSRMatrix<double> *spmm(GCOO<double> *A, dense_mat<double> *DenseMatrixB)
    {
#ifdef PROFILE
        Timer timer;
        auto time = timer.tick();
#endif

        double *d_Avalues, *d_Bvalues, *d_Cvalues, *h_Cvalues;
        int *d_Acols, *d_Arows, *d_AgIdxes, *d_AnnzPerGroup;

        size_t
            rowPtrA_size = (A->nnz) * sizeof(int),
            colPtrA_size = (A->nnz) * sizeof(int),
            dataValA_size = (A->nnz) * sizeof(double),
            gIdxesA_size = (A->num_group) * sizeof(int),
            nnzPerGroupA_size = (A->num_group) * sizeof(int),
            DenseMatrixB_size = (DenseMatrixB->total_size) * sizeof(double),
            dataValC_size = (A->num_row * DenseMatrixB->col_num) * sizeof(double);

        CHECK_CUDA(cudaMalloc(&d_Arows, rowPtrA_size));
        CHECK_CUDA(cudaMalloc(&d_Acols, colPtrA_size));
        CHECK_CUDA(cudaMalloc(&d_Avalues, dataValA_size));
        CHECK_CUDA(cudaMalloc(&d_AgIdxes, gIdxesA_size));
        CHECK_CUDA(cudaMalloc(&d_AnnzPerGroup, nnzPerGroupA_size));
        CHECK_CUDA(cudaMalloc(&d_Bvalues, DenseMatrixB_size));
        CHECK_CUDA(cudaMalloc(&d_Cvalues, dataValC_size));

        cudaMemcpy(d_Arows, A->rows, rowPtrA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Acols, A->cols, colPtrA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Avalues, A->values, dataValA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_AgIdxes, A->gIdexs, gIdxesA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_AnnzPerGroup, A->nnzpergroup, nnzPerGroupA_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Bvalues, DenseMatrixB->matrix, DenseMatrixB_size, cudaMemcpyHostToDevice);
        cudaMemset(d_Cvalues, 0, dataValC_size);

        // dim3 threadsPerBlock(b_value);
        dim3 threadsPerBlock(64);
        dim3 numBlocks((A->num_row + p_value - 1) / p_value, (DenseMatrixB->col_num + b_value - 1) / b_value);
#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Setup: " << time << std::endl;
        timer.tick();
#endif

        GCOOSpMMKernel<<<numBlocks, threadsPerBlock>>>(d_Avalues, d_Acols, d_Arows, d_AgIdxes, d_AnnzPerGroup,
                                                       A->num_col, A->num_row, d_Bvalues, DenseMatrixB->col_num,
                                                       DenseMatrixB->row_num, d_Cvalues);
        cudaDeviceSynchronize();
#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Compute: " << time << std::endl;
        timer.tick();
#endif

        h_Cvalues = (double *)malloc(dataValC_size);
        cudaMemcpy(h_Cvalues, d_Cvalues, dataValC_size, cudaMemcpyDeviceToHost);
        CSRMatrix<double> *ret = new CSRMatrix<double>(A->num_row, DenseMatrixB->col_num, h_Cvalues);

        cudaFree(d_Avalues);
        cudaFree(d_Bvalues);
        cudaFree(d_Cvalues);
        cudaFree(d_Acols);
        cudaFree(d_Arows);
        cudaFree(d_AgIdxes);
        cudaFree(d_AnnzPerGroup);

#ifdef PROFILE
        time = timer.tick();
        std::cout << "Cuda Teardown: " << time << std::endl;
#endif

        return ret;
    }

}
