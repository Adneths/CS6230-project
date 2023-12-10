#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

extern "C" {
#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>
#include <bebop/smc/csr_matrix.h>
}


template<typename T>
struct Vector {
    int len;
    T* data;
    Vector() : len(0), data(nullptr) {}
    Vector(int  len, T* data) : len(len), data(data) {}
};
template<typename T>
struct SpVector {
    int len, nnz;
    int* ind;
    T* data;
    SpVector() : len(0), nnz(0), ind(nullptr), data(nullptr) {}
    SpVector(int len, int nnz, int* ind, T* data) : len(len), nnz(nnz), ind(ind), data(data) {}
    SpVector(int len, T* arr) : len(len), nnz(0) {
        nnz = 0;
        for (int i = 0; i < len; i++)
            if (arr[i] != 0)
                nnz++;
        ind = (int*)malloc(nnz * sizeof(int));
        data = (T*)malloc(nnz * sizeof(T));

        int t = 0;
        for (int i = 0; i < len; i++)
            if (arr[i] != 0) {
                ind[t] = i;
                data[t] = arr[i];
            }
    }
};

typedef enum {ROW_MAJOR, COL_MAJOR} Order;
template<typename T>
struct Matrix {
    int rows, cols;
    T* data;
    Order order;
    Matrix() : rows(0), cols(0), data(nullptr), order(ROW_MAJOR) {}
    Matrix(int rows, int cols, T* data, Order order=ROW_MAJOR) : rows(rows), cols(cols), data(data), order(order) {}
    void info() {
        printf("%dx%d\n", rows, cols);
    }
};
template<typename T>
struct CSRMatrix {
    struct csr_matrix_t* src;
    int rows, cols, nnz;
    int* rowPtr;
    int* dataCol;
    T* dataVal;
    CSRMatrix() : src(nullptr), rows(0), cols(0), nnz(0), rowPtr(nullptr), dataCol(nullptr), dataVal(nullptr) {}
    CSRMatrix(int rows, int cols, int* rowPtr, int* dataCol, T* dataVal, int nnz)
        : src(nullptr), rows(rows), cols(cols), nnz(nnz), rowPtr(rowPtr), dataCol(dataCol), dataVal(dataVal) {}
    CSRMatrix(int rows, int cols, T* arr) : src(nullptr), rows(rows), cols(cols), nnz(0) {
        rowPtr = (int*)malloc((rows+1) * sizeof(int));
        for (int i = 0; i < rows; i++) {
            rowPtr[i] = nnz;
            for (int j = 0; j < cols; j++) {
                if (arr[i*cols+j] != 0) {
                    nnz++;
                }
            }
        }
        rowPtr[rows] = nnz;
        dataCol = (int*)malloc(nnz * sizeof(int));
        dataVal = (T*)malloc(nnz * sizeof(T));

        int t = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (arr[i*cols+j] != 0) {
                    dataCol[t] = j;
                    dataVal[t] = arr[i*cols+j];
                    t++;
                }
            }
        }
    }
    CSRMatrix(struct csr_matrix_t* mat)
        : src(mat), rows(csr_matrix_num_rows(mat)), cols(csr_matrix_num_cols(mat)),
        nnz(mat->nnz), rowPtr(mat->rowptr), dataCol(mat->colidx), dataVal((T*)mat->values)
    {
        if (mat->value_type == value_type_t::PATTERN) {
            dataVal = (T*)malloc(nnz * sizeof(T));
            std::fill(dataVal, dataVal+nnz, 1);
        }
    }
    ~CSRMatrix() {
        if (this->src != nullptr) {
            if (src->value_type == value_type_t::PATTERN)
                free(dataVal);
            destroy_csr_matrix(src);
        }
        else {
            if(dataVal != nullptr)
                free(dataVal);
            if(dataCol != nullptr)
                free(dataCol);
            if(rowPtr != nullptr)
                free(rowPtr);
        }
    }
    void transpose() {
        int* numRows = (int*)malloc((cols+1) * sizeof(int));
        std::memset(numRows, 0, sizeof(int) * (cols+1));
        for (int i = 0; i < nnz; i++)
            numRows[dataCol[i]+1]++;
        for (int i = 2; i < cols; i++)
            numRows[i] += numRows[i-1];
        int* nDataCol = (int*)malloc(nnz * sizeof(int));
        double* nDataVal = (double*)malloc(nnz * sizeof(double));
        for (int i = 0; i < rows; i++)
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                nDataCol[numRows[dataCol[j]]] = i;
                nDataVal[numRows[dataCol[j]]] = dataVal[j];
                numRows[dataCol[j]]++;
            }
        for (int i = rows; i > 0; i--)
            numRows[i] = numRows[i-1];
        numRows[0] = 0;

        this->replace(nnz, numRows, nDataCol, nDataVal);
    }
    void resparse() {
        int nnnz = 0;
        for (int i = 0; i < nnz; i++)
            if (dataVal[i] != 0)
                nnnz++;
        int* nRowPtr = (int*)malloc((rows+1) * sizeof(int));
        int* nDataCol = (int*)malloc(nnnz * sizeof(int));
        double* nDataVal = (double*)malloc(nnnz * sizeof(double));
        
        nnnz = 0;
        for (int i = 0; i < rows; i++){
            nRowPtr[i] = nnnz;
            for (int j = rowPtr[i]; j < rowPtr[i+1]; j++) {
                if (dataVal[j] != 0)
                {
                    nDataCol[nnnz] = dataCol[j];
                    nDataVal[nnnz] = dataVal[j];
                    nnnz++;
                }
            }
        }
        nRowPtr[rows] = nnnz;
        this->replace(nnnz, nRowPtr, nDataCol, nDataVal);
    }
    void replace(int nnnz, int* nRowPtr, int* nDataCol, double* nDataVal) {
        if (src != nullptr) {
            if (src->value_type == value_type_t::PATTERN)
                free(dataVal);
            destroy_csr_matrix(src);
        }
        else {
            free(this->rowPtr);
            free(this->dataCol);
            free(this->dataVal);
        }
        this->rowPtr = nRowPtr;
        this->dataCol = nDataCol;
        this->dataVal = nDataVal;
        this->nnz = nnnz;
    }
    void info() {
        printf("%dx%d (%d)\n", rows, cols, nnz);
    }
};

template <typename T>
bool operator==(const CSRMatrix<T>& a, const CSRMatrix<T>& b) {
    if (a->rowPtr==nullptr || a->dataCol==nullptr || a->dataVal==nullptr)
        return false;
    if (b->rowPtr==nullptr || b->dataCol==nullptr || b->dataVal==nullptr)
        return false;
    if (a->rows != b->rows || a->cols != b->cols || a->nnz != b->nnz)
        return false;

    for (int i = 0; i < a->rows+1; i++)
        if (a->rowPtr[i] != b->rowPtr[i])
            return false;
    for (int i = 0; i < a->nnz; i++)
        if (a->dataCol[i] != b->dataCol[i] || a->dataVal[i] != b->dataVal[i])
            return false;
    return true;
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, CSRMatrix<T>* mat) {
    if (mat->rowPtr==nullptr || mat->dataCol==nullptr || mat->dataVal==nullptr)
        return stream << "Invalid Matrix";
    /*
    stream << mat->rows << "x" << mat->cols << ": " << mat->nnz << std::endl;

    for (int r = 0; r < mat->rows; r++) {
        int s = mat->rowPtr[r];
        int e = mat->rowPtr[r+1];
        for (int c = 0; c < mat->cols; c++) {
            if (s < e && c == mat->dataCol[s]) {
                stream << mat->dataVal[s] << "\t";
                s++;
            }
            else
                stream << "0\t";
        }
        if (r != mat->rows-1) stream << std::endl;
    }*/
    stream << "{ rows = " << mat->rows << ", cols = " << mat->cols << ", nnz = " << mat->nnz << "," << std::endl;
    stream << "rowPtr = { " << mat->rowPtr[0];
    for(int i = 1; i < mat->rows+1; i++)
        stream << ", " << mat->rowPtr[i];
    stream << " }" << std::endl;
    stream << "dataCol = { " << mat->dataCol[0];
    for(int i = 1; i < mat->nnz; i++)
        stream << ", " << mat->dataCol[i];
    stream << " }" << std::endl;
    stream << "dataVal = { " << mat->dataVal[0];
    for(int i = 1; i < mat->nnz; i++)
        stream << ", " << mat->dataVal[i];
    stream << " }" << std::endl << "}";
    return stream;
}

#endif
