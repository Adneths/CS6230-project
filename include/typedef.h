#ifndef TYPEDEF_H
#define TYPEDEF_H

#include <iostream>
#include <vector>
#include <algorithm>

extern "C"
{
#include <bebop/smc/sparse_matrix.h>
#include <bebop/smc/sparse_matrix_ops.h>
#include <bebop/smc/csr_matrix.h>
}

template <typename T>
struct Vector
{
    int len;
    T *data;
    Vector() : len(0), data(nullptr) {}
    Vector(int len, T *data) : len(len), data(data) {}
};
template <typename T>
struct SpVector
{
    int len, nnz;
    int *ind;
    T *data;
    SpVector() : len(0), nnz(0), ind(nullptr), data(nullptr) {}
    SpVector(int len, int nnz, int *ind, T *data) : len(len), nnz(nnz), ind(ind), data(data) {}
    SpVector(int len, T *arr) : len(len), nnz(0)
    {
        nnz = 0;
        for (int i = 0; i < len; i++)
            if (arr[i] != 0)
                nnz++;
        ind = (int *)malloc(nnz * sizeof(int));
        data = (T *)malloc(nnz * sizeof(T));

        int t = 0;
        for (int i = 0; i < len; i++)
            if (arr[i] != 0)
            {
                ind[t] = i;
                data[t] = arr[i];
                t++;
            }
    }
};

template <typename T>
struct listformat_element
{
    int idx;
    T data;
    listformat_element() : idx(-1), data(0) {}
    listformat_element(int idx, T data) : idx(idx), data(data) {}
};

template <typename T>
struct LF_SpVector
{
    int len, nnz;
    struct listformat_element<T> *elements;
    LF_SpVector() : len(0), nnz(0), elements(nullptr) {}
    LF_SpVector(int len, int nnz, struct listformat_element<T> *elements) {}
    LF_SpVector(struct Vector<T> v) : len(v.len)
    {
        nnz = 0;
        for (int i = 0; i < len; i++)
        {
            if (v.data[i] != 0)
                nnz++;
        }
        elements = (struct listformat_element<T> *)malloc(nnz * sizeof(listformat_element<T>));
        int t = 0;
        for (int i = 0; i < len; i++)
        {
            if (v.data[i] != 0)
            {
                elements[t].idx = i;
                elements[t++].data = v.data[i];
            }
        }
    }
    LF_SpVector(int len, T *arr) : len(len)
    {
        nnz = 0;
        for (int i = 0; i < len; i++)
        {
            if (arr[i] != 0)
                nzz++;
        }
        elements = (struct listformat_element<T> *)malloc(nnz * sizeof(listformat_element<T>));
        int t = 0;
        for (int i = 0; i < len; i++)
        {
            if (arr[i] != 0)
            {
                elements[t].idx = i;
                elements[t++].data = v.data[i];
            }
        }
    }
    LF_SpVector(struct SpVector<T> v) : len(v.len), nnz(v.nnz) {
        elements = (struct listformat_element<T> *)malloc(nnz * sizeof(listformat_element<T>));
        for (int i = 0; i < nnz; i++) {
            elements[i].data = v.data[i];
            elements[i].idx = v.ind[i];
        }
    }
};

template <typename T>
struct Matrix
{
    int rows, cols;
    T *data;
    Matrix() : rows(0), cols(0), data(nullptr) {}
    Matrix(int rows, int cols, T *data) : rows(rows), cols(cols), data(data) {}
    void info()
    {
        printf("%dx%d\n", rows, cols);
    }
};
template <typename T>
struct CSRMatrix
{
    struct csr_matrix_t *src;
    int rows, cols, nnz;
    int *rowPtr;
    int *dataCol;
    T *dataVal;
    CSRMatrix() : src(nullptr), rows(0), cols(0), nnz(0), rowPtr(nullptr), dataCol(nullptr), dataVal(nullptr) {}
    CSRMatrix(int rows, int cols, int *rowPtr, int *dataCol, T *dataVal, int nnz)
        : src(nullptr), rows(rows), cols(cols), nnz(nnz), rowPtr(rowPtr), dataCol(dataCol), dataVal(dataVal) {}
    CSRMatrix(int rows, int cols, T *arr) : src(nullptr), rows(rows), cols(cols), nnz(0)
    {
        rowPtr = (int *)malloc((rows + 1) * sizeof(int));
        for (int i = 0; i < rows; i++)
        {
            rowPtr[i] = nnz;
            for (int j = 0; j < cols; j++)
            {
                if (arr[i * cols + j] != 0)
                {
                    nnz++;
                }
            }
        }
        rowPtr[rows] = nnz;
        dataCol = (int *)malloc(nnz * sizeof(int));
        dataVal = (T *)malloc(nnz * sizeof(T));

        int t = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (arr[i * cols + j] != 0)
                {
                    dataCol[t] = j;
                    dataVal[t] = arr[i * cols + j];
                    t++;
                }
            }
        }
    }
    CSRMatrix(struct csr_matrix_t *mat)
        : src(mat), rows(csr_matrix_num_rows(mat)), cols(csr_matrix_num_cols(mat)),
          nnz(mat->nnz), rowPtr(mat->rowptr), dataCol(mat->colidx), dataVal((T *)mat->values)
    {
        if (mat->value_type == value_type_t::PATTERN)
        {
            dataVal = (T *)malloc(nnz * sizeof(T));
            std::fill(dataVal, dataVal + nnz, 1);
        }
        std::fill(dataVal, dataVal + nnz, 1); // Testing
    }
    ~CSRMatrix()
    {
        if (this->src != nullptr)
        {
            if (src->value_type == value_type_t::PATTERN)
                free(dataVal);
            destroy_csr_matrix(src);
        }
        else
        {
            if (dataVal != nullptr)
                free(dataVal);
            if (dataCol != nullptr)
                free(dataCol);
            if (rowPtr != nullptr)
                free(rowPtr);
        }
    }
    void info()
    {
        printf("%dx%d (%d)\n", rows, cols, nnz);
    }
};

template <typename T>
struct CSCMatrix
{
    struct csc_matrix_t *src;
    int rows, cols, nnz;
    int *colPtr;
    int *dataRow;
    T *dataVal;
    CSCMatrix() : src(nullptr), rows(0), cols(0), nnz(0), colPtr(nullptr), dataRow(nullptr), dataVal(nullptr) {}
    CSCMatrix(int rows, int cols, int *colPtr, int *dataRow, T *dataVal, int nnz)
        : src(nullptr), rows(rows), cols(cols), nnz(nnz), colPtr(colPtr), dataRow(dataRow), dataVal(dataVal) {}
    CSCMatrix(int rows, int cols, T *arr) : src(nullptr), rows(rows), cols(cols), nnz(0)
    {
        colPtr = (int *)malloc((cols + 1) * sizeof(int));
        for (int i = 0; i < cols; i++)
        {
            colPtr[i] = nnz;
            for (int j = 0; j < rows; j++)
            {
                if (arr[j * cols + i] != 0)
                {
                    nnz++;
                }
            }
        }
        colPtr[cols] = nnz;
        dataRow = (int *)malloc(nnz * sizeof(int));
        dataVal = (T *)malloc(nnz * sizeof(T));

        int t = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (arr[j * cols + i] != 0)
                {
                    dataRow[t] = j;
                    dataVal[t] = arr[i * cols + j];
                    t++;
                }
            }
        }
    }
    CSCMatrix(struct csc_matrix_t *mat)
        : src(mat), rows(csc_matrix_num_rows(mat)), cols(csc_matrix_num_cols(mat)),
          nnz(mat->nnz), colPtr(mat->colptr), dataRow(mat->rowidx), dataVal((T *)mat->values)
    {
        if (mat->value_type == value_type_t::PATTERN)
        {
            dataVal = (T *)malloc(nnz * sizeof(T));
            std::fill(dataVal, dataVal + nnz, 1);
        }
        std::fill(dataVal, dataVal + nnz, 1); // Testing
    }
    ~CSCMatrix()
    {
        if (this->src != nullptr)
        {
            if (src->value_type == value_type_t::PATTERN)
                free(dataVal);
            destroy_csc_matrix(src);
        }
        else
        {
            if (dataVal != nullptr)
                free(dataVal);
            if (dataRow != nullptr)
                free(dataRow);
            if (colPtr != nullptr)
                free(colPtr);
        }
    }
    void info()
    {
        printf("%dx%d (%d)\n", rows, cols, nnz);
    }
};

template <typename T>
struct dense_vec
{
    int row_num;
    int cols = 1;
    T *vector;
    dense_vec(int row_num) : row_num(row_num), vector(new T[row_num])
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(-100, 100);

        for (int i = 0; i < row_num; i++)
        {
            vector[i] = dis(gen);
        }
    }

    ~dense_vec()
    {
        delete[] vector;
    }
};

template <typename T>
bool operator==(const CSRMatrix<T> &a, const CSRMatrix<T> &b)
{
    if (a->rowPtr == nullptr || a->dataCol == nullptr || a->dataVal == nullptr)
        return false;
    if (b->rowPtr == nullptr || b->dataCol == nullptr || b->dataVal == nullptr)
        return false;
    if (a->rows != b->rows || a->cols != b->cols || a->nnz != b->nnz)
        return false;

    for (int i = 0; i < a->rows + 1; i++)
        if (a->rowPtr[i] != b->rowPtr[i])
            return false;
    for (int i = 0; i < a->nnz; i++)
        if (a->dataCol[i] != b->dataCol[i] || a->dataVal[i] != b->dataVal[i])
            return false;
    return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, CSRMatrix<T> *mat)
{
    if (mat->rowPtr == nullptr || mat->dataCol == nullptr || mat->dataVal == nullptr)
        return stream << "Invalid Matrix";

    stream << mat->rows << "x" << mat->cols << ": " << mat->nnz << std::endl;

    for (int r = 0; r < mat->rows; r++)
    {
        int s = mat->rowPtr[r];
        int e = mat->rowPtr[r + 1];
        for (int c = 0; c < mat->cols; c++)
        {
            if (s < e && c == mat->dataCol[s])
            {
                stream << mat->dataVal[s] << "\t";
                s++;
            }
            else
                stream << "0\t";
        }
        if (r != mat->rows - 1)
            stream << std::endl;
    }
    return stream;
}

template <typename T>
bool operator==(const CSCMatrix<T> &a, const CSCMatrix<T> &b)
{
    if (a->colPtr == nullptr || a->dataRow == nullptr || a->dataVal == nullptr)
        return false;
    if (b->colPtr == nullptr || b->dataRow == nullptr || b->dataVal == nullptr)
        return false;
    if (a->rows != b->rows || a->cols != b->cols || a->nnz != b->nnz)
        return false;

    for (int i = 0; i < a->cols + 1; i++)
        if (a->colPtr[i] != b->colPtr[i])
            return false;
    for (int i = 0; i < a->nnz; i++)
        if (a->dataRow[i] != b->dataRow[i] || a->dataVal[i] != b->dataVal[i])
            return false;
    return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &stream, CSRMatrix<T> *mat)
{
    if (mat->colPtr == nullptr || mat->dataRow == nullptr || mat->dataVal == nullptr)
        return stream << "Invalid Matrix";

    stream << mat->rows << "x" << mat->cols << ": " << mat->nnz << std::endl;

    for (int r = 0; r < mat->cols; r++)
    {
        int s = mat->colPtr[r];
        int e = mat->colPtr[r + 1];
        for (int c = 0; c < mat->cols; c++)
        {
            if (s < e && c == mat->dataRow[s])
            {
                stream << mat->dataVal[s] << "\t";
                s++;
            }
            else
                stream << "0\t";
        }
        if (r != mat->cols - 1)
            stream << std::endl;
    }
    return stream;
}

template <typename T>
bool operator==(const SpVector<T> &a, const LF_SpVector<T> &b)
{
    if (a->ind == nullptr || a->data == nullptr)
        return false;
    if (b->elements == nullptr)
        return false;
    if (a->len != b->len || a->nnz != b->nnz)
        return false;

    for (int i = 0; i < a->len; i++)
        if (a->ind[i] != b->elements[i].idx)
            return false;
    for (int i = 0; i < a->nnz; i++)
        if (a->data[i] != b->elements[i].data)
            return false;
    return true;
}

#endif
