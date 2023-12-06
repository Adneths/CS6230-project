#ifndef TILESPGEMM_CUDA_H
#define TILESPGEMM_CUDA_H

#include "typedef.h"
#include <cstring>
#include <bitset>

template<typename T>
struct TileSpMatrix {
    int rows, cols, tileRows, tileCols, tileNnz;
    std::vector<int> tileRowPtr, tileColIdx, tileNnzs;
    std::vector<uint8_t> rowPtr, rowcolIdx;
    std::vector<double> vals;
    std::vector<uint16_t> masks;
    TileSpMatrix(CSRMatrix<T>* matrix) : rows(matrix->rows), cols(matrix->cols),
        tileRows((rows+15)/16), tileCols((cols+15)/16) {
        //rowcolIdx = (uint8_t*)malloc(matrix->nnzs * sizeof(uint8_t));
        //vals = (double*)malloc(matrix->nnzs * sizeof(double));
        rowcolIdx.resize(matrix->nnz);
        vals.resize(matrix->nnz);
        tileRowPtr.push_back(0);
        tileNnzs.push_back(0);

        int nnzCount = 0;
        for (int br = 0; br < rows; br += 16) {
            //std::cout << "br:" << br << std::endl;
            int index[16];
            std::memcpy(index, matrix->rowPtr+br, sizeof(int) * 16);

            /*std::cout << "index:";
            for (int i = 0; i < 16; i++)
                std::cout << index[i] << ", ";
            std::cout << std::endl;*/
            bool loop = false;
            for (int i = 0; i < 16; i++)
                if (index[i] < matrix->rowPtr[br+i+1])
                    loop = true;
            while (loop) {
                uint16_t mask[16]; int localNnz = 0;
                int bc = INT32_MAX;
                for (int i = 0; i < 16; i++)
                    if (index[i] < matrix->nnz && index[i] < matrix->rowPtr[br+i+1] && bc > matrix->dataCol[index[i]])
                        bc = matrix->dataCol[index[i]];
                std::memset(mask, 0, sizeof(uint16_t) * 16);
                //std::cout << "  bc:" << bc/16 << " (" << bc << ")" << std::endl;
                bc = 16*(bc/16);
                for (int tr = 0; tr < 16; tr++) {
                    rowPtr.push_back(localNnz);
                    int c;//, r = br+tr;
                    while (index[tr] < matrix->rowPtr[br+tr+1] && (c = matrix->dataCol[index[tr]]) < bc + 16) {
                        int tc = c - bc;
                        //std::cout << "    tr tc val:" << tr << ", " << tc << ", " << matrix->dataVal[index[tr]] << std::endl;
                        rowcolIdx[nnzCount] = uint8_t(tr<<4) | uint8_t(0x0f & tc);
                        vals[nnzCount] = matrix->dataVal[index[tr]];
                        mask[tr] |= 0b1 << (15 - tc);

                        index[tr]++; localNnz++; nnzCount++;
                    }
                }
                for (int i = 0; i < 16; i++)
                    masks.push_back(mask[i]);
                tileColIdx.push_back(bc/16);
                tileNnzs.push_back(nnzCount);

                loop = false;
                for (int i = 0; i < 16; i++)
                    if (index[i] < matrix->rowPtr[br+i+1])
                        loop = true;
                /*std::cout << "index:";
                for (int i = 0; i < 16; i++)
                    std::cout << index[i] << ", ";
                std::cout << std::endl;
                std::cout << "rowPtr:";
                for (int i = 0; i < 16; i++)
                    std::cout << matrix->rowPtr[br+i+1] << ", ";
                std::cout << std::endl;*/
            }
            tileRowPtr.push_back(tileColIdx.size()-1);
        }

        tileNnz = masks.size() / 16;
    }
    ~TileSpMatrix() { }
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, TileSpMatrix<T>* mat) {
    stream << "{ rows = " << mat->rows << ", cols = " << mat->cols << ", tileRows = " << mat->tileRows << ", tileCols = " << mat->tileCols << ", tileNnz = " << mat->tileNnz << "," << std::endl;
    stream << "tileRowPtr = { " << mat->tileRowPtr[0];
    for(int i = 1; i < mat->tileRowPtr.size(); i++)
        stream << ", " << mat->tileRowPtr[i];
    stream << " }" << std::endl;
    stream << "tileColIdx = { " << mat->tileColIdx[0];
    for(int i = 1; i < mat->tileColIdx.size(); i++)
        stream << ", " << mat->tileColIdx[i];
    stream << " }" << std::endl;
    stream << "tileNnzs = { " << mat->tileNnzs[0];
    for(int i = 1; i < mat->tileNnzs.size(); i++)
        stream << ", " << mat->tileNnzs[i];
    stream << " }" << std::endl;

    std::cout << mat->rowPtr.size() << std::endl;
    std::cout << mat->rowPtr[0] << std::endl;
    stream << "Tiles = {" << std::endl;
    for (int i = 0; i < mat->tileNnzs.size()-1; i++) {
        stream << "{" << std::endl << "rowPtr = { " << mat->rowPtr[16*i];
        for (int j = 1; j < 16; j++)
            stream << ", " << mat->rowPtr[16*i+j];
        stream << " }" << std::endl;

        stream << "rowIdx = { " << (mat->rowcolIdx[mat->tileNnzs[i]] >> 4);
        for (int j = mat->tileNnzs[i]+1; j < mat->tileNnzs[i+1]; j++)
            stream << ", " << (mat->rowcolIdx[j] >> 4);
        stream << " }" << std::endl;

        stream << "colIdx = { " << (mat->rowcolIdx[mat->tileNnzs[i]] & 0x0f);
        for (int j = mat->tileNnzs[i]+1; j < mat->tileNnzs[i+1]; j++)
            stream << ", " << (mat->rowcolIdx[j] & 0x0f);
        stream << " }" << std::endl;

        stream << "vals = { " << mat->vals[mat->tileNnzs[i]];
        for (int j = mat->tileNnzs[i]+1; j < mat->tileNnzs[i+1]; j++)
            stream << ", " << mat->vals[j];
        stream << " }" << std::endl;

        stream << "mask = { " << mat->masks[16*i];
        for (int j = 1; j < 16; j++) {
            stream << ", " << mat->masks[16*i+j];
        }
        stream << " }" << std::endl << "}";
    }

    return stream;
}

namespace cuda {
CSRMatrix<double>* tile_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B);
}
#endif
