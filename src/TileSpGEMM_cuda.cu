
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

#include "TileSpGEMM_cuda.h"

#ifdef PROFILE
#include <nvtx3/nvToolsExtCuda.h>
#endif

namespace cuda {

CSRMatrix<double>* tile_spgemm(CSRMatrix<double>* A, CSRMatrix<double>* B) {
    TileSpMatrix<double>* tileA = new TileSpMatrix<double>(A);
    std::cout << tileA << std::endl;
    return nullptr;
}

}