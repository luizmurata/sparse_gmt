#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cusparse.h>

#include "config.cuh"
#include "utils.cuh"
#include "coo.cuh"
#include "csr.cuh"

int main(int argc, char **argv)
{
    /* Check parameters passed */
    if (argc < 2) {
        std::cerr << "Error: Missing matrix data path." << std::endl;
        std::cerr << "Usage: ./transpose <path> [OPTIONAL: <device id>]" << std::endl;
        return -1;
    }
    
    /* Set cuda device used */
    int device = (argc > 2) ? atoi(argv[2]) : 0;
    CHECK_CUDA( cudaSetDevice(device) )

    /* Load matrix in COO format */
    auto matrix = load_coo_mm(argv[1]);
#ifdef DEBUG
    print_coo(matrix);
#endif

    /* Calculate data size */
    size_t com_bytes = matrix->nnz*sizeof(unsigned) + matrix->nnz*sizeof(unsigned) + matrix->nnz*sizeof(float);
    size_t unc_bytes = matrix->n_rows*matrix->n_cols*sizeof(float);
    std::cout << "Matrix of Dimensions [" << matrix->n_rows << "," << matrix->n_cols << "]" << std::endl;
    std::cout << "Compressed size: " << com_bytes/1024./1024. << " MB" << std::endl;
    std::cout << "Uncompressed: " << unc_bytes/1024./1024./1024. << " GB" << std::endl;
    std::cout << "Number of trials: " << TRIALS << " Block size: " << BLOCK_SIZE << std::endl;
    
    std::cout << std::endl << "COO Transposition" << std::endl;
    double total_time = 0.0;
    for (int i = 0; i < TRIALS; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto t = transpose_coo(matrix);
#ifdef DEBUG
        print_coo(t);
#endif
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        total_time += ms_double.count();
    }
    std::cout << "Bandwidth (compressed): " << (com_bytes*TRIALS / 1024. / 1024.) / (total_time / 1000.) << " MB/s" << std::endl;
    std::cout << "Bandwidth (uncompressed): " << (unc_bytes*TRIALS / 1024. / 1024. / 1024.) / (total_time / 1000.) << " GB/s" << std::endl;
    std::cout << "Total time: " << total_time/1000. << "s" << std::endl;

    /* CUSPARSE */
    auto handle = get_handle();
    auto matrix_csr = load_matrix_mm(argv[1]);
    matrix_csr->to_gpu(handle);
    
    std::cout << std::endl << "CSR Transposition" << std::endl;
    total_time = 0.0;
    for (int i = 0; i < TRIALS; i++)
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto t = transpose_cusparse(matrix_csr, handle);
#ifdef DEBUG
        print_csr(t);
#endif
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        total_time += ms_double.count();
    }
    std::cout << "Bandwidth (compressed): " << (com_bytes*TRIALS / 1024. / 1024.) / (total_time / 1000.) << " MB/s" << std::endl;
    std::cout << "Bandwidth (uncompressed): " << (unc_bytes*TRIALS / 1024. / 1024. / 1024.) / (total_time / 1000.) << " GB/s" << std::endl;
    std::cout << "Total time: " << total_time/1000. << "s" << std::endl;
}