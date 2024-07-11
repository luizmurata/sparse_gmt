#pragma once
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

#include "utils.cuh"

struct CSR {
    std::string name;
    /* CPU Data */
    std::vector<int32_t> col_index;
    std::vector<int32_t> row_index;
    std::vector<float>   v;
    int32_t rows;
    int32_t cols;
    int32_t nnz;

    /* GPU Data */
    bool in_gpu = false;
    int32_t *d_cols;
    int32_t *d_rows;
    float   *d_v;
    cusparseSpMatDescr_t desc;

    void to_gpu(cusparseHandle_t handle);
    ~CSR();
};

cusparseHandle_t get_handle();
std::shared_ptr<CSR> load_matrix_mm(std::string path);
std::shared_ptr<CSR> transpose_cusparse(std::shared_ptr<CSR> matrix, cusparseHandle_t handle);
void print_csr(std::shared_ptr<CSR> matrix);