#pragma once
#include <memory>
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>

#include "config.cuh"
#include "utils.cuh"

struct COO {
    /* CPU data */
    unsigned *x, *y;
    float *v;
    size_t n_rows;
    size_t n_cols;
    size_t nnz;

    /* GPU data */
    float in_gpu = false;
    unsigned *d_x, *d_y;
    float *d_v;

    ~COO();
};

std::shared_ptr<COO> load_coo_mm(std::string path);
std::tuple<std::shared_ptr<COO>, float> transpose_coo(std::shared_ptr<COO> matrix);
void print_coo(std::shared_ptr<COO> matrix);