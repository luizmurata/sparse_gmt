#include "utils.cuh"

std::tuple<std::vector<unsigned>, std::vector<unsigned>, std::vector<float>, size_t, size_t, size_t> 
load_entries(std::string path) {
    /* load file data */
    std::ifstream f(path);
    if (f.fail()) {
        throw std::invalid_argument("Error: File not found.");
    }

    while (f.peek() == '%')
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    size_t rows, cols, nnz;
    f >> rows >> cols >> nnz;

    std::vector<unsigned> x;
    std::vector<unsigned> y;
    std::vector<float> v;

    for (size_t c = 0; c < nnz; c++) {
        int32_t i, j;
        float value;
        f >> i >> j >> value;
        x.push_back(i-1);
        y.push_back(j-1);
        v.push_back(value);
    }

    f.close();
    return std::make_tuple(x, y, v, rows, cols, nnz);
}