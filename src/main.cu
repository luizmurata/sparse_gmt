#include <algorithm>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>

struct CSR {
    std::vector<size_t> col_index;
    std::vector<size_t> row_index;
    std::vector<float>  v;
};

struct COO_entry {
    size_t i;
    size_t j;
    float v;
};

CSR load_matrix_mm(std::string path) {
    /* load file data */
    std::ifstream f(path);
    size_t rows, cols, nnz;
    std::vector<COO_entry> entries;

    while (f.peek() == '%')
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    f >> rows >> cols >> nnz;

    for (size_t c = 0; c < nnz; c++) {
        size_t i, j;
        float value;
        f >> i >> j >> value;
        entries.push_back(COO_entry{i-1, j-1, value});
    }

    f.close();

    /* sort COO entries */
    auto compare = [](const COO_entry& a, const COO_entry& b) {
        if (a.i < b.i) return true;
        if (a.i == b.i) return a.j < b.j;
        return false;
    };
    std::sort(entries.begin(), entries.end(), compare);

    /* generate CSR matrix from sorted COO */
    CSR out;
    out.row_index = std::vector<size_t>(rows+1, 0);
    for (size_t i = 0; i < nnz; i++) {
        auto entry = entries[i];
        out.row_index[entry.i + 1] += 1;
        out.col_index.push_back(entry.j);
        out.v.push_back(entry.v);
    }
    for (size_t i = 0; i < rows; i++) {
        out.row_index[i + 1] += out.row_index[i];
    }

    return out;
}

int main(int argc, char **argv) {
    auto matrix = load_matrix_mm("../dataset/1138_bus.mtx");
    std::cout << matrix.col_index.size() << std::endl;
    std::cout << matrix.row_index.size() << std::endl;
    std::cout << matrix.v.size() << std::endl;
    std::cout << matrix.row_index[4] << std::endl;
    std::cout << matrix.col_index[4] << std::endl;
    std::cout << matrix.v[0] << std::endl;
    std::cout << matrix.v[1] << std::endl;
    std::cout << matrix.v[2] << std::endl;
    std::cout << matrix.v[3] << std::endl;
    std::cout << matrix.v[4] << std::endl;
}
