#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cusparse.h>

#define TRIALS 1000

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

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

cusparseHandle_t get_handle() {
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    return handle;
}

struct COO_entry {
    int32_t i;
    int32_t j;
    float   v;
};

std::shared_ptr<CSR> load_matrix_mm(std::string path) {
    /* load file data */
    std::ifstream f(path);
    if (f.fail()) {
        throw std::invalid_argument("Error: File not found.");
    }

    while (f.peek() == '%')
        f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    CSR out;
    size_t rows, cols, nnz;
    std::vector<COO_entry> entries;
    f >> rows >> cols >> nnz;
    out.rows = rows;
    out.cols = cols;
    out.nnz = nnz;

    for (size_t c = 0; c < nnz; c++) {
        int32_t i, j;
        float vue;
        f >> i >> j >> vue;
        entries.push_back(COO_entry{i-1, j-1, vue});
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
    out.row_index = std::vector<int32_t>(rows+1, 0);
    for (size_t i = 0; i < nnz; i++) {
        auto entry = entries[i];
        out.row_index[entry.i + 1] += 1;
        out.col_index.push_back(entry.j);
        out.v.push_back(entry.v);
    }
    for (size_t i = 0; i < rows; i++) {
        out.row_index[i + 1] += out.row_index[i];
    }

    return std::make_shared<CSR>(out);
}

void CSR::to_gpu(cusparseHandle_t handle) {
    /* Load to GPU Memory */
    int *d_rows, *d_cols;
    float *d_v;
    CHECK_CUDA( cudaMalloc((void**) &d_cols, this->col_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_rows, this->row_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_v,    this->v.size() * sizeof(float)) );

    CHECK_CUDA( 
            cudaMemcpy(
                    d_cols, 
                    this->col_index.data(),
                    this->col_index.size() * sizeof(int),
                    cudaMemcpyHostToDevice
            )
    );

    CHECK_CUDA( 
            cudaMemcpy(
                    d_rows, 
                    this->row_index.data(),
                    this->row_index.size() * sizeof(int),
                    cudaMemcpyHostToDevice
            )
    );

    CHECK_CUDA( 
            cudaMemcpy(
                    d_v,
                    this->v.data(),
                    this->v.size() * sizeof(float),
                    cudaMemcpyHostToDevice
            )
    );

    /* Load to cuSPARSE */
    cusparseSpMatDescr_t desc;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    CHECK_CUSPARSE(
            cusparseCreateCsr(
                &desc,
                this->rows,
                this->cols,
                this->v.size(),
                d_rows, d_cols,
                d_v,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
            )
    )

    this->d_rows = d_rows;
    this->d_cols = d_cols;
    this->d_v = d_v;
    this->desc = desc;
    this->in_gpu = true;
}

CSR::~CSR() {
    if (this->in_gpu) {
        std::cout << this->name << ": Freeing memory..." << std::endl;
        CHECK_CUDA( cudaFree(this->d_cols) )
        CHECK_CUDA( cudaFree(this->d_rows) )
        CHECK_CUDA( cudaFree(this->d_v) )
    }
}

/* Naive algorithm */
std::shared_ptr<CSR> transpose_cpu(std::shared_ptr<CSR> input) {
    /* Output matrix */
    CSR out{
        "B",
        std::vector<int32_t>(input->nnz, 0),
        std::vector<int32_t>(input->rows + 2, 0),
        std::vector<float>(input->nnz, 0.0),
        input->rows,
        input->cols,
        input->nnz
    };

    /* For each column, count elements and asign them to the rows */
    for (size_t i = 0; i < input->nnz; ++i) {
        ++out.row_index[input->col_index[i] + 2];
    }

    /* Shifted incremental sums */
    for (size_t i = 2; i < out.row_index.size(); ++i) {
        out.row_index[i] += out.row_index[i - 1];
    }

    /* Transpose values */
    for (size_t i = 0; i < input->nnz - 1; ++i) {
        for (size_t j = input->row_index[i]; j < input->row_index[i + 1]; ++j) {
            size_t index = out.row_index[input->col_index[j] + 1]++;
            out.v[index] = input->v[j];
            out.col_index[index] = i;
        }
    }
    out.row_index.pop_back();

    return std::make_shared<CSR>(out);
}

/* CUSPARSE transpose */
std::shared_ptr<CSR> transpose_cusparse(std::shared_ptr<CSR> matrix, cusparseHandle_t handle) {
    /* Create buffers for the transposed matrix */
    int *d_rows_t, *d_cols_t;
    float *d_v_t;
    CHECK_CUDA( cudaMalloc((void**) &d_cols_t, matrix->row_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_rows_t, matrix->col_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_v_t,    matrix->v.size() * sizeof(float)) );

    /* Transpose by converting from CSR to CSC */
    size_t bufferSize;
    void *buffer;

    CHECK_CUSPARSE(
        cusparseCsr2cscEx2_bufferSize(
            handle,
            matrix->rows,
            matrix->cols,
            matrix->v.size(),
            matrix->d_v, matrix->d_rows, matrix->d_cols,
            d_v_t, d_cols_t, d_rows_t,
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG_DEFAULT,
            &bufferSize
        )
    )
    CHECK_CUDA( cudaMalloc((void**) &buffer, bufferSize) )

    CHECK_CUSPARSE(
        cusparseCsr2cscEx2(
            handle,
            matrix->rows,
            matrix->cols,
            matrix->v.size(),
            matrix->d_v, matrix->d_rows, matrix->d_cols,
            d_v_t, d_cols_t, d_rows_t,
            CUDA_R_32F,
            CUSPARSE_ACTION_NUMERIC,
            CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG_DEFAULT,
            buffer
        )
    )

    /* Copy back data */
    int *rows, *cols;
    float *v;
    rows = new int[matrix->col_index.size()];
    cols = new int[matrix->row_index.size()];
    v = new float[matrix->v.size()];

    CHECK_CUDA( 
            cudaMemcpy(
                    rows, 
                    d_rows_t, 
                    matrix->col_index.size() * sizeof(int), 
                    cudaMemcpyDeviceToHost
            )
    );

    CHECK_CUDA( 
            cudaMemcpy(
                    cols, 
                    d_cols_t, 
                    matrix->row_index.size() * sizeof(int), 
                    cudaMemcpyDeviceToHost
            )
    );

    CHECK_CUDA( 
            cudaMemcpy(
                    v, 
                    d_v_t, 
                    matrix->v.size() * sizeof(float), 
                    cudaMemcpyDeviceToHost
            )
    );

    /* Create the transposed matrix */
    CSR out;
    out.name = "C";
    out.rows = matrix->cols;
    out.cols = matrix->rows;
    out.nnz  = matrix->nnz;
    out.d_cols = d_cols_t;
    out.d_rows = d_rows_t;
    out.d_v = d_v_t;

    for (size_t i = 0; i < matrix->col_index.size(); i++)
        out.row_index.push_back(rows[i]);
    delete []rows;

    for (size_t i = 0; i < matrix->row_index.size(); i++)
        out.col_index.push_back(cols[i]);
    delete []cols;

    for (size_t i = 0; i < matrix->v.size(); i++)
        out.v.push_back(v[i]);
    delete []v;

    auto r = std::make_shared<CSR>(out);
    r->in_gpu = true;
    return r;
}

void print_csr(std::shared_ptr<CSR> matrix) {
    std::cout << "COL: ";
    for (auto elem : matrix->col_index) {
        std::cout << elem << " ";
    }
    std::cout << "\n";

    std::cout << "ROW: ";
    for (auto elem : matrix->row_index) {
        std::cout << elem << " ";
    }
    std::cout << "\n";

    std::cout << "VAL: ";
    for (auto elem : matrix->v) {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

void print_csr(
        int *col_index,
        int *row_index,
        float *v,
        size_t col_size,
        size_t row_size,
        size_t v_size
) {
    std::cout << "COL: ";
    for (size_t i = 0; i < col_size; i++) {
        std::cout << col_index[i] << " ";
    }
    std::cout << "\n";

    std::cout << "ROW: ";
    for (size_t i = 0; i < row_size; i++) {
        std::cout << row_index[i] << " ";
    }
    std::cout << "\n";

    std::cout << "VAL: ";
    for (size_t i = 0; i < v_size; i++) {
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}

int main(int argc, char **argv) {
    /* Check parameters passed */
    if (argc < 2) {
        std::cerr << "Error: Missing matrix data path." << std::endl;
        std::cerr << "Usage: ./transpose <path> [OPTIONAL: <device id>]" << std::endl;
        return -1;
    }

    auto matrix = load_matrix_mm(argv[1]);
    matrix->name = "A";
    int device = (argc > 2) ? atoi(argv[2]) : 0;
    CHECK_CUDA( cudaSetDevice(device) )

#ifdef DEBUG
    std::cout << "INPUT MATRIX:\n";
    print_csr(matrix);
#endif

    /* CPU transpose */
    auto t = transpose_cpu(matrix);

#ifdef DEBUG
    std::cout << "\nRESULT (cpu):\n";
    print_csr(t);
#endif

    auto handle = get_handle();
    matrix->to_gpu(handle);
    t = transpose_cusparse(matrix, handle);

#ifdef DEBUG
    std::cout << "\nRESULT (cuSPARSE):\n";
    // print_csr(rows, cols, v, matrix.row_index.size(), matrix.col_index.size(), matrix.v.size());
    print_csr(t);
#endif
    return 0;
}
