#include "csr.cuh"

cusparseHandle_t get_handle() {
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    return handle;
}

std::shared_ptr<CSR> load_matrix_mm(std::string path) {
    /* Load entries */
    auto [x, y, v, rows, cols, nnz] = load_entries(path);
    auto coords = std::make_tuple(x, y);

    /* Create the CSR matrix */
    CSR out;
    out.rows = rows;
    out.cols = cols;
    out.nnz = nnz;

    /* sort COO entries */
    std::vector<size_t> indices;
    for (size_t i = 0; i < x.size(); i++) indices.push_back(i);
    std::stable_sort(indices.begin(), indices.end(),
        [&coords](size_t a, size_t b) {
            if (std::get<0>(coords)[a] < std::get<0>(coords)[b]) return true;
            if (std::get<0>(coords)[a] == std::get<0>(coords)[b])
	    	return std::get<1>(coords)[a] < std::get<1>(coords)[b];
            return false;
        }
    );

    /* generate CSR matrix from sorted COO */
    out.row_index = std::vector<int32_t>(rows+1, 0);
    for (size_t i = 0; i < nnz; i++) {
        size_t entry = indices[i];
        out.row_index[x[entry] + 1] += 1;
        out.col_index.push_back(y[entry]);
        out.v.push_back(v[entry]);
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
        //std::cout << this->name << ": Freeing memory..." << std::endl;
        CHECK_CUDA( cudaFree(this->d_cols) )
        CHECK_CUDA( cudaFree(this->d_rows) )
        CHECK_CUDA( cudaFree(this->d_v) )
    }
}

std::tuple<std::shared_ptr<CSR>, float> transpose_cusparse(std::shared_ptr<CSR> matrix, cusparseHandle_t handle) {
    /* Create buffers for the transposed matrix */
    int *d_rows_t, *d_cols_t;
    float *d_v_t;
    CHECK_CUDA( cudaMalloc((void**) &d_cols_t, matrix->row_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_rows_t, matrix->col_index.size() * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &d_v_t,    matrix->v.size() * sizeof(float)) );

    /* Transpose by converting from CSR to CSC */
    size_t bufferSize;
    void *buffer;

    /* Record time using CUDA */
    cudaEvent_t start, stop;
    float time;
    CHECK_CUDA( cudaEventCreate(&start) )
    CHECK_CUDA( cudaEventCreate(&stop) )
    CHECK_CUDA( cudaEventRecord(start, 0) )

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
    CHECK_CUDA( cudaFree(buffer) )

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

    CHECK_CUDA( cudaEventRecord(stop, 0) )
    CHECK_CUDA( cudaEventSynchronize(stop) )
    CHECK_CUDA( cudaEventElapsedTime(&time, start, stop) )
    CHECK_CUDA( cudaEventDestroy(start) )
    CHECK_CUDA( cudaEventDestroy(stop) )

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
    
    return std::make_tuple(r, time);
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
