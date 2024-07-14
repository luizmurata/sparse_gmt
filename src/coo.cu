#include "coo.cuh"

/* CUDA Kernels Used */

/* Blelloch Scan */
__global__
void partial_scan(unsigned* d_list,
                  unsigned *d_block_sums,
                  size_t n) {
    /* Temporary buffer containing the scan results for the thread */
    extern __shared__ unsigned int temp[];

    int thid = threadIdx.x;
    int index = blockDim.x * blockIdx.x + thid;

    /* Load data into shared memory, pad the last block */
    if (index >= n) temp[thid] = 0;
    else temp[thid] = d_list[index];
    __syncthreads();

    /* Upsweep / Reduce step */
    unsigned int i;
    for (i = 2; i <= blockDim.x; i <<= 1) {
        if ((thid + 1) % i == 0) {
            unsigned ai = thid;
            unsigned bi = ai - (i >> 1);
            temp[ai] += temp[bi];
        }
        __syncthreads();
    }

    /* Keep track of the block sum for the final aggregation of the results */
    if (thid == (blockDim.x - 1)) {
        d_block_sums[blockIdx.x] = temp[thid];
        temp[thid] = 0;
    }
    __syncthreads();

    /* Downsweep */
    for (i = i >> 1; i >= 2; i >>= 1)
    {
        if ((thid + 1) % i == 0) {
            unsigned ai = thid;
            unsigned bi = ai - (i >> 1);
            unsigned tmp = temp[bi];
            temp[bi] = temp[ai];
            temp[ai] += tmp;
        }
        __syncthreads();
    }

    /* Copy results back*/
    if (index < n) d_list[index] = temp[thid];
}

__global__
void aggregate_scan(unsigned *d_list,
                    unsigned *d_block_sums,
                    size_t n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    d_list[index] += d_block_sums[blockIdx.x]; /* Add the blocksum to adjust to the final result */
}

/* Wrapper function for the scan */
void scan(unsigned *d_list, size_t n, unsigned *d_total, unsigned *d_block_sums) {
        /* Number of blocks necessary */
        int gridSize = ceil(float(n) / float(BLOCK_SIZE));

        /* Make sure that the block sums buffer is zeroed before the scan */
        CHECK_CUDA(cudaMemset(d_block_sums, 0, gridSize * sizeof(unsigned int)));

        /* 1. Compute the partial scan on the blocks and keep the block sums */
        partial_scan<<<gridSize, BLOCK_SIZE, sizeof(unsigned) * BLOCK_SIZE>>>(d_list, d_block_sums, n);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        /* 2. Compute the partial scan on the block sums and keep the total */
        partial_scan<<<1, BLOCK_SIZE, sizeof(unsigned) * BLOCK_SIZE>>>(d_block_sums, d_total, gridSize);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        /* 3. Aggregate the block scans with the block sums */
        aggregate_scan<<<gridSize, BLOCK_SIZE>>>(d_list, d_block_sums, n);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());
}

/* Radix Sort for COO */
__global__ 
void test_bits(unsigned *d_list,
               unsigned *d_tests,
               unsigned mask, size_t n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    /* Test if the bit is 0 */
    d_tests[index] = ((d_list[index] & mask) == 0);
}

__global__
void flip_bits(unsigned *d_list, size_t n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    /* NOT operation */
    d_list[index] = ((d_list[index] + 1) % 2);
}

/* Shuffle data into the new found ordering */
__global__
void shuffle(unsigned *d_x, unsigned *d_outX,
             unsigned *d_y, unsigned *d_outY,
             float    *d_v, float *d_outV,
             unsigned *d_test,
             unsigned *d_zeros_scan,
             unsigned *d_ones_scan,
             unsigned *d_zeros_total,
             size_t n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    unsigned int new_index;
    if (d_test[index] == 0) new_index = d_zeros_scan[index]; /* It's a 0 so move it by the scan value found */
    else new_index = d_ones_scan[index] + *d_zeros_total; /* It's a 1 so also add the offset */

    d_outX[new_index] = d_x[index];
    d_outY[new_index] = d_y[index];
    d_outV[new_index] = d_v[index];
}

/* Wrapper function for the COO sort */
void coo_radix_sort(
               unsigned *const d_inX,
               unsigned *const d_inY,
               float *const d_inV,
               unsigned *const d_outX,
               unsigned *const d_outY,
               float *const d_outV,
               size_t n_rows,
               size_t n_cols,
               size_t nnz) {

    /* Copy input data to the temporary buffers used */
    unsigned *d_tmpX;
    unsigned *d_tmpY;
    float *d_tmpV;
    CHECK_CUDA(cudaMalloc(&d_tmpX, sizeof(unsigned) * nnz))
    CHECK_CUDA(cudaMalloc(&d_tmpY, sizeof(unsigned) * nnz))
    CHECK_CUDA(cudaMalloc(&d_tmpV, sizeof(float) * nnz))
    CHECK_CUDA(cudaMemcpy(d_tmpX, d_inX, sizeof(unsigned) * nnz, cudaMemcpyDeviceToDevice))
    CHECK_CUDA(cudaMemcpy(d_tmpY, d_inY, sizeof(unsigned) * nnz, cudaMemcpyDeviceToDevice))
    CHECK_CUDA(cudaMemcpy(d_tmpV, d_inV, sizeof(unsigned) * nnz, cudaMemcpyDeviceToDevice))

    /* Number of blocks necessary */
    int gridSize = ceil(float(nnz) / float(BLOCK_SIZE));

    /* Temporary buffers for scanning and sorting */
    unsigned *d_block_sums;  /* Sums of the scans for each block */
    unsigned *d_test;        /* Contains the results when testing the bits */
    unsigned *d_zeros_scan;  /* Scan of the elements with bit 0*/
    unsigned *d_ones_scan;   /* Scan of the elements with bit 1 */
    unsigned *d_zeros_total; /* Total amount of elements with bit 0 */
    unsigned *d_ones_total;  /* Total amount of elements with bit 1 */

    CHECK_CUDA(cudaMalloc((void **)&d_block_sums, gridSize * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_test, nnz * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_zeros_scan, nnz * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_ones_scan, nnz * sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_zeros_total, sizeof(unsigned)));
    CHECK_CUDA(cudaMalloc((void **)&d_ones_total, sizeof(unsigned)));

    /* Calculate the number of bits required for the radix sort */
    auto max_value = (n_cols > n_rows) ? n_cols : n_rows;
    unsigned bits = ((unsigned)ceil(log2(max_value)) + 1)*2; 
    unsigned half_bits = bits / 2;
    unsigned mask; /* bitmask applied when testing */
    
    /* Alternate between the temporary buffer and output buffer to avoid overwriting*/
    unsigned *d_inputs, *d_outputs;

    for (unsigned bit = 0; bit < bits; bit++) {
        /* Check if we are sorting by columns or rows */
        if (bit < half_bits) {
            mask = 1 << bit;
            d_inputs = d_tmpY;
            d_outputs = d_outY;
        } else {
            mask = 1 << (bit - half_bits);
            d_inputs = d_tmpX;
            d_outputs = d_outX;
        }

        /* Test if the bits are zero */
        if (bit % 2 == 0) test_bits<<<gridSize, BLOCK_SIZE>>>(d_inputs, d_test, mask, nnz);
        else test_bits<<<gridSize, BLOCK_SIZE>>>(d_outputs, d_test, mask, nnz);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        /* Scan for the zeros */
        CHECK_CUDA(cudaMemcpy(d_zeros_scan, d_test, nnz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
        scan(d_zeros_scan, nnz, d_zeros_total, d_block_sums);

        /* Flip for scanning the ones */
        flip_bits<<<gridSize, BLOCK_SIZE>>>(d_test, nnz);
        cudaDeviceSynchronize();
        CHECK_CUDA(cudaGetLastError());

        /* Scan for the ones*/
        CHECK_CUDA(cudaMemset(d_block_sums, 0, gridSize * sizeof(unsigned int)));
        CHECK_CUDA(cudaMemcpy(d_ones_scan, d_test, nnz * sizeof(unsigned), cudaMemcpyDeviceToDevice));
        scan(d_ones_scan, nnz, d_ones_total, d_block_sums);

        /* Re-shuffle the values according to the ordering found in this step */
        if (bit % 2 == 0) {
            shuffle<<<gridSize, BLOCK_SIZE>>>(
                d_tmpX, d_outX,
                d_tmpY, d_outY,
                d_tmpV, d_outV,
                d_test,
                d_zeros_scan,
                d_ones_scan,
                d_zeros_total,
                nnz
            );
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
        } else {
            shuffle<<<gridSize, BLOCK_SIZE>>>(
                d_outX, d_tmpX,
                d_outY, d_tmpY,
                d_outV, d_tmpV,
                d_test,
                d_zeros_scan,
                d_ones_scan,
                d_zeros_total,
                nnz
            );
            cudaDeviceSynchronize();
            CHECK_CUDA(cudaGetLastError());
        }
    }

    CHECK_CUDA(cudaFree(d_tmpX));
    CHECK_CUDA(cudaFree(d_tmpY));
    CHECK_CUDA(cudaFree(d_tmpV));
    CHECK_CUDA(cudaFree(d_test));
    CHECK_CUDA(cudaFree(d_zeros_scan));
    CHECK_CUDA(cudaFree(d_ones_scan));
    CHECK_CUDA(cudaFree(d_zeros_total));
    CHECK_CUDA(cudaFree(d_ones_total));
    CHECK_CUDA(cudaFree(d_block_sums));
}

std::shared_ptr<COO> load_coo_mm(std::string path) {
    /* Load file entries */
    auto [x, y, v, n_rows, n_cols, nnz] = load_entries(path);

    /* Allocate device buffers*/
    unsigned *outX = new unsigned[x.size()];
    unsigned *outY = new unsigned[y.size()];
    float *outV = new float[v.size()];
    unsigned *d_inX, *d_inY, *d_outX, *d_outY;
    float *d_inV, *d_outV;
    CHECK_CUDA(cudaMalloc(&d_inX, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_inY, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_outX, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_outY, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_inV, sizeof(float) * nnz))
    CHECK_CUDA(cudaMalloc(&d_outV, sizeof(float) * nnz))
    CHECK_CUDA(cudaMemcpy(d_inX, x.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_inY, y.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(d_inV, v.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice))
    
    /* Sort entries */
    coo_radix_sort(d_inX, d_inY, d_inV, d_outX, d_outY, d_outV, n_rows, n_cols, nnz);

    /* Copy back data*/
    CHECK_CUDA(cudaMemcpy(outX, d_outX, sizeof(int) * nnz, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(outY, d_outY, sizeof(int) * nnz, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(outV, d_outV, sizeof(float) * nnz, cudaMemcpyDeviceToHost))

    /* Free unused buffers */
    CHECK_CUDA(cudaFree(d_inX))
    CHECK_CUDA(cudaFree(d_inY))
    CHECK_CUDA(cudaFree(d_inV))


    /* Initialize matrix */
    COO out;
    out.nnz = nnz;
    out.n_rows = n_rows;
    out.n_cols = n_cols;
    out.x = outX;
    out.y = outY;
    out.v = outV;
    out.d_x = d_outX;
    out.d_y = d_outY;
    out.d_v = d_outV;

    auto r = std::make_shared<COO>(out);
    r->in_gpu = true;
    return r;
}

std::tuple<std::shared_ptr<COO>, float> transpose_coo(std::shared_ptr<COO> matrix) {
    /* Allocate device buffers*/
    size_t nnz = matrix->nnz;
    unsigned *outX = new unsigned[nnz];
    unsigned *outY = new unsigned[nnz];
    float *outV = new float[nnz];
    unsigned *d_outX, *d_outY;
    float *d_outV;
    CHECK_CUDA(cudaMalloc(&d_outX, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_outY, sizeof(int) * nnz))
    CHECK_CUDA(cudaMalloc(&d_outV, sizeof(float) * nnz))
    
    /* Record time using CUDA */
    cudaEvent_t start, stop;
    float time;
    CHECK_CUDA( cudaEventCreate(&start) )
    CHECK_CUDA( cudaEventCreate(&stop) )
    CHECK_CUDA( cudaEventRecord(start, 0) )
    
    /* Sort entries but with X and Y swapped */
    coo_radix_sort(matrix->d_y, matrix->d_x, matrix->d_v, d_outX, d_outY, d_outV, matrix->n_cols, matrix->n_rows, nnz);

    /* Copy back data*/
    CHECK_CUDA(cudaMemcpy(outX, d_outX, sizeof(int) * nnz, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(outY, d_outY, sizeof(int) * nnz, cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(outV, d_outV, sizeof(float) * nnz, cudaMemcpyDeviceToHost))

    CHECK_CUDA( cudaEventRecord(stop, 0) )
    CHECK_CUDA( cudaEventSynchronize(stop) )
    CHECK_CUDA( cudaEventElapsedTime(&time, start, stop) )
    CHECK_CUDA( cudaEventDestroy(start) )
    CHECK_CUDA( cudaEventDestroy(stop) )

    /* Initialize matrix */
    COO out;
    out.nnz = nnz;
    out.n_rows = matrix->n_rows;
    out.n_cols = matrix->n_cols;
    out.x = outX;
    out.y = outY;
    out.v = outV;
    out.d_x = d_outX;
    out.d_y = d_outY;
    out.d_v = d_outV;

    auto r = std::make_shared<COO>(out);
    r->in_gpu = true;
    return std::make_tuple(r, time);
}

COO::~COO() {
    if (this->in_gpu) {
        //std::cout << "Freeing memory..." << std::endl;
        CHECK_CUDA(cudaFree(this->d_x))
        CHECK_CUDA(cudaFree(this->d_y))
        CHECK_CUDA(cudaFree(this->d_v))
    }
}

void print_coo(std::shared_ptr<COO> matrix) {
    std::cout << "-----MATRIX-----" << std::endl;
    std::cout << "ROWS" << std::endl;
    for (size_t i = 0; i < matrix->nnz; i++) std::cout << matrix->x[i] << " ";
    std::cout << std::endl << "COLS" << std::endl;
    for (size_t i = 0; i < matrix->nnz; i++) std::cout << matrix->y[i] << " ";
    std::cout << std::endl << "VALS" << std::endl;
    for (size_t i = 0; i < matrix->nnz; i++) std::cout << matrix->v[i] << " ";
    std::cout << std::endl << "----------------" << std::endl; 
}