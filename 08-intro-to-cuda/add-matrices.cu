// $Smake: nvcc -O2 -o %F %f
//
// add-vectors.cu - addition of two arrays on GPU device
//
// This program follows a very standard pattern:
//  1) allocate memory on host
//  2) allocate memory on device
//  3) initialize memory on host
//  4) copy memory from host to device
//  5) execute kernel(s) on device
//  6) copy result(s) from device to host
//
// Note: it may be possible to initialize memory directly on the device,
// in which case steps 3 and 4 are not necessary, and step 1 is only
// necessary to allocate memory to hold results.

#include <stdio.h>
#include <cuda.h>
#define IDX(i,j,stride) ((i)+(j)*(stride))

//-----------------------------------------------------------------------------
// Kernel that executes on CUDA device

__global__ void add_matrices(
    float *c,      // out - pointer to result matrix c
    float *a,      // in  - pointer to summand matrix a
    float *b,      // in  - pointer to summand matrix b
    int width,     // in  - matrix row length
    int height     // in  - matrix column length
    )
{
    // Assume single block grid and 1-D block
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = column + row * height;  // column major

    // Only do calculation if we have real data to work with
    if (idx < (width * height)) c[idx] = a[idx] + b[idx];
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Main program executes on host device

int main(int argc, char* argv[])
{
    // determine vector length
    int width = 10;   // set default width
    int height = 10;  // set default height
    if (argc > 2)
    {
        height = atoi(argv[1]);   // override default length
        width = atoi(argv[2]);  // override default length
        if (width <= 0)
        {
            fprintf(stderr, "Matrix width must be positive\n");
            return EXIT_FAILURE;
        }
        if (height <= 0)
        {
            fprintf(stderr, "Matrix height must be positive\n");
            return EXIT_FAILURE;
        }
    }

    // determine vector size in bytes
    const size_t matrix_width = width * sizeof(float);
    const size_t matrix_height = height * sizeof(float);

    // declare pointers to vectors in host memory and allocate memory
    float *a, *b, *c;
    a = (float*) malloc(matrix_width * matrix_height);
    b = (float*) malloc(matrix_width * matrix_height);
    c = (float*) malloc(matrix_width * matrix_height);

    // declare pointers to vectors in device memory and allocate memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, matrix_width * matrix_height);
    cudaMalloc((void**) &d_b, matrix_width * matrix_height);
    cudaMalloc((void**) &d_c, matrix_width * matrix_height);

    // initialize vectors and copy them to device
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            a[IDX(i,j,height)] =   1.0 * i;
            b[IDX(i,j,height)] = 100.0 * i;
        }
    }
    cudaMemcpy(d_a, a, matrix_width * matrix_height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, matrix_width * matrix_height, cudaMemcpyHostToDevice);

    // do calculation on device
    dim3 block_size(16, 16);
    dim3 num_blocks((width - 1 + block_size.x) / block_size.x, 
                  (height - 1 + block_size.y) / block_size.y);
    add_matrices<<< num_blocks, block_size >>>(d_c, d_a, d_b, height, width);

    // retrieve result from device and store on host
    cudaMemcpy(c, d_c, matrix_width * matrix_height, cudaMemcpyDeviceToHost);

    // print results for vectors up to length 100
    if (width <= 10 && height <= 10)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                printf("%8.2f + %8.2f = %8.2f\n", a[IDX(i,j,height)], b[IDX(i,j,height)], c[IDX(i,j,height)]);
            }
        }
    }

    // cleanup and quit
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
  
    return 0;
}