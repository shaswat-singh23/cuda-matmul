#include <cublas_v2.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#define tile 32
#define coarse 4
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

__global__ void tiledGeneralWithCoarsening (float* A, float* B, float* C, int j, int k, int l){
    __shared__ float Atile [tile][tile];
    __shared__ float Btile [tile][tile];
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int Row = by*tile + ty;

    float sum [coarse] = {0.0f};
    for (int phase = 0; phase<(k+tile-1)/tile; phase++){
        if(Row<j && (phase*tile + tx)<k)
        Atile[ty][tx] = A[tx + tile*phase + Row*k];
        else Atile[ty][tx] = 0.0f;
        
        for (int c = 0; c<coarse; c++){
            int Col = bx*tile*coarse + c*tile + tx;
            if (Col<l && (phase*tile+ty)<k)
            Btile[ty][tx] = B[Col + l*(phase*tile + ty)];
            else Btile[ty][tx] = 0.0f;
            __syncthreads();

            for (int t=0; t<tile; t++) sum[c] += Atile[ty][t]*Btile[t][tx];
            __syncthreads();
        }
    }
    for (int c = 0; c<coarse; c++){
        int Col = bx*tile*coarse + c*tile + tx;
        if (Row<j && Col<l) C[Row*l + Col] = sum[c];
    }
}



int matmul(float* A, float* B, float* C, float* D, int j, int k, int l){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    size_t minSharedMem = 2 * tile * tile * sizeof(float);
    if (minSharedMem > prop.sharedMemPerBlock){
        fprintf(stderr, "Tile size %d requires %zu bytes shared mem, device only supports %zu\n", tile, minSharedMem, prop.sharedMemPerBlock);
        return -1;
    }

    if (tile*tile > prop.maxThreadsPerBlock){
        fprintf(stderr, "Block size %dx%d exceeds device max %d threads per block\n", tile, tile, prop.maxThreadsPerBlock);
        return -1;
    }

    if (prop.major < 7){
        fprintf(stderr, "Requires compute capability 7.0+ (Volta or newer). Device is %d.%d\n", prop.major, prop.minor);
        return -1;
    }
    
    float* A_D, *B_D, *C_D, *D_D;
    size_t asize = (size_t)j*k*sizeof(float);
    size_t bsize = (size_t)l*k*sizeof(float);
    size_t csize = (size_t)j*l*sizeof(float);
    size_t dsize = (size_t)j*l*sizeof(float);


    CUDA_CHECK(cudaMalloc((void**)&A_D, asize));
    CUDA_CHECK(cudaMalloc((void**)&B_D, bsize));
    CUDA_CHECK(cudaMalloc((void**)&C_D, csize));
    CUDA_CHECK(cudaMalloc((void**)&D_D, csize));
    CUDA_CHECK(cudaMemcpy(A_D, A, asize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_D, B, bsize, cudaMemcpyHostToDevice));
    
    dim3 dimBlock(32,32);
    dim3 dimGridCoarse(ceil(l/(32.0f*coarse)),ceil(j/32.0f));
    dim3 dimGridTiled(ceil(l/32.0f),ceil(j/32.0f));

    cublasHandle_t handle;
cublasCreate(&handle);
const float alpha = 1.0f;
const float beta = 0.0f;

// You want C = A × B in row-major (A is j×k, B is k×l, C is j×l)
// cuBLAS computes column-major, so swap operands:
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            l, j, k,          // dimensions (swapped)
            &alpha,
            B_D, l,           // B first, leading dim l
            A_D, k,           // A second, leading dim k
            &beta,
            D_D, l);        // output, leading dim lcublasDestroy(handle);
    tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (A_D, B_D, C_D, j, k, l);    

    CUDA_CHECK(cudaMemcpy(C, C_D, csize, cudaMemcpyDeviceToHost));
cudaMemcpy(D, D_D, csize, cudaMemcpyDeviceToHost);
int errors = 0;
for (size_t i = 0; i < (size_t)j * l; i++) {
    float diff = fabsf(C[i] - D[i]);
    float rel = diff / (fabsf(D[i]) + 1e-8f);  // +epsilon avoids div-by-zero
    if (rel > 1e-3f && diff > 1e-4f) {
        if (errors < 10) printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], D[i]);
        errors++;
    }
}
printf("Total mismatches: %d / %zu\n", errors, (size_t)j * l);
    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(C_D);
    cudaFree(D_D);

    return 0;
}

int main(){
    int j = 128; //a row and c row
    int k = 256; //a col and b row
    int l = 192; //b col and c col
    float* A = new float[j*k]();
    float* B = new float[k*l]();
    float* C = new float[j*l]();
    float* D = new float[j*l]();

    for (int i = 0; i<j*k; i++) A[i]=(float)rand()/RAND_MAX;
    for (int i = 0; i<k*l; i++) B[i]=(float)rand()/RAND_MAX;
    if (matmul(A, B, C, D, j, k, l) !=0){
        fprintf(stderr, "matmul failed\n");
        return 1;
    }

    
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    return 0;
    return 0;
}