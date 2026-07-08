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


__global__ void tiledGeneral (float* A, float* B, float* C, int j, int k, int l){
    __shared__ float Atile [tile][tile];
    __shared__ float Btile [tile][tile];
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int Row = by*tile + ty;
    int Col = bx*tile + tx;

    float sum {};
    for (int phase = 0; phase<(k+tile-1)/tile; phase++){
        if(Row<j && (phase*tile + tx)<k)
        Atile[ty][tx] = A[tx + tile*phase + Row*k];
        else Atile[ty][tx] = 0.0f;
        if (Col<l && (phase*tile+ty)<k)
        Btile[ty][tx] = B[Col + l*(phase*tile + ty)];
        else Btile[ty][tx] = 0.0f;
        __syncthreads();

       for (int t=0; t<tile; t++) sum += Atile[ty][t]*Btile[t][tx];
       __syncthreads();
    }
    if (Row<j && Col<l) C[Row*l + Col] = sum;
}

int matmul(float* A, float* B, float* C, float* D, int j, int k, int l){    
    float* A_D, *B_D, *C_D, *D_D;
    size_t asize = (size_t)j*k*sizeof(float);
    size_t bsize = (size_t)l*k*sizeof(float);
    size_t csize = (size_t)j*l*sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&A_D, asize));
    CUDA_CHECK(cudaMalloc((void**)&B_D, bsize));
    CUDA_CHECK(cudaMalloc((void**)&C_D, csize));
    CUDA_CHECK(cudaMalloc((void**)&D_D, csize));

    CUDA_CHECK(cudaMemcpy(A_D, A, asize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_D, B, bsize, cudaMemcpyHostToDevice));
    dim3 dimBlock(32,32);
    dim3 dimGridCoarse(ceil(l/(32.0f*coarse)),ceil(j/32.0f));
    dim3 dimGridTiled(ceil(l/32.0f),ceil(j/32.0f));
 

    tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (A_D, B_D, C_D, j, k, l);    

    tiledGeneral<<<dimGridTiled,dimBlock>>> (A_D, B_D, D_D, j, k, l);

    CUDA_CHECK(cudaMemcpy(C, C_D, csize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(D, D_D, csize, cudaMemcpyDeviceToHost));
    
    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(C_D);
    cudaFree(D_D);

    return 0;
}

int main(){
    size_t j = 4096; //a row and c row
    size_t k = 4096; //a col and b row
    size_t l = 4096; //b col and c col
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
    for (int i=0; i<32; i++){
        std::cout<<C[i]<<' '<< D[i]<<'\n';
        //if (abs(C[i]-D[i])>0.001) std::cout<<i<<'\n';
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    return 0;
}