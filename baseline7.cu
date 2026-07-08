#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)


#define blockj 64
#define blockk 8
#define blockl 64
#define colsize 8
#define rowsize 8
__global__ void blocktiling2d (const float* A, const float* B, float* C, int j, int k, int l){
    __shared__ float Atile[blockj][blockk+1];
    __shared__ float Btile[blockk][blockl];
    A += blockIdx.y*k*blockj;
    B += blockIdx.x*blockl;
    C += blockIdx.y*blockj*l + blockIdx.x*blockl;
    const int t = threadIdx.x;

    const uint resultsperblock = blockj*blockl;
    const uint threadsperblock = resultsperblock/(rowsize*colsize);

    const uint threadcol = t % (blockl/rowsize);
    const uint threadrow = t / (blockl/rowsize);
    const uint rowA = t/blockk;
    const uint colA = t%blockk;
    const uint rowB = t/blockl;
    const uint colB = t%blockl;
    float squareresults[colsize*rowsize] = {0.0};

    float cacheA[colsize] = {0.0};
    float cacheB[rowsize] = {0.0};

    for (int phase = 0; phase<k; phase += blockk){
        //loading in values to smem
        for (uint loadedrows = 0; loadedrows<blockj; loadedrows += threadsperblock/blockk){
            if ((loadedrows+rowA+blockj*blockIdx.y)<j && phase+colA<k)
                Atile[loadedrows+rowA][colA] = A[phase + (loadedrows+rowA)*k+colA];
            else Atile[loadedrows+rowA][colA] = 0.0;
        }
        for (uint loadedrows = 0; loadedrows<blockk; loadedrows += threadsperblock/blockl){
            if ((phase+loadedrows+rowB)<k && blockIdx.x*blockl+colB<l)
                Btile[loadedrows + rowB][colB] = B[(loadedrows+rowB+phase)*l + colB];
            else Btile[loadedrows + rowB][colB] = 0.0;
        }
        __syncthreads();

        for (uint dotproduct = 0; dotproduct<blockk; dotproduct++){
            for (uint aelem = 0; aelem<colsize; aelem++){
                cacheA[aelem] = Atile[aelem + threadrow*colsize][dotproduct];
            }
            for (uint belem = 0; belem<rowsize; belem++){
                cacheB[belem] = Btile[dotproduct][threadcol*rowsize+belem];
            }
            for (uint squarerow = 0; squarerow<colsize; squarerow++){
                for (uint squarecol = 0; squarecol<rowsize; squarecol++){
                    squareresults[squarerow*rowsize + squarecol] += cacheA[squarerow]*cacheB[squarecol];
                }
            }
        }
        __syncthreads();
    }
    for (uint squarecol = 0; squarecol<colsize; squarecol++){
        for (uint squarerow = 0; squarerow<rowsize; squarerow++){
            if ((blockIdx.y*blockj + threadrow*colsize + squarerow) < j && (blockIdx.x*blockl + threadcol*rowsize + squarecol) < l)
            C[(threadrow*colsize + squarerow)*l + threadcol*rowsize + squarecol] = squareresults[squarerow*rowsize + squarecol];
        }
    }
    
}

int main(){
    long j, k, l;
    j=k=l=2048;
    float *A, *B, *C;
    A = new float[j*k];
    B = new float[k*l];
    C = new float[j*l];

    for (int i = 0; i<j*k; i++) A[i]=(float)rand()/RAND_MAX;
    for (int i = 0; i<k*l; i++) B[i]=(float)rand()/RAND_MAX;
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc((void**)&dA, j*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, k*l*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, j*l*sizeof(float)));
    cudaMemcpy(dA, A, j*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, k*l*sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimGrid2dblocktiling(ceil(l/64.0f),ceil(j/64.0f));

    blocktiling2d<<<dimGrid2dblocktiling,64>>> (dA, dB, dC, j, k, l);

    cudaMemcpy(C, dC, j*l*sizeof(float), cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}