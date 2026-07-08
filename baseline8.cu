#include <cublas_v2.h>
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

#define blockj 128
#define blockk 8
#define blockl 128
#define colsize 8
#define rowsize 8

__global__ void vectorized2dblocktiled (float* A, float* B, float* C, int j, int k, int l){
    __shared__ float Atile[blockj*blockk];
    __shared__ float Btile[blockk][blockl];
    A += blockIdx.y*k*blockj;
    B += blockIdx.x*blockl;
    C += blockIdx.y*blockj*l + blockIdx.x*blockl;
    const int t = threadIdx.x;

    const uint resultsperblock = blockj*blockl;
    const uint threadsperblock = resultsperblock/(rowsize*colsize);

    const uint threadcol = t % (blockl/rowsize);
    const uint threadrow = t / (blockl/rowsize);
    const uint rowA = t/(blockk/4);
    const uint colA = t%(blockk/4);
    const uint rowB = t/(blockl/4);
    const uint colB = t%(blockl/4);
    float squareresults[colsize*rowsize] = {0.0};

    float cacheA[colsize] = {0.0};
    float cacheB[rowsize] = {0.0};

    for (int phase = 0; phase<k; phase += blockk){
        int globrowa = blockIdx.y*blockj + rowA;
        int globcola = phase + colA*4;
        
        if(globrowa<j && globcola + 3<k){
            float4 temp = reinterpret_cast<float4*>(&A[rowA*k +colA*4])[0];
            Atile[(colA*4 + 0) * blockj + rowA] = temp.x;
            Atile[(colA*4 + 1) * blockj + rowA] = temp.y;
            Atile[(colA*4 + 2) * blockj + rowA] = temp.z;
            Atile[(colA*4 + 3) * blockj + rowA] = temp.w;
        }
        else if (globrowa<j){
            for (int i=0; i<4; i++)
                Atile[(colA*4 + i) * blockj + rowA] = (globcola + i<k) ? A[rowA*k + colA*4 + i]:0.0f;
        }
        else{
            for (int i=0; i<4; i++) Atile[(colA*4 + i)*blockj + rowA]=0.0f;
        }

        int globrowb = phase+rowB;
        int globcolb = blockIdx.x*blockl + colB*4;
        if (globrowb<k && globcolb + 3 < l)
        reinterpret_cast<float4 *>(&Btile[rowB][colB*4])[0] = 
            reinterpret_cast<float4 *>(&B[rowB*l + colB*4])[0];
        else if (globrowb<k){
            for (int i =0; i<4; i++){
                Btile[rowB][colB*4 + i] = (globcolb + i<l)?B[rowB*l + colB*4 + i]:0.0f;
            }
        } else{
            for (int i =0; i<4; i++) Btile[rowB][colB*4+i]=0.0f;
        }
        //loading in values to smem
        __syncthreads();
        A += blockk;
        B += blockk*l;
        for (uint dotproduct = 0; dotproduct<blockk; dotproduct++){
            for (uint aelem = 0; aelem<colsize; aelem++){
                cacheA[aelem] = Atile[dotproduct*blockj + threadrow*colsize + aelem];
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
    for (uint squarerow = 0; squarerow<colsize; squarerow++){
        for (uint squarecol = 0; squarecol<rowsize; squarecol+=4){
            int globrow = threadrow*colsize + squarerow + blockIdx.y*blockj;
            int globcol = threadcol*rowsize + squarecol + blockIdx.x*blockl;
            if (globrow<j && (globcol+3)<l){
                float4 temp;
                temp.x = squareresults[squarerow*rowsize + squarecol];
                temp.y = squareresults[squarerow*rowsize + squarecol + 1];
                temp.z = squareresults[squarerow*rowsize + squarecol + 2];
                temp.w = squareresults[squarerow*rowsize + squarecol + 3];

                reinterpret_cast<float4 *>(&C[(threadrow*colsize+squarerow)*l + threadcol*rowsize + squarecol])[0] = temp;
            }
            else {
            if (globrow<j)
                for (int tempit=0; tempit<4; tempit++){ 
                    if (globcol<l)
                    C[(threadrow*colsize + squarerow)*l + threadcol*rowsize + squarecol+tempit] = squareresults[squarerow*rowsize + squarecol+tempit];}
            }
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


    dim3 dimGrid2dblocktiling(ceil(l/128.0f),ceil(j/128.0f));

    vectorized2dblocktiled<<<dimGrid2dblocktiling,256>>> (dA, dB, dC, j, k, l);

    cudaMemcpy(C, dC, j*l*sizeof(float), cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}