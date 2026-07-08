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


#define bj 64
#define bk 8
#define bl 64
#define colsize 8
__global__ void blocktiling1d (const float* A, const float* B, float* C, int j, int k, int l){
    __shared__ float Atile[bj][bk];
    __shared__ float Btile[bk][bl];
    const int t = threadIdx.x;
    A += blockIdx.y*k*bj;
    B += blockIdx.x*bl;
    C += blockIdx.y*bj*l + blockIdx.x*bl;
    float colresults[colsize] = {0.0};
    for (int phase = 0; phase < k; phase+=bk){
        if((phase+t%bk)<k && blockIdx.y*bj+t/bk<j)
        Atile[t/bk][t%bk] = A[phase + t/bk*k + t%bk];
        else Atile[t/bk][t%bk] = 0.0f;
        if ((phase+t/bl)<k && t%bl+blockIdx.x*bl<l)
        Btile[t/bl][t%bl] = B[phase*l + t/bl*l + t%bl];
        else Btile[t/bl][t%bl] = 0.0f;
        __syncthreads();

        for (int dot = 0; dot<bk; dot++){
            float btemp = Btile[dot][t%bl];
            for (int cColId = 0; cColId<colsize; cColId++){
                colresults[cColId] += btemp*Atile[t/bj*colsize+cColId][dot];
            }
        }
        __syncthreads();
    }

    for (int cColId = 0; cColId<colsize; cColId++){
        if (blockIdx.x*bl +t%bl<l && blockIdx.y*bj+t/bl*colsize+cColId<j)
        C[(cColId+t/bl*colsize)*l+t%bl] = colresults[cColId];
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


    dim3 dimGrid1d(ceil(l/64.0f),ceil(j/64.0f));
    blocktiling1d<<<dimGrid1d,512>>> (dA, dB, dC, j, k, l);
    cudaMemcpy(C, dC, j*l*sizeof(float), cudaMemcpyDeviceToHost);

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}