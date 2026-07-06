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

    CUDA_CHECK(cudaMalloc((void**)&A_D, asize));
    CUDA_CHECK(cudaMalloc((void**)&B_D, bsize));
    CUDA_CHECK(cudaMalloc((void**)&C_D, csize));
    CUDA_CHECK(cudaMalloc((void**)&D_D, csize));

    CUDA_CHECK(cudaMemcpy(A_D, A, asize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_D, B, bsize, cudaMemcpyHostToDevice));
    dim3 dimBlock(32,32);
    dim3 dimGridCoarse(ceil(l/(32.0f*coarse)),ceil(j/32.0f));
    dim3 dimGrid1d(ceil(l/64.0f),ceil(j/64.0f));
 
    blocktiling1d<<<dimGrid1d,512>>>(A_D, B_D, C_D, j, k, l);



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