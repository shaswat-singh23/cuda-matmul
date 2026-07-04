#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#define tile 32
#define coarse 4
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

__global__ void baseline4(float* A, float* B, float* C, int width) {
int col = blockIdx.x*blockDim.x+threadIdx.x;
int row = blockIdx.y*blockDim.y+threadIdx.y;
if (col<width && row<width){
    float sum = 0;
    for (int k=0; k<width;k++){
        sum+=A[row*width+k]*B[k*width+col];
    }
    C[row*width+col]=sum;
}    
}

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

int main(){
    FILE* csv = fopen("results.csv", "w");
    fprintf(csv, "N,kernel,gflops,time_ms\n");
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192};
    long maxsize = sizes[sizes.size()-1];
    long j, k, l;
    float *A, *B, *C, *C_ref;
    A = new float[maxsize*maxsize];
    B = new float[maxsize*maxsize];
    C = new float[maxsize*maxsize];
    C_ref = new float[maxsize*maxsize];

    for (int i = 0; i<maxsize*maxsize; i++) A[i]=(float)rand()/RAND_MAX;
    for (int i = 0; i<maxsize*maxsize; i++) B[i]=(float)rand()/RAND_MAX;
    float *dA, *dB, *dC, *dC_ref;
    float time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    CUDA_CHECK(cudaMalloc((void**)&dA, maxsize*maxsize*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, maxsize*maxsize*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, maxsize*maxsize*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC_ref, maxsize*maxsize*sizeof(float)));
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;



    for (const int& size:sizes){
        j=k=l=size;
        std::cout<<"Multiplication of two square matrices of "<<size<<'x'<<size<<'\n';

        cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        l, j, k,          // dimensions (swapped)
        &alpha,
        dB, l,           // B first, leading dim l
        dA, k,           // A second, leading dim k
        &beta,
        dC_ref, l);
        cudaMemcpy(C_ref, dC_ref, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost);

        dim3 dimBlock(32,32);
        dim3 dimGridCoarse(ceil(l/(32.0f*coarse)),ceil(j/32.0f));
        dim3 dimGridTiled(ceil(l/32.0f),ceil(j/32.0f));

        baseline4<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        } 

        tiledGeneral<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        }

        tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        }
        
        
        
        int repeat =20;
        long flops = 2*j*k*l;
        for (int d=0; d<3; d++){
            baseline4<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            baseline4<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for naive kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(repeat*1000), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "naive", (repeat*flops*1e-6)/time, time/20);
        fflush(stdout);

        for (int d=0; d<3; d++){
            tiledGeneral<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            tiledGeneral<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for tiled kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(repeat*1000), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "tiled", (repeat*flops*1e-6)/time, time/20);
        fflush(stdout);

        for (int d=0; d<3; d++){
            tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for coarsened kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(1000*repeat), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "tiled with coarsening", (repeat*flops*1e-6)/time, time/20);
        fflush(stdout);

        for (int d=0; d<3; d++){
            cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            l, j, k,          // dimensions (swapped)
            &alpha,
            dB, l,           // B first, leading dim l
            dA, k,           // A second, leading dim k
            &beta,
            dC_ref, l);        
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            l, j, k,          // dimensions (swapped)
            &alpha,
            dB, l,           // B first, leading dim l
            dA, k,           // A second, leading dim k
            &beta,
            dC_ref, l);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for cuBLAS kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(1000*repeat), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "cuBLAS", (repeat*flops*1e-6)/time, time/20);
        fflush(stdout);
        CUDA_CHECK(cudaMemcpy(dC, dC_ref, sizeof(float)*j*l, cudaMemcpyDeviceToDevice));
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);

}