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

#define bj 64
#define bk 8
#define bl 64
#define colsize 8
__global__ void baseline6 (const float* A, const float* B, float* C, int j, int k, int l){
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

__global__ void baseline5coarse (float* A, float* B, float* C, int j, int k, int l){
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


__global__ void baseline5 (float* A, float* B, float* C, int j, int k, int l){
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

#define blockj 128
#define blockk 8
#define blockl 128
#define colsize 8
#define rowsize 8
__global__ void __launch_bounds__ ((blockj*blockl)/(colsize*rowsize),1) baseline7 (const float* A, const float* B, float* C, int j, int k, int l){
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
    for (uint squarerow = 0; squarerow<colsize; squarerow++){
        for (uint squarecol = 0; squarecol<rowsize; squarecol++){
            if ((blockIdx.y*blockj + threadrow*colsize + squarerow) < j && (blockIdx.x*blockl + threadcol*rowsize + squarecol) < l)
            C[(threadrow*colsize + squarerow)*l + threadcol*rowsize + squarecol] = squareresults[squarerow*rowsize + squarecol];
        }
    }
    
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
        dim3 dimGrid2dblocktiling(ceil(l/64.0f),ceil(j/64.0f));

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

        baseline5<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        }

        baseline5coarse<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        }

        baseline6<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        CUDA_CHECK(cudaMemcpy(C, dC, maxsize*maxsize*sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < (size_t)j * l; i++) {
            float diff = fabsf(C[i] - C_ref[i]);
            float rel = diff / (fabsf(C_ref[i]) + 1e-8f);  // +epsilon avoids div-by-zero
            if (rel > 1e-3f && diff > 1e-4f) {
                printf("Mismatch at %zu: kernel=%f ref=%f\n", i, C[i], C_ref[i]);
                exit(EXIT_FAILURE);
            }
        }

        baseline7<<<dimGrid2dblocktiling,64>>> (dA, dB, dC, j, k, l);
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
            baseline5<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            baseline5<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
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
            baseline5coarse<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            baseline5coarse<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
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
            baseline6<<<dim3(ceil(size/64.0f), ceil(size/64.0f)),512>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            baseline6<<<dim3(ceil(size/64.0f), ceil(size/64.0f)),512>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for 1d blocktiled kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(repeat*1000), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "1d blocktiled", (repeat*flops*1e-6)/time, time/20);
        fflush(stdout);

        for (int d=0; d<3; d++){
            baseline7<<<dimGrid2dblocktiling,64>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(start);
        for (int d=0; d<repeat; d++){
            baseline7<<<dimGrid2dblocktiling,64>>> (dA, dB, dC, j, k, l);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(start);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);
        printf(
            "Average time for 2d blocktiled kernel: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
            "(%ld).\n",
            time/(repeat*1000), 
            (repeat*flops*1e-6)/time, l);
        fprintf(csv, "%ld,%s,%.1f,%.3f\n", j, "2d blocktiled", (repeat*flops*1e-6)/time, time/20);
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