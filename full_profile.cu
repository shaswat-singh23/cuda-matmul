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

__global__ void baseline8 (float* A, float* B, float* C, int j, int k, int l){
    __shared__ float Atile[blockj*blockk];
    __shared__ float Btile[blockk][blockl];
    A += blockIdx.y*k*blockj;
    B += blockIdx.x*blockl;
    C += blockIdx.y*blockj*l + blockIdx.x*blockl;
    const int t = threadIdx.x;

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
    float *A, *B, *C, *C_ref;
    A = new float[j*k];
    B = new float[k*l];
    C = new float[j*l];
    C_ref = new float[j*l];

    for (int i = 0; i<j*k; i++) A[i]=(float)rand()/RAND_MAX;
    for (int i = 0; i<k*l; i++) B[i]=(float)rand()/RAND_MAX;
    float *dA, *dB, *dC, *dC_ref;
    CUDA_CHECK(cudaMalloc((void**)&dA, j*k*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, k*l*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, j*l*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC_ref, j*l*sizeof(float)));
    cudaMemcpy(dA, A, j*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, k*l*sizeof(float), cudaMemcpyHostToDevice);


    dim3 dimBlock(32,32);
    dim3 dimGridCoarse(ceil(l/(32.0f*coarse)),ceil(j/32.0f));
    dim3 dimGridTiled(ceil(l/32.0f),ceil(j/32.0f));
    dim3 dimGrid2dblocktiling(ceil(l/128.0f),ceil(j/128.0f));
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    baseline4<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j);
    baseline5<<<dimGridTiled,dimBlock>>> (dA, dB, dC, j, k, l);
    baseline5coarse<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
    baseline6<<<dimGridCoarse,dimBlock>>> (dA, dB, dC, j, k, l);
    baseline7<<<dimGrid2dblocktiling,256>>> (dA, dB, dC, j, k, l);
    baseline8<<<dimGrid2dblocktiling,256>>> (dA, dB, dC, j, k, l);

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        l, j, k,          // dimensions (swapped)
        &alpha,
        dB, l,           // B first, leading dim l
        dA, k,           // A second, leading dim k
        &beta,
        dC_ref, l);
    cudaMemcpy(C_ref, dC_ref, j*l*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, dC, j*l*sizeof(float), cudaMemcpyDeviceToHost);

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