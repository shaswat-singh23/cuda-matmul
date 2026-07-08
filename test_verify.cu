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

#define blockj 128
#define blockk 8
#define blockl 128
#define colsize 8
#define rowsize 8
__global__ void blocktiling2d (const float* A, const float* B, float* C, int j, int k, int l){
    __shared__ float Atile[blockj][blockk+1];
    __shared__ float Btile[blockk][blockl+1];
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
    dim3 dimGrid2d(ceil(l/128.0f),ceil(j/128.0f));

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
    //tiledGeneralWithCoarsening<<<dimGridCoarse,dimBlock>>> (A_D, B_D, C_D, j, k, l);    
    baseline8<<<dimGrid2d,256>>>(A_D, B_D, C_D, j, k, l);
    //sgemmVectorize<<<dimGrid2d,256>>>(j, k, l, 1.0, A_D, B_D, 0.0, C_D);
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
    int j = 1024; //a row and c row
    int k = 768; //a col and b row
    int l = 512; //b col and c col
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
}