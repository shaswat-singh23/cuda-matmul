#include <stdio.h>
#include <chrono>
#include <iostream>
#include <stdlib.h>
__global__ void matmulkernel(float* A, float* B, float* C, int width) {
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

void matmul(float*A,float*B,float*C,int size){
    float *A_d, *B_d, *C_d;
    int bsize = size*size*sizeof(float);
    cudaMalloc((void**)&A_d, bsize);
    cudaMalloc((void**)&B_d, bsize);
    cudaMalloc((void**)&C_d, bsize);
    cudaMemcpy(A_d, A, bsize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, bsize, cudaMemcpyHostToDevice);
    dim3 dimGrid(ceil(float(size)/16),ceil(float(size)/16));
    dim3 dimBlock(16,16);
    matmulkernel<<<dimGrid, dimBlock >>>(A_d,B_d,C_d,size);
    cudaMemcpy(C, C_d, bsize, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N =1024;
    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];
    //stuff
    for (int i=0; i<N*N; i++){
        A[i]=(float)rand()/RAND_MAX;
        B[i]=(float)rand()/RAND_MAX;
        C[i]=0.0f;
    } 
    auto start = std::chrono::high_resolution_clock::now();
    matmul (A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;
    printf("Time: %.4f seconds\n", elapsed.count());

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}