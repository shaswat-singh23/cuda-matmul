#include <chrono>
#include <cstdio>
void matmul(float* A, float* B, float* C, int N){
    for (int i=0; i<N;i++){
        for (int j=0; j<N; j++){
            float sum0=0, sum1=0, sum2=0, sum3=0, rem=0;
            for (int k=0; k<N-3; k+=4){
                sum0 += A[i*N+k]*B[k*N+j];
                sum1 += A[i*N+k+1]*B[(k+1)*N+j];
                sum2 += A[i*N+k+2]*B[(k+2)*N+j];
                sum3 += A[i*N+k+3]*B[(k+3)*N+j];
            }
            for (int b = 0; b<N%4; b++) rem += A[i*N+b]*B[(b*N)+j];
            C[i*N+j]+=sum0+sum1+sum2+sum3+rem;
        }
    }
}
int main(){
    int N =1024;
    float* A = new float[N*N];
    float* B = new float[N*N];
    float* C = new float[N*N];
    //stuff
    for (int i=0; i<N*N; i++){
        A[i]=1.0f;
        B[i]=1.0f;
        C[i]=0.0f;
    } 
    auto start = std::chrono::high_resolution_clock::now();
    matmul (A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;
    printf("Time: %.4f seconds\n", elapsed.count());
    /*for (int i =0; i<N*N; i++){
        printf("%.1f ", C[i]);
    }*/
    delete[] A;
    delete[] B;
    delete[] C;
}


