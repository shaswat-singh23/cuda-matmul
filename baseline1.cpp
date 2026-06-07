#include <chrono>
#include <cstdio>
void matmul(float* A, float* B, float* C, int N){
    for (int i=0; i<N;i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                C[i*N+j]+=A[i*N+k]*B[k*N+j];
            }
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
    delete[] A;
    delete[] B;
    delete[] C;

}


