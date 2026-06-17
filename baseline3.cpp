#include <chrono>
#include <cstdio>
void matmul(float* A, float* B, float* C, int N){
    int t=256; //largest b 3b^2<C (l1 cache 512 kb on my laptop) that evenly goes into 1024 matrix size
    for (int i =0; i<N; i+=t){
        for (int k =0; k<N; k+=t){
            for (int j=0; j<N; j+=t){
                //tiling with sub matrix multiplications
                for (int i1=i; i1< i+t; i1++){
                    for (int k1=k; k1<k+t; k1++){
                        float r =A[i1*N+k1];
                        for (int j1=j; j1<j+t; j1++){
                            C[i1*N+j1]+=r*B[k1*N+j1];
                        }
                    }
                }
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
    /*for (int i =0; i<N*N; i++){
        printf("%.1f ", C[i]);
    }*/
    delete[] A;
    delete[] B;
    delete[] C;
}


