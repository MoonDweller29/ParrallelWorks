#include "kernels.h"
#include "cuda_utils/cuda_macro.h"
#include <cuda.h>
#include <iostream>

#define PI 3.1415926535

__constant__ int shape[3];
__constant__ double L[3];
__constant__ double a_t;
__constant__ int Nmin[3];
__constant__ double h[3];

__device__ double U(double x, double y, double z, double t) {
    return sin(x*3*PI/L[0])*
        sin(y*2*PI/L[1])*
        sin(z*2*PI/L[2])*
        cos(a_t*t + 4*PI);
}

__device__ int rawInd(int i, int j, int k) {
    return ((i+1)*(shape[1]+2) + (j+1))*(shape[2]+2) + (k+1);
}

__global__ void fillU0(double* block)
{
    // indexing is reverted
    // due to restriction of thread count on z axis
    int i = blockIdx.z;
    int j = blockIdx.y;
    int k = threadIdx.x;

    block[rawInd(i,j,k)] = U((Nmin[0]+i)*h[0], (Nmin[1]+j)*h[1], (Nmin[2]+k)*h[2], 0);
}

void fillU0(Mat3D &block,
    double L_host[], int Nmin_host[], double h_host[], double a_t_host,
    cudaStream_t stream
) {
    dim3 gridSize = dim3(1, block.shape(1), block.shape(0));
    dim3 blockSize = dim3(block.shape(2), 1, 1);

    int dim_size[] = { block.shape(0), block.shape(1), block.shape(2) };

    SAFE_CALL( cudaMemcpyToSymbol(shape, static_cast<int*>(dim_size), 3*sizeof(int)) )
    SAFE_CALL( cudaMemcpyToSymbol(L, static_cast<double*>(L_host), 3*sizeof(double)) )
    SAFE_CALL( cudaMemcpyToSymbol(Nmin, static_cast<int*>(Nmin_host), 3*sizeof(int)) )
    SAFE_CALL( cudaMemcpyToSymbol(h, static_cast<double*>(h_host), 3*sizeof(double)) )
    SAFE_CALL( cudaMemcpyToSymbol(a_t, &a_t_host, sizeof(double)) )

    fillU0<<<gridSize, blockSize, 0, stream>>>(block.device());
}
