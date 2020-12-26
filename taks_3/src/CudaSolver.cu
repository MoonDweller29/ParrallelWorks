#include "CudaSolver.h"
#include "cuda_utils/cuda_macro.h"
#include <cuda.h>
#include <iostream>

#define PI 3.1415926535

__constant__ int shape[3];
__constant__ double L[3];
__constant__ double a_t;
__constant__ int Nmin[3];
__constant__ double h[3];
__constant__ double tau;

__device__ double U(double x, double y, double z, double t) {
    return sin(x*3*PI/L[0])*
        sin(y*2*PI/L[1])*
        sin(z*2*PI/L[2])*
        cos(a_t*t + 4*PI);
}

__device__ int rawInd(int i, int j, int k) {
    return ((i+1)*(shape[1]+2) + (j+1))*(shape[2]+2) + (k+1);
}

__global__ void fillU0(double* block) {
    // indexing is reverted
    // due to restriction of thread count on z axis
    int i = blockIdx.z;
    int j = blockIdx.y;
    int k = threadIdx.x;

    block[rawInd(i,j,k)] = U((Nmin[0]+i)*h[0], (Nmin[1]+j)*h[1], (Nmin[2]+k)*h[2], 0);
}

__device__ double laplacian(double *block, int i, int j, int k) {
    double res = 0;
    double center = block[rawInd(i,j,k)];
    res += (block[rawInd(i-1,j,k)] - 2*center + block[rawInd(i+1,j,k)]) / (h[0]*h[0]);
    res += (block[rawInd(i,j-1,k)] - 2*center + block[rawInd(i,j+1,k)]) / (h[1]*h[1]);
    res += (block[rawInd(i,j,k-1)] - 2*center + block[rawInd(i,j,k+1)]) / (h[2]*h[2]);

    return res;
}

__global__ void fillU1(double* block0, double* block1) {
    // indexing is reverted
    // due to restriction of thread count on z axis
    int i = blockIdx.z;
    int j = blockIdx.y;
    int k = threadIdx.x;
    int raw_ind = rawInd(i,j,k);

    block1[raw_ind] = block0[raw_ind] + tau*tau*0.5*laplacian(block0, i,j,k);
}

__global__ void step(double *block0, double *block1, double *block2) {
    // indexing is reverted
    // due to restriction of thread count on z axis
    int i = blockIdx.z;
    int j = blockIdx.y;
    int k = threadIdx.x;
    int raw_ind = rawInd(i,j,k);

    block2[raw_ind] = 2*block1[raw_ind] - block0[raw_ind] + tau*tau*laplacian(block1, i,j,k);

}

void CudaSolver::setL(double new_L[]) {
    for (int i = 0; i < 3; ++i) {
        _L[i] = new_L[i];
    }
}
void CudaSolver::seth(double new_h[]) {
    for (int i = 0; i < 3; ++i) {
        _h[i] = new_h[i];
    }
}
void CudaSolver::setTau(double new_tau) {
    _tau = new_tau;
}
void CudaSolver::setNmin(int new_Nmin[]) {
    for (int i = 0; i < 3; ++i) {
        _Nmin[i] = new_Nmin[i];
    }
}
void CudaSolver::seta_t(double new_a_t) {
    _a_t = new_a_t;
}

void CudaSolver::fillU0(Mat3D &block, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block.shape(1), block.shape(0));
    dim3 blockSize = dim3(block.shape(2), 1, 1);

    block_size[0] = block.shape(0);
    block_size[1] = block.shape(1);
    block_size[2] = block.shape(2);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(block_size), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(L, static_cast<double*>(_L), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(Nmin, static_cast<int*>(_Nmin), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(h, static_cast<double*>(_h), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(a_t, &_a_t, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::fillU0<<<gridSize, blockSize, 0, stream>>>(block.device());
}

void CudaSolver::fillU1(const Mat3D &block0, Mat3D &block1, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block0.shape(1), block0.shape(0));
    dim3 blockSize = dim3(block0.shape(2), 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(block_size), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(tau, &_tau, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::fillU1<<<gridSize, blockSize, 0, stream>>>(block0.device(), block1.device());
}

void CudaSolver::step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block0.shape(1), block0.shape(0));
    dim3 blockSize = dim3(block0.shape(2), 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(block_size), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(tau, &_tau, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::step<<<gridSize, blockSize, 0, stream>>>(block0.device(), block1.device(), block2.device());
}
