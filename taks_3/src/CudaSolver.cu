#include "CudaSolver.h"
#include "cuda_utils/cuda_macro.h"
#include <cuda.h>
#include <iostream>
#include <fstream>

#define PI 3.1415926535
#define MIN_VAL -1e10

__constant__ int shape[3];
__constant__ double L[3];
__constant__ double a_t;
__constant__ int Nmin[3];
__constant__ double h[3];
__constant__ double tau;
__constant__ double t_const;

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

__global__ void fillMinVal(double* buf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    buf[idx] = MIN_VAL;
}

__device__ double calcErr(double value, int i, int j, int k) {
    double u_val = U((Nmin[0]+i)*h[0], (Nmin[1]+j)*h[1], (Nmin[2]+k)*h[2], t_const);
    return fabs(u_val - value);
}

//calculates error over original block
__global__ void reduce_err(double *block, double *out_arr) {
    extern __shared__ double sdata[];

    // indexing is reverted
    // due to restriction of thread count on z axis
    int i = blockIdx.z;
    int j = blockIdx.y;
    int k = threadIdx.x;

    int tid = threadIdx.x;
    unsigned int raw_ind = rawInd(i,j,k);

    sdata[tid] = calcErr(block[raw_ind], i, j, k);
    if (k + blockDim.x < shape[2]) {
        sdata[tid] = fmax(sdata[tid], 
            calcErr(block[raw_ind+blockDim.x], i, j, k+blockDim.x));
    }
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_arr[gridDim.y*i + j] = sdata[0];
    }
}


//assumes that input arr size is a power of 2
__global__ void max_reduce(double *arr, double *out_arr) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = fmax(arr[i], arr[i+blockDim.x]);
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_arr[blockIdx.x] = sdata[0];
    }
}

__global__ void set_zero_slice(double *block, int ind, int axis) {
    int coord[3];
    coord[axis] = ind;
    coord[(axis+1) % 3] = blockIdx.x;
    coord[(axis+2) % 3] = threadIdx.x;
    int raw_ind = rawInd(coord[0], coord[1], coord[2]);

    block[raw_ind] = 0;
}

__global__ void get_slice(double *block, double *slice, int ind, int axis) {
    int coord[3];
    coord[axis] = ind;
    coord[(axis+1) % 3] = blockIdx.x;
    coord[(axis+2) % 3] = threadIdx.x;
    int raw_ind = rawInd(coord[0], coord[1], coord[2]);

    slice[blockIdx.x*blockDim.x + threadIdx.x] = block[raw_ind];
}

__global__ void set_slice(double *block, double *slice, int ind, int axis) {
    int coord[3];
    coord[axis] = ind;
    coord[(axis+1) % 3] = blockIdx.x;
    coord[(axis+2) % 3] = threadIdx.x;
    int raw_ind = rawInd(coord[0], coord[1], coord[2]);

    block[raw_ind] = slice[blockIdx.x*blockDim.x + threadIdx.x];
}

static int round_up_to_next_pow2(unsigned int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
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
void CudaSolver::setBlockSize(int I, int J, int K) {
    Nsize[0] = I;
    Nsize[1] = J;
    Nsize[2] = K;
}
void CudaSolver::mallocResources(Stream &stream, int rank) {
    int buf0_size = round_up_to_next_pow2(Nsize[0]*Nsize[1]);
    buf[0].malloc(buf0_size);
    buf[1].malloc(buf0_size/(1024*2));
    dim3 gridSize = dim3(buf0_size/1024, 1, 1);
    dim3 blockSize = dim3(1024, 1, 1);
    fillMinVal<<<gridSize, blockSize, 0, *stream>>>(buf[0].data());
    reduce_res.malloc(1, true);

    // stream.synchronize();
    // HostVec cpu_dump(buf[0].size());
    // SAFE_CALL( cudaMemcpy(cpu_dump.data(), buf[0].data(), sizeof(double)*buf[0].size(), cudaMemcpyDeviceToHost) )

    // if (rank == 0) {
    //     std::cout << "DUMP\n";
    //     std::ofstream myfile;
    //     myfile.open ("orig_data.txt");
    //     for (int i = 0; i < cpu_dump.size(); ++i) {
    //         myfile << i <<": "<< cpu_dump[i] << std::endl;
    //     }
    // }
}


void CudaSolver::fillU0(Mat3D &block, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block.shape(1), block.shape(0));
    dim3 blockSize = dim3(block.shape(2), 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(Nsize), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(L, static_cast<double*>(_L), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(Nmin, static_cast<int*>(_Nmin), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(h, static_cast<double*>(_h), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(a_t, &_a_t, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::fillU0<<<gridSize, blockSize, 0, stream>>>(block.device());
}

void CudaSolver::fillU1(const Mat3D &block0, Mat3D &block1, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block0.shape(1), block0.shape(0));
    dim3 blockSize = dim3(block0.shape(2), 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(Nsize), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(tau, &_tau, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::fillU1<<<gridSize, blockSize, 0, stream>>>(block0.device(), block1.device());
}

void CudaSolver::step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2, cudaStream_t stream) {
    dim3 gridSize = dim3(1, block0.shape(1), block0.shape(0));
    dim3 blockSize = dim3(block0.shape(2), 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(Nsize), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(tau, &_tau, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    ::step<<<gridSize, blockSize, 0, stream>>>(block0.device(), block1.device(), block2.device());
}

void CudaSolver::reduceErr(const Mat3D &block, double u_t, cudaStream_t stream, int rank) {
    _t = u_t;

    dim3 gridSize = dim3(1, block.shape(1), block.shape(0));
    int block_len = round_up_to_next_pow2(block.shape(2));
    dim3 blockSize = dim3(block_len/2, 1, 1);

    SAFE_CALL( cudaMemcpyToSymbolAsync(shape, static_cast<int*>(Nsize), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(L, static_cast<double*>(_L), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(Nmin, static_cast<int*>(_Nmin), 3*sizeof(int), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(h, static_cast<double*>(_h), 3*sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(a_t, &_a_t, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )
    SAFE_CALL( cudaMemcpyToSymbolAsync(t_const, &_t, sizeof(double), 0, cudaMemcpyHostToDevice, stream) )

    reduce_err<<<gridSize, blockSize, blockSize.x*sizeof(double), stream>>>(block.device(), buf[0].data());
    
    int elem_count = buf[0].size()/(1024*2);
    gridSize = dim3(elem_count, 1, 1);
    blockSize = dim3(1024, 1, 1);
    max_reduce<<<gridSize, blockSize, blockSize.x*sizeof(double), stream>>>(buf[0].data(), buf[1].data());

    gridSize = dim3(1, 1, 1);
    blockSize = dim3(elem_count/2, 1, 1);
    max_reduce<<<gridSize, blockSize, blockSize.x*sizeof(double), stream>>>(buf[1].data(), buf[0].data());
    SAFE_CALL( cudaMemcpyAsync(reduce_res.data(), buf[0].data(), sizeof(double), cudaMemcpyDeviceToHost, stream) )
}

double CudaSolver::getErr() {
    return reduce_res.data()[0];
}

void CudaSolver::setZeroSlice(Mat3D &block, int ind, int axis, cudaStream_t stream) {
    if(axis < 0 || axis > 2) {
        std::cerr << "wrong axis " << axis << std::endl;
    }

    dim3 gridSize = dim3(Nsize[(axis+1)%3], 1, 1);
    dim3 blockSize = dim3(Nsize[(axis+2)%3], 1, 1);
    set_zero_slice<<<gridSize, blockSize, 0, stream>>>(block.device(), ind, axis);
}

void CudaSolver::getSlice(const Mat3D &block, DeviceVec& slice, int ind, int axis, cudaStream_t stream) {
    if(axis < 0 || axis > 2) {
        std::cerr << "wrong axis " << axis << std::endl;
    }

    dim3 gridSize = dim3(Nsize[(axis+1)%3], 1, 1);
    dim3 blockSize = dim3(Nsize[(axis+2)%3], 1, 1);
    get_slice<<<gridSize, blockSize, 0, stream>>>(block.device(), slice.data(), ind, axis);
}
void CudaSolver::setSlice(Mat3D &block, const DeviceVec& slice, int ind, int axis, cudaStream_t stream) {
    if(axis < 0 || axis > 2) {
        std::cerr << "wrong axis " << axis << std::endl;
    }

    dim3 gridSize = dim3(Nsize[(axis+1)%3], 1, 1);
    dim3 blockSize = dim3(Nsize[(axis+2)%3], 1, 1);
    set_slice<<<gridSize, blockSize, 0, stream>>>(block.device(), slice.data(), ind, axis);
}
