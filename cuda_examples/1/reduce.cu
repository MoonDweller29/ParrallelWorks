#include "reduce.h"
#include <iostream>

__constant__ float fill_value;

__global__ void fillBuf(float* buf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  buf[idx] = fill_value;
}

__global__ void fast_max_reduce(float *arr, float *out_arr) {
	extern __shared__ float sdata[];

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

__global__ void slow_max_reduce(float *arr, float *out_arr) {
	extern __shared__ float sdata[];

    int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = arr[i];
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

int round_up_to_next_pow2(unsigned int x) {
	x--;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	x++;

	return x;
}

float max_reduce(HostVec& host_arr, const Stream& stream) {
	float res = -1;
	size_t size = round_up_to_next_pow2(host_arr.size());

	DeviceVec device_arr;
	device_arr.malloc(size);

	cudaMemcpy(device_arr.data(), host_arr.data(), sizeof(float)*host_arr.size(), cudaMemcpyHostToDevice);
	dim3 gridSize = dim3(1, 1, 1);
    dim3 blockSize = dim3(size - host_arr.size(), 1, 1);

    float float_min = -1e7;
    cudaMemcpyToSymbolAsync(fill_value, &float_min, sizeof(float), 0, cudaMemcpyHostToDevice, *stream);
    fillBuf<<<gridSize, blockSize, 0, *stream>>>(&(device_arr.data()[host_arr.size()]));
	
	DeviceVec buf[2];
	buf[0].malloc(size/4);
	buf[1].malloc(size/4);

	blockSize = dim3(size/2, 1, 1);


    fast_max_reduce<<<gridSize, blockSize, blockSize.x*sizeof(float), *stream>>>(device_arr.data(), buf[0].data());
	stream.synchronize();
	cudaMemcpy(&res, buf[0].data(), sizeof(float), cudaMemcpyDeviceToHost);

	return res;
}
