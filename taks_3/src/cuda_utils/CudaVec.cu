#include "CudaVec.h"
#include "cuda.h"

DeviceVec::DeviceVec() : v(NULL)
{}

cudaError_t DeviceVec::malloc(size_t size) {
    return cudaMalloc((void**)&v, size*sizeof(double));
}


DeviceVec::~DeviceVec() {
    if (v != NULL) {
        cudaFree(v);
    }
}

HostVec::HostVec() : v(NULL), _size(0)
{}

cudaError_t HostVec::malloc(size_t size, bool locked) {
    _size = size;
    _locked = locked;
    if (locked) {
        return cudaMallocHost((void**)&v, size*sizeof(double));
    } else {
        v = new double[_size];
        return cudaSuccess;
    }
}

double &HostVec::operator[](int i) {
    return v[i];
}

HostVec::~HostVec() {
    if (v != NULL) {
        if (_locked) {
            cudaFree(v);
        } else {
            delete[] v;
        }
    }   
}



PinnedVec::PinnedVec() :
    _host(NULL), _device(NULL), _size(0)
{}

cudaError_t PinnedVec::malloc(size_t size) {
    _size = size;
    cudaError_t err = cudaHostAlloc((void**)&_host, size*sizeof(double), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&_device, _host, 0);

    return err;
}

double &PinnedVec::operator[](int i) {
    return _host[i];
}

PinnedVec::~PinnedVec() {
    if (_host != NULL) {
        cudaFree(_host);
    }
}