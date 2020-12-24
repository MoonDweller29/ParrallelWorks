#include "CudaVec.h"
#include "cuda.h"

DeviceVec::DeviceVec() : v(NULL)
{}

cudaError_t DeviceVec::malloc(size_t size) {
    return cudaMalloc((void**)&v, size*sizeof(float));
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
        return cudaMallocHost((void**)&v, size*sizeof(float));
    } else {
        v = new float[_size];
        return cudaSuccess;
    }
}

float &HostVec::operator[](int i) {
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


// PinnedVec::PinnedVec() :
//     _host(NULL), _device(NULL), _size(NULL)
// {}

// PinnedVec::~PinnedVec() {
//     if (_host != NULL) {
//         cudaFree(_host);
//     }
// }