#include "CudaVec.h"
#include "cuda.h"
#include <sstream>
#include <iostream>


DeviceVec::DeviceVec() : v(NULL), _size(0)
{}

DeviceVec::DeviceVec(size_t size) : v(NULL), _size(size) {
    malloc(size);
}

cudaError_t DeviceVec::malloc(size_t size) {
    clear();
    _size = size;
    return cudaMalloc((void**)&v, size*sizeof(double));
}

void DeviceVec::clear() {
    if (v != NULL) {
        cudaFree(v);
    }
    v = NULL;
}

DeviceVec::~DeviceVec() {
    clear();
}




HostVec::HostVec() : v(NULL), _size(0)
{}

HostVec::HostVec(size_t size, bool locked) :
    v(NULL), _size(size), _locked(locked)
{
    malloc(size, locked);
}

cudaError_t HostVec::malloc(size_t size, bool locked) {
    clear();
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

double &HostVec::at(int i) {
    if (i >= 0 && i < _size) {
        return v[i];
    } else {
        std::stringstream s;
        s << "Error in HostVec::at : index " << i << " is out of range [0, " << _size << ")";
        std::cout << s.str() << std::endl;
        throw s.str();
    }
}


void HostVec::clear() {
    if (v != NULL) {
        if (_locked) {
            cudaFree(v);
        } else {
            delete[] v;
        }
    }
    v = NULL;
}

HostVec::~HostVec() {
    clear();
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