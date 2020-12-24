#include "CudaVec.h"
#include "cuda.h"

DeviceVec::DeviceVec() : v(NULL)
{}

cudaError_t DeviceVec::malloc(size_t size) {
	return cudaMalloc((void**)&v, size);
}


DeviceVec::~DeviceVec() {
	if (v != NULL) {
    	cudaFree(v);
	}
}






// PinnedVec::PinnedVec() {

// }

// PinnedVec::~PinnedVec() {
// 	if (_host != NULL) {
// 		cudaFree(_host);
// 	}
// }