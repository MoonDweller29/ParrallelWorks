#pragma once

class DeviceVec {
public:
	DeviceVec();
	~DeviceVec();

	float* data() const { return v; }
	cudaError_t malloc(size_t size);
private:
	float* v = NULL;
};


// class PinnedVec {
// public:
// 	PinnedVec();
// 	~PinnedVec();

// 	float* host()   const { return _host; }
// 	float* device() const { return _device; }
// private:
// 	float* _host = NULL;
// 	float* _device = NULL;
// 	size_t _size = 0;
// }