#pragma once
#include <cuda.h>

class DeviceVec {
public:
	DeviceVec();
	DeviceVec(size_t size);
	~DeviceVec();

	double* data() const { return v; }
	cudaError_t malloc(size_t size);
	void fill(double value, cudaStream_t stream = 0);

private:
	size_t _size;
	double* v;

	void clear();
};

class HostVec {
public:
	HostVec();
	HostVec(size_t size, bool locked = false);
	~HostVec();

	cudaError_t malloc(size_t size, bool locked = false);
	void fill(double value);
	
	double *data() const { return v; }
	size_t size() const { return _size; }
	
	double &operator[](int i);
	double &at(int i);

private:
	double* v;
	size_t _size;
	bool _locked;

	void clear();
};

class PinnedVec {
public:
	PinnedVec();
	~PinnedVec();

	cudaError_t malloc(size_t size);
	double &operator[](int i);

	double* host()   const { return _host; }
	double* device() const { return _device; }
private:
	double* _host;
	double* _device;
	size_t _size;
};