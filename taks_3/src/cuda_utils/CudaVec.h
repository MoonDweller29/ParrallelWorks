#pragma once

class DeviceVec {
public:
	DeviceVec();
	~DeviceVec();

	double* data() const { return v; }
	cudaError_t malloc(size_t size);
private:
	double* v = NULL;
};

class HostVec {
public:
	HostVec();
	~HostVec();

	cudaError_t malloc(size_t size, bool locked = false);
	double *data() const { return v; }
	double &operator[](int i);

private:
	double* v = NULL;
	size_t _size = 0;
	bool _locked = false;
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
	double* _host = NULL;
	double* _device = NULL;
	size_t _size = 0;
};