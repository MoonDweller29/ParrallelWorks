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

class HostVec {
public:
	HostVec();
	HostVec(size_t size, bool locked = false);
	~HostVec();

	cudaError_t malloc(size_t size, bool locked = false);
	float *data() const { return v; }
	size_t size() const { return _size; }
	float &operator[](int i);

private:
	float* v = NULL;
	size_t _size = 0;
	bool _locked = false;

	void clear();
};

class PinnedVec {
public:
	PinnedVec();
	~PinnedVec();

	cudaError_t malloc(size_t size);
	float &operator[](int i);

	float* host()   const { return _host; }
	float* device() const { return _device; }
private:
	float* _host = NULL;
	float* _device = NULL;
	size_t _size = 0;
};