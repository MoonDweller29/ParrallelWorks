#pragma once
#include "cuda_utils/CudaVec.h"
#include <vector>

//padded with border width 1 by default
class Mat3D
{
public:
	Mat3D(int I, int J, int K);
	~Mat3D(){};

	double *device() const { return device_data.data(); }
	void toCPU();
	void toGPU();

	double &operator()(int i, int j, int k);
	const double &operator()(int i, int j, int k) const;
	const int &shape(int i) const;
	void slice(int ind, int axis, HostVec &out_slice) const;
	int sliceLen(int axis) const;

	void setSlice(int ind, int axis, HostVec &other_slice);
	void setZeroSlice(int ind, int axis);
	void fill(double value, cudaStream_t stream = 0);

	void print(bool padded = false) const;
	void saveToCSV(const char* filename) const;
	void save(const char* filename) const;

private:
	const int _pad;
	int _I, _J, _K;
	int _rI, _rJ, _rK; //real data size
	int _shape[3];
	HostVec host_data;
	DeviceVec device_data;
};