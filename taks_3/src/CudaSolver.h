#pragma once
#include "Mat3D.h"
#include "cuda_utils/CudaVec.h"
#include "cuda_utils/Stream.h"
#include "cuda_utils/Event.h"

class CudaSolver
{
public:
	void setL(double new_L[]);
	void seth(double new_h[]);
	void setTau(double new_tau);
	void setNmin(int new_Nmin[]);
	void seta_t(double new_a_t);
	void setBlockSize(int I, int J, int K);
	void mallocResources(Stream &stream, int rank);

	void fillU0(Mat3D &block, cudaStream_t stream);
	void fillU1(const Mat3D &block0, Mat3D &block1, cudaStream_t stream);
	void step(const Mat3D &block0, const Mat3D &block1, Mat3D &block2, cudaStream_t stream);
	void reduceErr(const Mat3D &block, double u_t, Stream &stream, const Event& wait_event);
	double getErr();

	void setZeroSlice(Mat3D &block, int ind, int axis, cudaStream_t stream);
	void getSlice(const Mat3D &block, DeviceVec& slice, int ind, int axis, cudaStream_t stream);
	void setSlice(Mat3D &block, const DeviceVec& slice, int ind, int axis, cudaStream_t stream);
private:
	double _L[3];
	double _h[3];
	double _tau;
	int _Nmin[3]; //start index in main grid for process
	double _a_t;
	double _t;
	int Nsize[3];
	DeviceVec buf[2];
	HostVec reduce_res;
};
